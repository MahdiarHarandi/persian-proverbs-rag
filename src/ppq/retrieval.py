"""Retrieval implementations used by the project pipeline.

Meaning mode uses character 3-gram TF-IDF over project-authored family glosses
plus conservative BM25 re-ranking. Surface mode combines five conservative surface
signals:

* compact-exact equality for punctuation/spacing/ZWNJ-only differences;
* compact substring matching for informative fragments of a stored FFE;
* character 3-gram TF-IDF, augmented by asymmetric *query coverage*;
* token-level BM25 as a complementary word-order-robust lexical signal;
* bounded fuzzy re-scoring for edit-like spelling corruption.

Query coverage answers a different question from cosine similarity: instead of
asking how similar two complete strings are, it asks how much of the user's
query is supported by a candidate. This matters for incomplete quotations,
small wording changes, and colloquial forms where the correct stored FFE may be
longer than the input. Coverage is computed on compact character bigrams and is
IDF-weighted so common fragments contribute less than distinctive ones.

The coverage, BM25, and fuzzy signals are deliberately conservative:

* it is disabled for compact queries shorter than
  ``MIN_COVERAGE_COMPACT_CHARS``;
* it only re-scores documents already retrieved by the character 3-gram index,
  so it does not introduce a new candidate pool;
* coverage only re-scores trigram candidates and BM25/fuzzy contributions are
  bounded so they cannot override the stronger exact/partial evidence scores;
* fuzzy matching inspects only the strongest lexical candidates and therefore
  cannot create a confident answer from fuzzy similarity alone.

As before, the retriever returns identifiers and scores only. Displayable text
is resolved later by :mod:`ppq.corpus`; retrieval-normalized strings are never
rendered as quotations.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from .corpus import Corpus
from .models import Candidate, SearchMode
from .normalize import char_ngrams, normalize_compact, retrieval_tokens


class CharNgramIndex:
    """A compact TF-IDF index over character n-grams."""

    def __init__(self, documents: list[str], n: int = 3):
        if not documents:
            raise ValueError("the index needs at least one document")
        self.n = n
        self._document_count = len(documents)
        self._postings: dict[str, set[int]] = defaultdict(set)

        document_frequencies: Counter[str] = Counter()
        term_frequencies: list[Counter[str]] = []
        for index, document in enumerate(documents):
            frequencies = Counter(char_ngrams(document, n))
            term_frequencies.append(frequencies)
            for gram in frequencies:
                document_frequencies[gram] += 1
                self._postings[gram].add(index)

        self._idf = {
            gram: math.log(1.0 + self._document_count / (1.0 + frequency))
            for gram, frequency in document_frequencies.items()
        }
        self._vectors: list[dict[str, float]] = []
        self._norms: list[float] = []
        for frequencies in term_frequencies:
            vector = {gram: count * self._idf[gram] for gram, count in frequencies.items()}
            self._vectors.append(vector)
            self._norms.append(math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0)

    def search(self, query: str) -> list[tuple[int, float]]:
        frequencies = Counter(char_ngrams(query, self.n))
        query_vector = {
            gram: count * self._idf[gram]
            for gram, count in frequencies.items()
            if gram in self._idf
        }
        if not query_vector:
            return []

        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        document_ids: set[int] = set()
        for gram in query_vector:
            document_ids.update(self._postings[gram])

        # Accumulate dot products directly through the inverted postings.
        # This is algebraically identical to scanning every candidate document
        # across every query gram, but avoids repeated dictionary lookups for
        # absent grams. It materially reduces exhaustive-evaluation time while
        # preserving the exact TF-IDF cosine definition.
        dot_products: dict[int, float] = defaultdict(float)
        for gram, query_weight in query_vector.items():
            for document_id in self._postings[gram]:
                dot_products[document_id] += query_weight * self._vectors[document_id][gram]

        scored = [
            (
                document_id,
                dot_product / (query_norm * self._norms[document_id]),
            )
            for document_id, dot_product in dot_products.items()
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0]))


class ExactPartialSurfaceIndex:
    """Compact exact/substring matcher for stored surface variants.

    ``compact`` normalization removes whitespace boundaries after Persian
    character folding, so ``سرگذشت`` and ``سر گذشت`` are comparable without
    rewriting the stored quotation. Exact compact matches are strongest.
    Informative query fragments are allowed to match inside a longer stored
    variant. Very short fragments are ignored because they are commonly
    shared across unrelated FFEs.
    """

    EXACT_SCORE = 1.0
    PARTIAL_SCORE = 0.99
    MIN_PARTIAL_COMPACT_CHARS = 8

    def __init__(self, documents: list[str]):
        if not documents:
            raise ValueError("the index needs at least one document")
        self._compact_documents = [normalize_compact(document) for document in documents]
        self._exact: dict[str, list[int]] = defaultdict(list)
        for document_id, compact in enumerate(self._compact_documents):
            if compact:
                self._exact[compact].append(document_id)

    def search(self, query: str) -> list[tuple[int, float]]:
        compact_query = normalize_compact(query)
        if not compact_query:
            return []

        exact_ids = self._exact.get(compact_query, ())
        if exact_ids:
            # Do not simultaneously treat an exact stored FFE as a partial of
            # longer expressions. Exact corpus membership is stronger and
            # should not become ambiguous merely because another proverb
            # happens to contain the same phrase.
            return [(document_id, self.EXACT_SCORE) for document_id in exact_ids]

        if len(compact_query) < self.MIN_PARTIAL_COMPACT_CHARS:
            return []

        # A fixed partial evidence score is intentional. If the same fragment
        # occurs in multiple families, they tie and the downstream margin
        # verifier will CLARIFY instead of using document length as a hidden
        # tie-breaker.
        return [
            (document_id, self.PARTIAL_SCORE)
            for document_id, compact_document in enumerate(self._compact_documents)
            if compact_query in compact_document
        ]


class QueryCoverageSurfaceIndex:
    """Asymmetric lexical coverage over compact character bigrams.

    Cosine similarity penalizes a long stored expression when the user only
    remembers a fragment. Coverage instead measures the fraction of the
    *query's* distinctive character bigrams that occur in a candidate.

    Unknown query bigrams remain in the denominator with a maximal IDF-like
    weight. This is important: a typo should reduce coverage rather than being
    silently ignored just because that misspelled bigram never appeared in the
    corpus.
    """

    NGRAM_N = 2
    MIN_COVERAGE_COMPACT_CHARS = 8
    COVERAGE_WEIGHT = 0.30

    def __init__(self, documents: list[str]):
        if not documents:
            raise ValueError("the index needs at least one document")
        self._document_count = len(documents)
        self._document_grams: list[set[str]] = []
        self._postings: dict[str, set[int]] = defaultdict(set)
        document_frequencies: Counter[str] = Counter()

        for document_id, document in enumerate(documents):
            compact = normalize_compact(document)
            grams = self._grams(compact)
            self._document_grams.append(grams)
            document_frequencies.update(grams)
            for gram in grams:
                self._postings[gram].add(document_id)

        self._idf = {
            gram: math.log(1.0 + self._document_count / (1.0 + frequency))
            for gram, frequency in document_frequencies.items()
        }
        # OOV query grams must count as uncovered evidence. Using the largest
        # plausible IDF prevents misspellings from disappearing from the
        # denominator.
        self._oov_idf = math.log(1.0 + self._document_count)

    @classmethod
    def _grams(cls, compact: str) -> set[str]:
        if not compact:
            return set()
        if len(compact) < cls.NGRAM_N:
            return {compact}
        return {
            compact[index : index + cls.NGRAM_N] for index in range(len(compact) - cls.NGRAM_N + 1)
        }

    def _prepare_query(self, query: str) -> tuple[set[str], float]:
        compact_query = normalize_compact(query)
        if len(compact_query) < self.MIN_COVERAGE_COMPACT_CHARS:
            return set(), 0.0
        query_grams = self._grams(compact_query)
        if not query_grams:
            return set(), 0.0
        denominator = sum(self._idf.get(gram, self._oov_idf) for gram in query_grams)
        return query_grams, denominator

    def _coverage_from_prepared(
        self,
        query_grams: set[str],
        denominator: float,
        document_id: int,
    ) -> float:
        if not query_grams or denominator <= 0.0:
            return 0.0
        matched = query_grams & self._document_grams[document_id]
        numerator = sum(self._idf.get(gram, self._oov_idf) for gram in matched)
        return numerator / denominator

    def coverage(self, query: str, document_id: int) -> float:
        query_grams, denominator = self._prepare_query(query)
        return self._coverage_from_prepared(query_grams, denominator, document_id)

    def rescore(
        self,
        query: str,
        ranking: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Add a bounded coverage bonus to an existing lexical ranking.

        ``base + w * coverage * (1-base)`` keeps scores in ``[0, 1]`` and
        leaves a perfect base score unchanged. Coverage cannot create a new
        candidate here; it only supplies additional evidence for candidates
        already found by the trigram index.
        """
        query_grams, denominator = self._prepare_query(query)
        if not query_grams:
            return ranking

        # Accumulate only the query-gram evidence that actually occurs in the
        # corpus. This inverted-index pass is substantially cheaper than
        # intersecting the query set with every retrieved document separately.
        numerators: dict[int, float] = defaultdict(float)
        for gram in query_grams:
            weight = self._idf.get(gram, self._oov_idf)
            for document_id in self._postings.get(gram, ()):
                numerators[document_id] += weight

        rescored = []
        for document_id, base_score in ranking:
            coverage = numerators.get(document_id, 0.0) / denominator
            score = base_score + self.COVERAGE_WEIGHT * coverage * (1.0 - base_score)
            rescored.append((document_id, score))
        return sorted(rescored, key=lambda item: (-item[1], item[0]))


class BM25SurfaceIndex:
    """Dependency-free BM25 lexical index for surface quotations.

    Character n-grams are strong for local spelling similarity, but they are
    sensitive to word order and can under-rank a quotation when a user only
    remembers a bag of important words. BM25 adds a complementary token-level
    signal over retrieval-normalized whitespace tokens.

    ``search`` returns a bounded evidence value in ``[0, 1]`` rather than raw
    BM25 so it can be combined conservatively with the existing score. The
    normalization denominator is the summed IDF of *known* query terms; OOV
    wrapper words therefore do not erase useful lexical evidence. A candidate
    must match at least ``MIN_MATCHED_TERMS_FOR_BONUS`` distinct query terms to
    receive a score boost. This prevents a single common word from making an
    unrelated proverb look confident.
    """

    K1 = 1.2
    B = 0.75
    BM25_WEIGHT = 0.35
    MIN_MATCHED_TERMS_FOR_BONUS = 2

    def __init__(self, documents: list[str]):
        if not documents:
            raise ValueError("the index needs at least one document")
        self._document_count = len(documents)
        self._term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        self._postings: dict[str, set[int]] = defaultdict(set)
        document_frequencies: Counter[str] = Counter()

        for document_id, document in enumerate(documents):
            frequencies = Counter(retrieval_tokens(document))
            self._term_frequencies.append(frequencies)
            length = sum(frequencies.values())
            self._document_lengths.append(length)
            for term in frequencies:
                document_frequencies[term] += 1
                self._postings[term].add(document_id)

        self._average_document_length = (sum(self._document_lengths) / self._document_count) or 1.0
        self._idf = {
            term: math.log(1.0 + (self._document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def _term_score(self, term: str, document_id: int) -> float:
        frequency = self._term_frequencies[document_id].get(term, 0)
        if not frequency:
            return 0.0
        length = self._document_lengths[document_id]
        denominator = frequency + self.K1 * (
            1.0 - self.B + self.B * length / self._average_document_length
        )
        return self._idf[term] * (frequency * (self.K1 + 1.0)) / denominator

    def search(self, query: str) -> list[tuple[int, float]]:
        query_terms = tuple(dict.fromkeys(retrieval_tokens(query)))
        known_terms = [term for term in query_terms if term in self._idf]
        if not known_terms:
            return []

        # Accumulate BM25 contributions through postings rather than scanning
        # each candidate against every query token. The formula and one-token
        # safety cap are unchanged; only the execution plan is faster.
        raw_scores: dict[int, float] = defaultdict(float)
        matched_counts: Counter[int] = Counter()
        for term in known_terms:
            for document_id in self._postings[term]:
                raw_scores[document_id] += self._term_score(term, document_id)
                matched_counts[document_id] += 1

        # For an average-length document containing each query term once, BM25
        # contributes approximately that term's IDF. This yields a stable,
        # interpretable evidence scale independent of the number of corpus docs.
        normalizer = sum(self._idf[term] for term in known_terms) or 1.0
        scored: list[tuple[int, float]] = []
        for document_id, raw in raw_scores.items():
            evidence = min(1.0, raw / normalizer)
            # Encode the conservative-bonus eligibility in the sign-free score:
            # one-term matches can still expand the candidate pool, but their
            # evidence is capped so they cannot materially change confidence.
            if matched_counts[document_id] < self.MIN_MATCHED_TERMS_FOR_BONUS:
                evidence = min(evidence, 0.15)
            scored.append((document_id, evidence))
        return sorted(scored, key=lambda item: (-item[1], item[0]))

    def rescore(
        self,
        base_ranking: list[tuple[int, float]],
        bm25_ranking: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Union BM25 candidates with the current ranking and add a bounded bonus.

        BM25-only documents can enter the candidate pool, but with a maximum
        score of ``BM25_WEIGHT`` they cannot independently pass the current
        Surface ACCEPT threshold. Existing candidates can receive a bounded
        lexical boost when token evidence agrees with the character evidence.
        """
        base_scores = dict(base_ranking)
        bm25_scores = dict(bm25_ranking)
        document_ids = set(base_scores) | set(bm25_scores)
        rescored: list[tuple[int, float]] = []
        for document_id in document_ids:
            base = base_scores.get(document_id, 0.0)
            evidence = bm25_scores.get(document_id, 0.0)
            score = base + self.BM25_WEIGHT * evidence * (1.0 - base)
            rescored.append((document_id, min(1.0, score)))
        return sorted(rescored, key=lambda item: (-item[1], item[0]))


class BM25GlossIndex(BM25SurfaceIndex):
    """Small BM25 re-ranker for meaning/gloss retrieval.

    Meaning retrieval is deliberately lexical and interpretable. Character
    3-gram TF-IDF remains the candidate generator; BM25 only
    re-ranks those candidates with a small calibration-selected bonus.  It
    therefore cannot introduce a BM25-only family or turn token overlap into a
    semantic claim.

    ``BONUS_WEIGHT`` is a conservative calibration value that improved
    development-set retrieval without introducing wrong/OOD accepts.
    """

    BONUS_WEIGHT = 0.05

    def rescore(
        self,
        query: str,
        base_ranking: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        if not base_ranking:
            return []
        evidence = dict(self.search(query))
        rescored = []
        for document_id, base_score in base_ranking:
            bm25 = evidence.get(document_id, 0.0)
            score = base_score + self.BONUS_WEIGHT * bm25 * (1.0 - base_score)
            rescored.append((document_id, min(1.0, score)))
        return sorted(rescored, key=lambda item: (-item[1], item[0]))


class FuzzySurfaceIndex:
    """Conservative fuzzy re-scoring for edit-like surface corruption.

    This signal is intentionally *not* a general semantic retriever.  It targets
    local character edits, colloquial spellings, token-boundary damage, and a
    few nearby orthographic corruptions after the normal lexical retrievers have
    already produced candidates.

    Keeping fuzzy matching as a re-ranker has two advantages: it is cheap on the
    2,067-variant corpus and it cannot make an unrelated fuzzy-only document
    become a confident answer.  Only the highest lexical candidates are
    inspected and the final bonus is bounded.
    """

    NGRAM_N = 2
    MIN_COMPACT_CHARS = 8
    TOKEN_SIMILARITY_THRESHOLD = 0.72
    MAX_TOKEN_LENGTH_DELTA = 2
    MAX_RESCORE_CANDIDATES = 8
    MIN_ROUGH_COVERAGE_FOR_EDIT = 0.45
    MIN_EVIDENCE = 0.55
    FUZZY_WEIGHT = 0.40
    SKIP_BASE_SCORE = 0.95

    def __init__(self, documents: list[str]):
        if not documents:
            raise ValueError("the index needs at least one document")
        self._compact_documents = [normalize_compact(document) for document in documents]
        self._token_documents = [retrieval_tokens(document) for document in documents]
        self._gram_sets = [self._grams(compact) for compact in self._compact_documents]

    @classmethod
    def _grams(cls, compact: str) -> set[str]:
        if not compact:
            return set()
        if len(compact) < cls.NGRAM_N:
            return {compact}
        return {
            compact[index : index + cls.NGRAM_N] for index in range(len(compact) - cls.NGRAM_N + 1)
        }

    @staticmethod
    def _semi_global_edit_distance(query: str, document: str) -> int:
        """Levenshtein distance from ``query`` to the best document substring."""
        if not query:
            return 0
        previous = [0] * (len(document) + 1)
        for row_index, query_char in enumerate(query, 1):
            current = [row_index] + [0] * len(document)
            for column_index, document_char in enumerate(document, 1):
                current[column_index] = min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + (query_char != document_char),
                )
            previous = current
        return min(previous)

    @staticmethod
    def _allowed_compact_edits(query_length: int) -> int:
        if query_length < 12:
            return 1
        if query_length < 24:
            return 2
        if query_length < 36:
            return 3
        return 4

    def _token_repair_evidence(
        self,
        query_tokens: tuple[str, ...],
        document_tokens: tuple[str, ...],
    ) -> tuple[float, int, tuple[str, ...]]:
        if not query_tokens or not document_tokens:
            return 0.0, 0, ()

        used_query: set[int] = set()
        used_document: set[int] = set()
        exact_matches = 0
        for query_index, query_token in enumerate(query_tokens):
            for document_index, document_token in enumerate(document_tokens):
                if document_index in used_document:
                    continue
                if query_token == document_token:
                    used_query.add(query_index)
                    used_document.add(document_index)
                    exact_matches += 1
                    break

        possible_repairs: list[tuple[float, int, int]] = []
        for query_index, query_token in enumerate(query_tokens):
            if query_index in used_query or len(query_token) < 2:
                continue
            for document_index, document_token in enumerate(document_tokens):
                if document_index in used_document or len(document_token) < 2:
                    continue
                if abs(len(query_token) - len(document_token)) > self.MAX_TOKEN_LENGTH_DELTA:
                    continue
                similarity = SequenceMatcher(
                    None, query_token, document_token, autojunk=False
                ).ratio()
                if similarity >= self.TOKEN_SIMILARITY_THRESHOLD:
                    possible_repairs.append((similarity, query_index, document_index))

        possible_repairs.sort(reverse=True)
        repairs: list[float] = []
        for similarity, query_index, document_index in possible_repairs:
            if query_index in used_query or document_index in used_document:
                continue
            used_query.add(query_index)
            used_document.add(document_index)
            repairs.append(similarity)

        unmatched_long = tuple(
            token
            for index, token in enumerate(query_tokens)
            if index not in used_query and len(token) >= 3
        )
        if not repairs:
            return 0.0, 0, unmatched_long

        covered = exact_matches + sum(repairs)
        coverage = covered / len(query_tokens)
        mean_repair = sum(repairs) / len(repairs)
        evidence = coverage * (0.80 + 0.20 * mean_repair)
        return evidence, len(repairs), unmatched_long

    def evidence(self, query: str, document_id: int) -> float:
        compact_query = normalize_compact(query)
        if len(compact_query) < self.MIN_COMPACT_CHARS:
            return 0.0

        query_tokens = retrieval_tokens(query)
        token_evidence, repair_count, unmatched_long = self._token_repair_evidence(
            query_tokens, self._token_documents[document_id]
        )
        if repair_count > 0 and not unmatched_long:
            return token_evidence

        query_grams = self._grams(compact_query)
        if not query_grams:
            return 0.0
        rough_coverage = len(query_grams & self._gram_sets[document_id]) / len(query_grams)
        if rough_coverage < self.MIN_ROUGH_COVERAGE_FOR_EDIT:
            return 0.0

        edit_distance = self._semi_global_edit_distance(
            compact_query, self._compact_documents[document_id]
        )
        allowed_edits = self._allowed_compact_edits(len(compact_query))
        compact_evidence = 0.0
        if 0 < edit_distance <= allowed_edits:
            compact_evidence = 1.0 - edit_distance / len(compact_query)

        # If a long query token is unaccounted for, token evidence alone could
        # reward a semantic word substitution.  Require the independent compact
        # edit path before retaining that token signal.
        if unmatched_long and compact_evidence <= 0.0:
            token_evidence = 0.0
        if repair_count <= 0:
            token_evidence = 0.0
        return max(compact_evidence, token_evidence)

    def rescore(
        self,
        query: str,
        base_ranking: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Re-score only the strongest lexical candidates with fuzzy evidence."""
        if len(normalize_compact(query)) < self.MIN_COMPACT_CHARS or not base_ranking:
            return base_ranking
        if base_ranking[0][1] >= 0.90:
            second = base_ranking[1][1] if len(base_ranking) > 1 else 0.0
            if base_ranking[0][1] - second >= 0.05:
                return base_ranking

        rescored = list(base_ranking)
        for index, (document_id, base_score) in enumerate(rescored[: self.MAX_RESCORE_CANDIDATES]):
            if base_score >= self.SKIP_BASE_SCORE:
                continue
            evidence = self.evidence(query, document_id)
            if evidence < self.MIN_EVIDENCE:
                continue
            score = base_score + self.FUZZY_WEIGHT * evidence * (1.0 - base_score)
            rescored[index] = (document_id, min(1.0, score))
        return sorted(rescored, key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True)
class SurfaceEvidence:
    """Inspectable evidence used by the Phase-7 surface fusion ranker.

    The values are deliberately kept on bounded ``[0, 1]`` scales.  They are
    *retrieval evidence*, not probabilities.  Keeping the components explicit
    makes the fusion auditable and lets tests/ablations verify that a confident
    answer is supported by several independent surface signals rather than one
    opaque score.
    """

    tfidf: float = 0.0
    coverage: float = 0.0
    bm25: float = 0.0
    fuzzy: float = 0.0


class SurfaceEvidenceFusion:
    """Fuse the graded surface signals with explicit consensus rules.

    Phases 4--6 added coverage, BM25 and fuzzy matching sequentially.  That
    pipeline was safe but the final score depended on the order in which the
    bonuses happened to be applied.  Phase 7 makes the combination explicit:

    * the base score is a weighted noisy-OR over TF-IDF, coverage, BM25 and
      fuzzy evidence;
    * fuzzy evidence is computed only for the strongest lexical candidates;
    * three transparent consensus patterns can raise a well-supported result
      above the selective-accept boundary;
    * exact/partial evidence remains outside this class and keeps its stronger
      categorical scores (1.00 / 0.99).

    The consensus rules are intentionally conservative.  They require either
    strong independent edit evidence or agreement between several token/char
    signals.  Generic, OOD and semantic word-substitution controls do not meet
    these conjunctions in the shipped regression fixtures.
    """

    COVERAGE_WEIGHT = QueryCoverageSurfaceIndex.COVERAGE_WEIGHT
    BM25_WEIGHT = BM25SurfaceIndex.BM25_WEIGHT
    FUZZY_WEIGHT = FuzzySurfaceIndex.FUZZY_WEIGHT
    MAX_FUZZY_CANDIDATES = 4
    MAX_LEXICAL_CANDIDATES_PER_SIGNAL = 128

    MIN_CONSENSUS_COMPACT_CHARS = 8
    CONSENSUS_BASE_SCORE = 0.86
    CONSENSUS_QUALITY_SPAN = 0.06

    # Character/edit agreement: useful when spacing is damaged or an input is
    # compacted and BM25 therefore has little or no token evidence.
    EDIT_TFIDF_MIN = 0.28
    EDIT_COVERAGE_MIN = 0.82
    EDIT_FUZZY_MIN = 0.90

    # Token + edit agreement: useful for partial quotations with one or more
    # local spelling errors.
    TOKEN_EDIT_TFIDF_MIN = 0.28
    TOKEN_EDIT_COVERAGE_MIN = 0.68
    TOKEN_EDIT_BM25_MIN = 0.90
    TOKEN_EDIT_FUZZY_MIN = 0.85

    # Pure lexical agreement: useful for reordered keyword queries where fuzzy
    # edit similarity is neither necessary nor desirable.
    LEXICAL_MIN_COMPACT_CHARS = 12
    LEXICAL_MIN_TOKENS = 3
    LEXICAL_TFIDF_MIN = 0.60
    LEXICAL_COVERAGE_MIN = 0.75
    LEXICAL_BM25_MIN = 0.78

    def __init__(
        self,
        tfidf: CharNgramIndex,
        coverage: QueryCoverageSurfaceIndex,
        bm25: BM25SurfaceIndex | None,
        fuzzy: FuzzySurfaceIndex | None,
    ):
        self._tfidf = tfidf
        self._coverage = coverage
        self._bm25 = bm25
        self._fuzzy = fuzzy

    @staticmethod
    def _noisy_or(evidence: SurfaceEvidence) -> float:
        """Combine bounded evidence without making the score order-dependent."""
        return 1.0 - (
            (1.0 - evidence.tfidf)
            * (1.0 - SurfaceEvidenceFusion.COVERAGE_WEIGHT * evidence.coverage)
            * (1.0 - SurfaceEvidenceFusion.BM25_WEIGHT * evidence.bm25)
            * (1.0 - SurfaceEvidenceFusion.FUZZY_WEIGHT * evidence.fuzzy)
        )

    @classmethod
    def _consensus_target(cls, values: tuple[float, ...]) -> float:
        quality = sum(values) / len(values)
        return min(
            1.0,
            cls.CONSENSUS_BASE_SCORE + cls.CONSENSUS_QUALITY_SPAN * quality,
        )

    @classmethod
    def _apply_consensus(
        cls,
        query: str,
        evidence: SurfaceEvidence,
        score: float,
    ) -> float:
        compact_length = len(normalize_compact(query))
        tokens = retrieval_tokens(query)
        if compact_length < cls.MIN_CONSENSUS_COMPACT_CHARS:
            return score

        if (
            evidence.tfidf >= cls.EDIT_TFIDF_MIN
            and evidence.coverage >= cls.EDIT_COVERAGE_MIN
            and evidence.fuzzy >= cls.EDIT_FUZZY_MIN
        ):
            score = max(
                score,
                cls._consensus_target(
                    (
                        evidence.tfidf,
                        evidence.coverage,
                        evidence.fuzzy,
                    )
                ),
            )

        if (
            len(tokens) >= 2
            and evidence.tfidf >= cls.TOKEN_EDIT_TFIDF_MIN
            and evidence.coverage >= cls.TOKEN_EDIT_COVERAGE_MIN
            and evidence.bm25 >= cls.TOKEN_EDIT_BM25_MIN
            and evidence.fuzzy >= cls.TOKEN_EDIT_FUZZY_MIN
        ):
            score = max(
                score,
                cls._consensus_target(
                    (
                        evidence.tfidf,
                        evidence.coverage,
                        evidence.bm25,
                        evidence.fuzzy,
                    )
                ),
            )

        if (
            compact_length >= cls.LEXICAL_MIN_COMPACT_CHARS
            and len(tokens) >= cls.LEXICAL_MIN_TOKENS
            and evidence.tfidf >= cls.LEXICAL_TFIDF_MIN
            and evidence.coverage >= cls.LEXICAL_COVERAGE_MIN
            and evidence.bm25 >= cls.LEXICAL_BM25_MIN
        ):
            score = max(
                score,
                cls._consensus_target(
                    (
                        evidence.tfidf,
                        evidence.coverage,
                        evidence.bm25,
                    )
                ),
            )
        return min(1.0, score)

    def search(self, query: str) -> list[tuple[int, float]]:
        """Return one fused lexical ranking for Surface mode."""
        # The final service consumes only a small top-K family ranking. Keeping
        # the strongest 128 candidates from each independent lexical signal is
        # ample for fusion while avoiding a second full-corpus Python loop for
        # broad queries that share common character grams or words with more
        # than a thousand variants. Exact/partial matching is handled before
        # this graded path, and the shipped regressions verify that this pruning
        # preserves Recall@5 and safety.
        tfidf_ranking = self._tfidf.search(query)[: self.MAX_LEXICAL_CANDIDATES_PER_SIGNAL]
        tfidf_scores = dict(tfidf_ranking)
        bm25_ranking = (
            self._bm25.search(query)[: self.MAX_LEXICAL_CANDIDATES_PER_SIGNAL]
            if self._bm25 is not None
            else []
        )
        bm25_scores = dict(bm25_ranking)
        document_ids = set(tfidf_scores) | set(bm25_scores)
        if not document_ids:
            return []

        query_grams, coverage_denominator = self._coverage._prepare_query(query)
        coverage_scores: dict[int, float] = {}
        preliminary: list[tuple[int, float]] = []
        for document_id in document_ids:
            coverage = 0.0
            if document_id in tfidf_scores and query_grams:
                coverage = self._coverage._coverage_from_prepared(
                    query_grams,
                    coverage_denominator,
                    document_id,
                )
            coverage_scores[document_id] = coverage
            evidence = SurfaceEvidence(
                tfidf=tfidf_scores.get(document_id, 0.0),
                coverage=coverage,
                bm25=bm25_scores.get(document_id, 0.0),
            )
            preliminary.append((document_id, self._noisy_or(evidence)))

        preliminary.sort(key=lambda item: (-item[1], item[0]))
        top_score = preliminary[0][1]
        second_score = preliminary[1][1] if len(preliminary) > 1 else 0.0
        clear_lexical_winner = top_score >= 0.90 and top_score - second_score >= 0.05
        fuzzy_ids = (
            set()
            if clear_lexical_winner
            else {document_id for document_id, _ in preliminary[: self.MAX_FUZZY_CANDIDATES]}
        )

        fused: list[tuple[int, float]] = []
        for document_id, preliminary_score in preliminary:
            fuzzy = 0.0
            if (
                self._fuzzy is not None
                and document_id in fuzzy_ids
                and preliminary_score < FuzzySurfaceIndex.SKIP_BASE_SCORE
            ):
                fuzzy = self._fuzzy.evidence(query, document_id)
            evidence = SurfaceEvidence(
                tfidf=tfidf_scores.get(document_id, 0.0),
                coverage=coverage_scores[document_id],
                bm25=bm25_scores.get(document_id, 0.0),
                fuzzy=fuzzy,
            )
            score = self._apply_consensus(
                query,
                evidence,
                self._noisy_or(evidence),
            )
            fused.append((document_id, score))
        return sorted(fused, key=lambda item: (-item[1], item[0]))

    def evidence(self, query: str, document_id: int) -> SurfaceEvidence:
        """Expose component evidence for tests/debugging of a known document."""
        tfidf = dict(self._tfidf.search(query)).get(document_id, 0.0)
        coverage = self._coverage.coverage(query, document_id) if tfidf else 0.0
        bm25 = (
            dict(self._bm25.search(query)).get(document_id, 0.0) if self._bm25 is not None else 0.0
        )
        fuzzy = self._fuzzy.evidence(query, document_id) if self._fuzzy else 0.0
        return SurfaceEvidence(
            tfidf=tfidf,
            coverage=coverage,
            bm25=bm25,
            fuzzy=fuzzy,
        )


class FixedExpressionRetriever:
    """Search stored spellings or glosses with task-appropriate lexical IR."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        enable_bm25: bool = True,
        enable_fuzzy: bool = True,
        enable_fusion: bool = True,
        enable_meaning_bm25: bool = True,
    ):
        self.corpus = corpus
        self.enable_bm25 = enable_bm25
        self.enable_fuzzy = enable_fuzzy
        self.enable_fusion = enable_fusion
        self.enable_meaning_bm25 = enable_meaning_bm25
        self._surface_units = corpus.all_variants()
        self._surface_document_by_variant_id = {
            variant.variant_id: document_id
            for document_id, (_, variant) in enumerate(self._surface_units)
        }
        surface_documents = [variant.text for _, variant in self._surface_units]
        self._surface_exact_partial = ExactPartialSurfaceIndex(surface_documents)
        self._surface_index = CharNgramIndex(surface_documents)
        self._surface_coverage = QueryCoverageSurfaceIndex(surface_documents)
        self._surface_bm25 = BM25SurfaceIndex(surface_documents) if enable_bm25 else None
        self._surface_fuzzy = FuzzySurfaceIndex(surface_documents) if enable_fuzzy else None
        self._surface_fusion = SurfaceEvidenceFusion(
            self._surface_index,
            self._surface_coverage,
            self._surface_bm25,
            self._surface_fuzzy,
        )
        self._gloss_units = list(corpus.expressions)
        gloss_documents = [expression.gloss for expression in self._gloss_units]
        self._gloss_index = CharNgramIndex(gloss_documents)
        self._gloss_bm25 = BM25GlossIndex(gloss_documents) if enable_meaning_bm25 else None

    @staticmethod
    def _merge_rankings(
        *rankings: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Merge evidence on the same document using its strongest score."""
        scores: dict[int, float] = {}
        for ranking in rankings:
            for document_id, score in ranking:
                scores[document_id] = max(score, scores.get(document_id, 0.0))
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def surface_evidence(
        self,
        query: str,
        candidate: Candidate,
    ) -> SurfaceEvidence:
        """Return inspectable component evidence for one Surface candidate.

        This is consumed by the selective verifier. Evidence remains
        retrieval-only metadata; it never becomes display text.
        """
        document_id = self._surface_document_by_variant_id.get(candidate.variant_id)
        if document_id is None:
            return SurfaceEvidence()
        return self._surface_fusion.evidence(query, document_id)

    def search(
        self,
        query: str,
        mode: SearchMode | str,
        limit: int = 10,
    ) -> list[Candidate]:
        if limit <= 0:
            return []
        search_mode = SearchMode(mode)
        if search_mode is SearchMode.SURFACE:
            exact_partial_ranking = self._surface_exact_partial.search(query)
            if exact_partial_ranking:
                # Exact/compact-partial evidence is categorical and stronger
                # than any graded lexical score.  Short-circuiting here also
                # avoids expensive fuzzy work for the large exact regression.
                ranked = exact_partial_ranking
            elif self.enable_fusion:
                ranked = self._surface_fusion.search(query)
            else:
                # Sequential signal path retained only for reproducible
                # ablations. Production searches use the explicit fusion
                # ranker above.
                tfidf_ranking = self._surface_index.search(query)
                coverage_ranking = self._surface_coverage.rescore(query, tfidf_ranking)
                if self._surface_bm25 is not None:
                    bm25_ranking = self._surface_bm25.search(query)
                    lexical_ranking = self._surface_bm25.rescore(coverage_ranking, bm25_ranking)
                else:
                    lexical_ranking = coverage_ranking
                if self._surface_fuzzy is not None:
                    lexical_ranking = self._surface_fuzzy.rescore(query, lexical_ranking)
                ranked = lexical_ranking
            units = [
                (expression.expression_id, expression.family_id, variant.variant_id)
                for expression, variant in self._surface_units
            ]
        else:
            # Meaning retrieval stays intentionally lexical: character TF-IDF
            # generates the candidate pool and a small BM25 gloss-overlap signal
            # only re-ranks those candidates.
            ranked = self._gloss_index.search(query)
            if self._gloss_bm25 is not None:
                ranked = self._gloss_bm25.rescore(query, ranked)
            units = [
                (
                    expression.expression_id,
                    expression.family_id,
                    expression.canonical_variant_id,
                )
                for expression in self._gloss_units
            ]

        # Different spellings of one expression must not occupy several ranks.
        output: list[Candidate] = []
        seen_families: set[str] = set()
        for document_id, score in ranked:
            expression_id, family_id, variant_id = units[document_id]
            if family_id in seen_families:
                continue
            output.append(
                Candidate(
                    expression_id=expression_id,
                    family_id=family_id,
                    variant_id=variant_id,
                    score=score,
                )
            )
            seen_families.add(family_id)
            if len(output) >= limit:
                break
        return output
