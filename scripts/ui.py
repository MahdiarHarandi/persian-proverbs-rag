#!/usr/bin/env python3
"""Polished Persian Gradio interface for the corpus-grounded PPQ runtime."""

from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path

import gradio as gr

from ppq.bootstrap import build_runtime

ROOT = Path(__file__).resolve().parents[1]

DECISION_META = {
    "COMPOSED": (
        "پاسخ مبتنی بر شواهد",
        "success",
        "عبارت نمایش‌داده‌شده مستقیماً از پیکرهٔ بازبینی‌شده بازیابی شده است.",
    ),
    "PASS_THROUGH": (
        "درخواست عمومی",
        "neutral",
        "این درخواست به عبارت ثابت فارسی نیاز ندارد و PPQ آگاهانه در آن دخالت نمی‌کند.",
    ),
    "CLARIFY": (
        "نیاز به توضیح بیشتر",
        "warning",
        "برای انتخاب مطمئن، اطلاعات بیشتری از کاربر لازم است.",
    ),
    "ABSTAIN": (
        "شواهد ناکافی",
        "warning",
        "سامانه به‌جای حدس‌زدن، از ارائهٔ عبارت نامطمئن خودداری کرده است.",
    ),
    "NOT_ATTESTED_IN_CURRENT_CORPUS": (
        "در پیکره تأیید نشد",
        "danger",
        "صورت بررسی‌شده در پیکرهٔ فعلی به‌عنوان رکورد معتبر پیدا نشد.",
    ),
}

ACTION_FA = {
    "SEARCH_BY_MEANING": "جست‌وجو بر اساس معنا",
    "ATTEST": "بررسی اصالت",
    "RESTORE_SURFACE": "بازسازی صورت عبارت",
    "EXPLAIN": "توضیح معنا",
    "RETRIEVE_EXAMPLES": "بازیابی چند نمونه",
}

ROUTE_FA = {
    "FFE_REQUIRED": "نیازمند عبارت ثابت فارسی",
    "GENERAL": "درخواست عمومی",
    "CLARIFY": "نیازمند شفاف‌سازی",
}

REVIEW_FA = {"human-reviewed": "بازبینی انسانی"}
TYPE_FA = {"fixed_expression": "عبارت ثابت فارسی"}


def _safe(value: object) -> str:
    """Escape every runtime value before placing it in an HTML component."""
    return html.escape(str(value or ""), quote=True)


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _rtl_line_html(line: str) -> str:
    """Render one Persian line with nested bidi isolation for quoted FFEs."""
    escaped = _safe(line)
    escaped = re.sub(
        r"«([^»]*)»",
        r'<span class="ppq-quote" dir="rtl">«<bdi dir="rtl">\1</bdi>»</span>',
        escaped,
    )
    return f'<p class="ppq-answer-line" dir="rtl"><bdi dir="rtl">{escaped}</bdi></p>'


def _answer_body_html(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        lines = ["پاسخی برای نمایش وجود ندارد."]
    return "".join(_rtl_line_html(line) for line in lines)


def _plan_summary_html(output: object) -> str:
    runtime = getattr(output, "runtime", None)
    plan = getattr(runtime, "plan", None)
    if plan is None:
        return ""

    route = _enum_value(getattr(plan, "route", ""))
    actions = [_enum_value(action) for action in getattr(plan, "actions", ())]
    target_span = getattr(plan, "target_span", None)

    chips = [
        '<span class="ppq-plan-chip ppq-route-chip">'
        '<span class="ppq-chip-label">مسیر</span>'
        f"{_safe(ROUTE_FA.get(route, route))}</span>"
    ]
    chips.extend(
        '<span class="ppq-plan-chip">'
        '<span class="ppq-chip-label">عملیات</span>'
        f"{_safe(ACTION_FA.get(action, action))}</span>"
        for action in actions
    )
    if target_span:
        chips.append(
            '<span class="ppq-plan-chip ppq-target-chip">'
            '<span class="ppq-chip-label">عبارت هدف</span>'
            f'<bdi dir="rtl">«{_safe(target_span)}»</bdi></span>'
        )
    return f'<div class="ppq-plan-summary">{"".join(chips)}</div>'


def _answer_html(output: object, elapsed: float) -> str:
    decision = _enum_value(getattr(output, "decision", ""))
    title, tone, description = DECISION_META.get(
        decision,
        (decision or "نتیجه", "neutral", ""),
    )
    if decision == "PASS_THROUGH":
        body = (
            "این پرسش خارج از دامنهٔ تخصصی PPQ تشخیص داده شد. در یک سامانهٔ "
            "یکپارچه، درخواست بدون تغییر به دستیار عمومی سپرده می‌شود."
        )
    else:
        body = getattr(output, "text", None) or "پاسخی برای نمایش وجود ندارد."

    return f"""
    <section class="ppq-result-card {tone}" dir="rtl" aria-live="polite">
      <div class="ppq-card-kicker">نتیجهٔ پردازش</div>
      <div class="ppq-result-head">
        <div class="ppq-status-wrap">
          <span class="ppq-status-icon" aria-hidden="true"></span>
          <span class="ppq-status">{_safe(title)}</span>
        </div>
        <span class="ppq-latency">{elapsed:.1f} ثانیه</span>
      </div>
      <div class="ppq-answer">{_answer_body_html(body)}</div>
      {_plan_summary_html(output)}
      <div class="ppq-result-desc">{_safe(description)}</div>
    </section>
    """


def _match_card_html(match: object, *, suggestion: bool) -> str:
    raw_type = str(getattr(match, "expression_type", ""))
    raw_review = str(getattr(match, "review_status", ""))
    expression_type = TYPE_FA.get(raw_type, raw_type)
    review_status = REVIEW_FA.get(raw_review, raw_review)
    badge = "پیشنهاد نزدیک" if suggestion else "شاهد تأییدشده"
    card_class = "suggestion" if suggestion else ""
    return f"""
    <article class="ppq-match {card_class}" dir="rtl">
      <div class="ppq-match-topline">
        <span class="ppq-evidence-badge">{badge}</span>
        <span class="ppq-record-id" dir="ltr">{_safe(getattr(match, "expression_id", ""))}</span>
      </div>
      <div class="ppq-expression" dir="rtl">
        <span class="ppq-quote" dir="rtl">«<bdi dir="rtl">{_safe(getattr(match, "text", ""))}</bdi>»</span>
      </div>
      <div class="ppq-gloss" dir="rtl"><bdi dir="rtl">{_safe(getattr(match, "gloss", ""))}</bdi></div>
      <div class="ppq-source-row">
        <span>{_safe(expression_type)}</span>
        <span>{_safe(review_status)}</span>
        <span>{_safe(getattr(match, "source_name", ""))}</span>
      </div>
    </article>
    """


def _evidence_html(output: object) -> str:
    runtime = getattr(output, "runtime", None)
    matches = list(getattr(runtime, "matches", ()) or ())
    suggestions = list(getattr(runtime, "suggestions", ()) or ())

    if not matches and not suggestions:
        return """
        <section class="ppq-evidence-card ppq-evidence-empty" dir="rtl" aria-live="polite">
          <div class="ppq-card-kicker">شفافیت پاسخ</div>
          <div class="ppq-empty-icon" aria-hidden="true">⌁</div>
          <h2>رکوردی انتخاب نشد</h2>
          <p>در این تصمیم، عبارت قابل نمایشی از پیکره انتخاب نشده است.</p>
        </section>
        """

    cards = "".join(_match_card_html(match, suggestion=False) for match in matches) + "".join(
        _match_card_html(match, suggestion=True) for match in suggestions
    )
    total = len(matches) + len(suggestions)
    return f"""
    <section class="ppq-evidence-card" dir="rtl" aria-live="polite">
      <div class="ppq-evidence-heading">
        <div>
          <div class="ppq-card-kicker">شفافیت پاسخ</div>
          <h2>شواهد پیکره</h2>
        </div>
        <span class="ppq-count-badge">{total} رکورد</span>
      </div>
      <div class="ppq-evidence-list">{cards}</div>
    </section>
    """


def _technical_payload(output: object, query: str, elapsed: float) -> dict:
    runtime = output.runtime
    plan = runtime.plan
    plan_payload = None
    if plan is not None:
        actions = [_enum_value(action) for action in plan.actions]
        route = _enum_value(plan.route)
        plan_payload = {
            "route": route,
            "route_fa": ROUTE_FA.get(route, route),
            "actions": actions,
            "actions_fa": [ACTION_FA.get(action, action) for action in actions],
            "target_span": plan.target_span,
            "semantic_query": plan.semantic_query,
            "requested_count": plan.requested_count,
            "needs_explanation": plan.needs_explanation,
        }

    def match_to_dict(match: object) -> dict:
        return {
            "expression_id": match.expression_id,
            "family_id": match.family_id,
            "variant_id": match.variant_id,
            "text": match.text,
            "gloss": match.gloss,
            "expression_type": match.expression_type,
            "source_name": match.source_name,
            "review_status": match.review_status,
        }

    return {
        "query": query,
        "decision": output.decision.value,
        "runtime_decision": runtime.decision.value,
        "plan": plan_payload,
        "matches": [match_to_dict(match) for match in runtime.matches],
        "suggestions": [match_to_dict(match) for match in runtime.suggestions],
        "generation_used": output.generation_used,
        "generation_valid": output.generation_valid,
        "runtime_reason": runtime.reason,
        "response_reason": output.reason,
        "planner_error": runtime.planner_error,
        "elapsed_seconds": round(elapsed, 3),
    }


INITIAL_ANSWER_HTML = """
<section class="ppq-result-card ppq-initial-card" dir="rtl">
  <div class="ppq-card-kicker">فضای پاسخ</div>
  <div class="ppq-initial-mark" aria-hidden="true">« »</div>
  <h2>آمادهٔ دریافت درخواست فارسی</h2>
  <p>پاسخ نهایی، تصمیم سامانه و مسیر پردازش در این بخش نمایش داده می‌شود.</p>
  <div class="ppq-mini-flow" aria-label="مراحل پردازش">
    <span>درک درخواست</span><i></i><span>کشف شواهد</span><i></i><span>پاسخ ایمن</span>
  </div>
</section>
"""

INITIAL_EVIDENCE_HTML = """
<section class="ppq-evidence-card ppq-evidence-empty ppq-initial-evidence" dir="rtl">
  <div class="ppq-card-kicker">شفافیت پاسخ</div>
  <div class="ppq-empty-icon" aria-hidden="true">⌁</div>
  <h2>شواهد انتخاب‌شده</h2>
  <p>عبارت، معنا، منبع و وضعیت بازبینی رکوردهای انتخاب‌شده اینجا دیده می‌شود.</p>
</section>
"""

PENDING_ANSWER_HTML = """
<section class="ppq-result-card ppq-pending-card" dir="rtl" aria-live="polite">
  <div class="ppq-card-kicker">در حال پردازش</div>
  <div class="ppq-pending-row">
    <span class="ppq-spinner" aria-hidden="true"></span>
    <div><h2>سامانه در حال بررسی درخواست است</h2><p>برنامه‌ریزی، بازیابی و اعتبارسنجی شواهد ممکن است چند ثانیه زمان ببرد.</p></div>
  </div>
</section>
"""

EMPTY_QUERY_HTML = """
<section class="ppq-result-card warning" dir="rtl" aria-live="polite">
  <div class="ppq-card-kicker">ورودی نامعتبر</div>
  <div class="ppq-result-head">
    <div class="ppq-status-wrap"><span class="ppq-status-icon"></span><span class="ppq-status">پرسشی وارد نشده است</span></div>
  </div>
  <div class="ppq-result-desc ppq-no-divider">یک درخواست فارسی بنویسید و دکمهٔ «پردازش درخواست» را بزنید.</div>
</section>
"""

HERO_HTML = """
<section class="ppq-hero" dir="rtl">
  <div class="ppq-hero-grid">
    <div class="ppq-hero-copy">
      <div class="ppq-brand-row">
        <div class="ppq-logo" aria-hidden="true"><span>پ</span><span>پ</span><span>ک</span></div>
        <div>
          <div class="ppq-eyebrow"><span class="ppq-live-dot"></span> پروژهٔ کارشناسی مهندسی کامپیوتر</div>
          <div class="ppq-brand-en" dir="ltr">PERSIAN PROVERB &amp; QUOTATION ASSISTANT</div>
        </div>
      </div>
      <h1>عبارت درست،<br><em>با شواهد قابل بررسی</em></h1>
      <p>سامانه‌ای برای بازیابی، تکمیل، بررسی اصالت و توضیح ضرب‌المثل‌ها و عبارات ثابت فارسی؛ بدون ساختن عبارت حدسی.</p>
      <div class="ppq-trust-row">
        <span>پاسخ مبتنی بر پیکره</span>
        <span>خودداری در نبود شواهد</span>
        <span>نمایش مسیر تصمیم</span>
      </div>
    </div>
    <div class="ppq-pipeline" aria-label="معماری سطح بالای سامانه">
      <div class="ppq-pipeline-head"><span>مسیر یک درخواست</span><b dir="ltr">PPQ / LIVE</b></div>
      <div class="ppq-pipeline-step active"><strong>۰۱</strong><div><b>درک درخواست</b><small>مسیریابی و تشخیص عملیات</small></div><span>✓</span></div>
      <div class="ppq-pipeline-line"></div>
      <div class="ppq-pipeline-step"><strong>۰۲</strong><div><b>کشف و اعتبارسنجی</b><small>بازیابی معنایی و بررسی شواهد</small></div><span>⌁</span></div>
      <div class="ppq-pipeline-line"></div>
      <div class="ppq-pipeline-step"><strong>۰۳</strong><div><b>پاسخ کنترل‌شده</b><small>نقل مستقیم عبارت از پیکره</small></div><span>↗</span></div>
    </div>
  </div>
  <div class="ppq-stat-row">
    <div><strong>۱۸۲۰</strong><span>خانوادهٔ عبارت</span></div>
    <div><strong>۲۰۶۷</strong><span>گونهٔ ثبت‌شده</span></div>
    <div><strong>۲۰۰</strong><span>پرامپت بنچمارک مهرشده</span></div>
    <div><strong>۹۷٪</strong><span>دقت مسیریابی</span></div>
  </div>
</section>
"""

UI_HEAD = """
<meta name="description" content="PPQ: corpus-grounded Persian proverb and quotation assistant">
<meta name="theme-color" content="#0b2530">
<script>
  document.documentElement.setAttribute("lang", "fa");
  document.documentElement.setAttribute("dir", "rtl");
</script>
"""

UI_CSS = r"""
:root {
  --ppq-navy: #0b2530;
  --ppq-navy-2: #123844;
  --ppq-emerald: #0f766e;
  --ppq-bg: #f3f6f6;
  --ppq-ink: #102a34;
  --ppq-muted: #64747a;
  --ppq-line: #dfe8e7;
  --ppq-shadow: 0 22px 55px rgba(17, 50, 59, .09);
}

html, body {
  background:
    radial-gradient(circle at 10% 0%, rgba(166, 219, 205, .28), transparent 31rem),
    radial-gradient(circle at 95% 24%, rgba(212, 226, 203, .35), transparent 27rem),
    var(--ppq-bg) !important;
  min-height: 100%;
}

.gradio-container {
  width: 100% !important;
  max-width: none !important;
  margin: 0 !important;
  padding: 0 !important;
  direction: rtl !important;
  background: transparent !important;
  color: var(--ppq-ink) !important;
  font-family: Vazirmatn, "Segoe UI", Tahoma, Arial, sans-serif !important;
}

.ppq-app-shell {
  width: min(1180px, calc(100% - 32px)) !important;
  margin: 0 auto !important;
  padding: 24px 0 18px !important;
  gap: 0 !important;
}

.ppq-hero {
  position: relative;
  overflow: hidden;
  isolation: isolate;
  color: #fff;
  border: 1px solid rgba(255, 255, 255, .11);
  border-radius: 30px;
  padding: 36px 38px 0;
  background:
    linear-gradient(135deg, rgba(255,255,255,.035) 0 1px, transparent 1px 12px),
    linear-gradient(130deg, var(--ppq-navy), #0d3038 58%, #164a48);
  box-shadow: 0 28px 70px rgba(9, 37, 48, .19);
}

.ppq-hero::before,
.ppq-hero::after {
  content: "";
  position: absolute;
  z-index: -1;
  border-radius: 50%;
  pointer-events: none;
}

.ppq-hero::before {
  width: 330px;
  height: 330px;
  left: -150px;
  top: -180px;
  background: rgba(90, 197, 171, .17);
}

.ppq-hero::after {
  width: 190px;
  height: 190px;
  right: 44%;
  bottom: -145px;
  border: 42px solid rgba(255, 255, 255, .035);
}

.ppq-hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.32fr) minmax(300px, .68fr);
  gap: 42px;
  align-items: center;
}

.ppq-brand-row { display: flex; align-items: center; gap: 13px; }

.ppq-logo {
  width: 51px;
  height: 51px;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  place-items: center;
  border-radius: 15px;
  padding: 0 7px;
  background: rgba(255, 255, 255, .11);
  border: 1px solid rgba(255, 255, 255, .18);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.13);
  font-size: 12px;
  font-weight: 900;
  color: #c9f7e8;
}

.ppq-logo span:nth-child(2) { opacity: .68; }
.ppq-logo span:nth-child(3) { opacity: .42; }

.ppq-eyebrow {
  display: flex;
  align-items: center;
  gap: 7px;
  color: #d7e8e5;
  font-size: 12px;
  font-weight: 700;
}

.ppq-live-dot {
  width: 7px;
  height: 7px;
  display: inline-block;
  border-radius: 50%;
  background: #64d7b3;
  box-shadow: 0 0 0 5px rgba(100, 215, 179, .11);
}

.ppq-brand-en {
  margin-top: 5px;
  color: rgba(222, 240, 235, .5);
  font-size: 8.5px;
  font-weight: 800;
  letter-spacing: 1.25px;
}

.ppq-hero h1 {
  margin: 27px 0 13px;
  color: #fff;
  font-size: clamp(35px, 4.2vw, 59px);
  line-height: 1.27;
  letter-spacing: -1.8px;
  font-weight: 900;
}

.ppq-hero h1 em { color: #91e5c9; font-style: normal; }

.ppq-hero-copy > p {
  max-width: 690px;
  margin: 0;
  color: #c6d9d7;
  font-size: 14px;
  line-height: 2.05;
}

.ppq-trust-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 21px; }

.ppq-trust-row span {
  position: relative;
  padding: 7px 23px 7px 11px;
  border: 1px solid rgba(255,255,255,.11);
  border-radius: 999px;
  background: rgba(255,255,255,.055);
  color: #d8e7e5;
  font-size: 10.5px;
}

.ppq-trust-row span::before {
  content: "";
  position: absolute;
  right: 10px;
  top: 50%;
  width: 5px;
  height: 5px;
  transform: translateY(-50%);
  border-radius: 50%;
  background: #69d9b7;
}

.ppq-pipeline {
  padding: 20px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 22px;
  background: rgba(6, 29, 37, .36);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06);
  backdrop-filter: blur(10px);
}

.ppq-pipeline-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 15px;
  margin-bottom: 14px;
  border-bottom: 1px solid rgba(255,255,255,.1);
  color: #c4d8d5;
  font-size: 10px;
  font-weight: 700;
}

.ppq-pipeline-head b { color: #72d8b7; font-size: 8px; letter-spacing: 1px; }

.ppq-pipeline-step {
  display: grid;
  grid-template-columns: 29px 1fr 25px;
  gap: 10px;
  align-items: center;
}

.ppq-pipeline-step > strong { color: rgba(255,255,255,.32); font-size: 10px; direction: ltr; }
.ppq-pipeline-step > div { display: flex; flex-direction: column; gap: 3px; }
.ppq-pipeline-step b { color: #f5fbfa; font-size: 12px; }
.ppq-pipeline-step small { color: #8fa9a7; font-size: 9px; }

.ppq-pipeline-step > span {
  width: 25px;
  height: 25px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  color: #7adfbd;
  background: rgba(105,217,183,.09);
  font-size: 10px;
}

.ppq-pipeline-step.active > strong { color: #7be0be; }
.ppq-pipeline-line { width: 1px; height: 18px; margin: 4px 14px 4px 0; background: linear-gradient(#63c9aa, rgba(99,201,170,.08)); }

.ppq-stat-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  margin: 30px -38px 0;
  padding: 0 38px;
  border-top: 1px solid rgba(255,255,255,.09);
  background: rgba(3, 21, 29, .18);
}

.ppq-stat-row > div { display: flex; align-items: baseline; gap: 8px; padding: 17px 0; }
.ppq-stat-row > div + div { border-right: 1px solid rgba(255,255,255,.08); padding-right: 24px; }
.ppq-stat-row strong { color: #fff; font-size: 18px; font-weight: 900; }
.ppq-stat-row span { color: #9fb4b2; font-size: 9.5px; }

.ppq-workspace {
  margin-top: 20px;
  padding: 23px 24px 21px !important;
  border: 1px solid rgba(214, 226, 224, .94) !important;
  border-radius: 24px !important;
  background: rgba(255,255,255,.94) !important;
  box-shadow: var(--ppq-shadow) !important;
  gap: 0 !important;
}

.ppq-section-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 14px;
}

.ppq-section-heading h2 { margin: 0; color: var(--ppq-ink); font-size: 18px; font-weight: 900; }
.ppq-section-heading p { margin: 5px 0 0; color: var(--ppq-muted); font-size: 11.5px; line-height: 1.8; }
.ppq-key-hint { direction: rtl; white-space: nowrap; color: #829196; font-size: 9.5px; }

.ppq-key-hint kbd {
  margin: 0 3px;
  padding: 3px 7px;
  border: 1px solid #dfe7e5;
  border-bottom-width: 2px;
  border-radius: 6px;
  background: #f8faf9;
  color: #52656b;
  font-family: inherit;
  font-size: 9px;
}

.ppq-prompt-frame {
  position: relative;
  z-index: 2;
  padding: 5px;
  border: 1px solid #d9e5e2;
  border-radius: 17px;
  background: #f8fbfa;
  transition: border-color .18s ease, box-shadow .18s ease, background .18s ease;
}

.ppq-prompt-frame:focus-within { border-color: #4cad99; background: #fff; box-shadow: 0 0 0 4px rgba(15,118,110,.09); }

#ppq-prompt {
  position: relative !important;
  z-index: 3 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

#ppq-prompt textarea {
  min-height: 136px !important;
  padding: 15px 16px !important;
  border: 0 !important;
  outline: 0 !important;
  border-radius: 13px !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #16333c !important;
  caret-color: var(--ppq-emerald) !important;
  cursor: text !important;
  pointer-events: auto !important;
  user-select: text !important;
  direction: rtl !important;
  text-align: right !important;
  unicode-bidi: plaintext !important;
  font-family: Vazirmatn, "Segoe UI", Tahoma, Arial, sans-serif !important;
  font-size: 15px !important;
  line-height: 2.05 !important;
  resize: vertical !important;
}

#ppq-prompt textarea::placeholder { color: #98a6aa !important; opacity: 1 !important; }
.ppq-actions { align-items: center !important; gap: 10px !important; margin-top: 12px !important; }

.ppq-primary, .ppq-secondary, .ppq-example {
  font-family: Vazirmatn, "Segoe UI", Tahoma, Arial, sans-serif !important;
  box-shadow: none !important;
  transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, background .16s ease !important;
}

.ppq-primary {
  min-height: 47px !important;
  border: 1px solid var(--ppq-navy) !important;
  border-radius: 13px !important;
  background: var(--ppq-navy) !important;
  color: #fff !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  box-shadow: 0 9px 18px rgba(11,37,48,.16) !important;
}

.ppq-primary:hover { transform: translateY(-1px); background: var(--ppq-navy-2) !important; box-shadow: 0 12px 23px rgba(11,37,48,.2) !important; }

.ppq-secondary {
  min-height: 47px !important;
  border: 1px solid #dce6e4 !important;
  border-radius: 13px !important;
  background: #fff !important;
  color: #53676d !important;
  font-size: 11px !important;
  font-weight: 700 !important;
}

.ppq-secondary:hover { border-color: #aabdb9 !important; background: #f8fbfa !important; }
.ppq-examples-wrap { margin-top: 17px; padding-top: 15px; border-top: 1px solid #edf2f1; }
.ppq-examples-label { margin-bottom: 8px; color: #6b7d82; font-size: 10px; font-weight: 800; }
.ppq-examples-row { gap: 8px !important; }

.ppq-example {
  min-height: 40px !important;
  border: 1px solid #e0e9e7 !important;
  border-radius: 11px !important;
  background: #fbfcfc !important;
  color: #53676c !important;
  font-size: 10.5px !important;
  font-weight: 650 !important;
}

.ppq-example:hover { transform: translateY(-1px); border-color: #a7c7bf !important; background: #f2faf7 !important; color: #175f56 !important; }
.ppq-output-grid { align-items: stretch !important; gap: 14px !important; margin-top: 18px !important; }
.ppq-output-column { min-width: 0 !important; gap: 0 !important; }
.ppq-html-output { padding: 0 !important; background: transparent !important; }

.ppq-result-card, .ppq-evidence-card {
  height: 100%;
  min-height: 255px;
  padding: 22px 23px;
  border: 1px solid var(--ppq-line);
  border-radius: 22px;
  background: rgba(255,255,255,.94);
  box-shadow: 0 13px 36px rgba(20,53,61,.055);
}

.ppq-card-kicker { margin-bottom: 12px; color: #8b9b9f; font-size: 9px; font-weight: 900; letter-spacing: .2px; }
.ppq-result-head, .ppq-evidence-heading, .ppq-match-topline { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.ppq-status-wrap { display: flex; align-items: center; gap: 9px; }

.ppq-status-icon {
  width: 11px;
  height: 11px;
  display: inline-block;
  border: 3px solid #a8b5b8;
  border-radius: 50%;
  box-shadow: 0 0 0 4px rgba(123, 141, 147, .09);
}

.ppq-status { color: #3d555d; font-size: 13px; font-weight: 900; }
.ppq-result-card.success .ppq-status-icon { border-color: #1da87e; box-shadow: 0 0 0 4px rgba(29,168,126,.1); }
.ppq-result-card.success .ppq-status { color: #087257; }
.ppq-result-card.warning .ppq-status-icon { border-color: #e59b2e; box-shadow: 0 0 0 4px rgba(229,155,46,.1); }
.ppq-result-card.warning .ppq-status { color: #9b6513; }
.ppq-result-card.danger .ppq-status-icon { border-color: #d95454; box-shadow: 0 0 0 4px rgba(217,84,84,.1); }
.ppq-result-card.danger .ppq-status { color: #a43b3b; }

.ppq-latency, .ppq-count-badge {
  direction: rtl;
  white-space: nowrap;
  padding: 5px 9px;
  border-radius: 999px;
  background: #f0f5f4;
  color: #6f8186;
  font-size: 9px;
  font-weight: 700;
}

.ppq-answer { margin-top: 22px; color: #102d36; font-size: 17px; font-weight: 750; line-height: 2.12; }
.ppq-answer-line { margin: 0 0 5px; direction: rtl; text-align: right; unicode-bidi: plaintext; }
.ppq-answer-line bdi, .ppq-quote, .ppq-quote bdi { direction: rtl; unicode-bidi: isolate; }
.ppq-plan-summary { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 18px; }

.ppq-plan-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  border: 1px solid #e2ece9;
  border-radius: 8px;
  background: #f8fbfa;
  color: #476168;
  font-size: 9px;
}

.ppq-chip-label { color: #92a1a4; font-size: 8px; font-weight: 800; }
.ppq-route-chip { border-color: #cce9df; background: #f0faf6; color: #126b59; }
.ppq-target-chip { width: 100%; justify-content: flex-start; }

.ppq-result-desc {
  margin-top: 17px;
  padding-top: 13px;
  border-top: 1px solid #eef3f2;
  color: #7d8e92;
  font-size: 10px;
  line-height: 1.9;
}

.ppq-no-divider { border-top: 0; padding-top: 3px; }

.ppq-evidence-heading h2, .ppq-evidence-empty h2, .ppq-initial-card h2, .ppq-pending-card h2 {
  margin: 0;
  color: #17323b;
  font-size: 14px;
  font-weight: 900;
}

.ppq-evidence-list { margin-top: 12px; }
.ppq-match { padding: 15px 0 4px; border-top: 1px solid #edf2f1; }
.ppq-match:first-child { border-top: 0; padding-top: 6px; }
.ppq-evidence-badge { color: #11705c; font-size: 8px; font-weight: 900; }
.ppq-match.suggestion .ppq-evidence-badge { color: #9a6b1c; }
.ppq-record-id { color: #a3afb2; font-family: ui-monospace, Consolas, monospace; font-size: 8px; }

.ppq-expression {
  margin-top: 10px;
  color: #102d36;
  font-size: 16px;
  font-weight: 900;
  line-height: 1.8;
  unicode-bidi: plaintext;
}

.ppq-gloss { margin-top: 6px; color: #5d7177; font-size: 11.5px; line-height: 1.95; unicode-bidi: plaintext; }
.ppq-source-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 11px; }
.ppq-source-row span { padding: 3px 7px; border-radius: 6px; background: #f3f7f6; color: #849398; font-size: 7.8px; }

.ppq-evidence-empty, .ppq-initial-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
}

.ppq-initial-card { background: linear-gradient(145deg, #fff, #f8fbfa); }
.ppq-initial-evidence { background: #fafcfc; }
.ppq-initial-mark, .ppq-empty-icon { margin-bottom: 13px; color: #84b7a8; font-size: 26px; font-weight: 300; }

.ppq-evidence-empty p, .ppq-initial-card > p, .ppq-pending-card p {
  max-width: 480px;
  margin: 8px 0 0;
  color: #7a8c91;
  font-size: 10.5px;
  line-height: 1.9;
}

.ppq-mini-flow { display: flex; align-items: center; gap: 7px; margin-top: 22px; color: #6e8287; font-size: 8px; font-weight: 800; }
.ppq-mini-flow i { width: 22px; height: 1px; background: #cbdad6; }
.ppq-pending-card { display: flex; flex-direction: column; justify-content: center; }
.ppq-pending-row { display: flex; align-items: center; gap: 16px; }

.ppq-spinner {
  width: 27px;
  height: 27px;
  flex: 0 0 auto;
  border: 3px solid #d9ebe6;
  border-top-color: var(--ppq-emerald);
  border-radius: 50%;
  animation: ppq-spin .8s linear infinite;
}

@keyframes ppq-spin { to { transform: rotate(360deg); } }

.ppq-accordion {
  margin-top: 13px !important;
  overflow: hidden !important;
  border: 1px solid #dde8e5 !important;
  border-radius: 14px !important;
  background: rgba(255,255,255,.86) !important;
  box-shadow: none !important;
}

.ppq-accordion > .label-wrap, .ppq-accordion > button {
  direction: rtl !important;
  color: #586d72 !important;
  font-family: Vazirmatn, "Segoe UI", Tahoma, Arial, sans-serif !important;
  font-size: 10.5px !important;
  font-weight: 750 !important;
}

.ppq-tech, .ppq-tech .json-holder, .ppq-tech pre {
  direction: ltr !important;
  text-align: left !important;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace !important;
}

.ppq-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 21px 4px 4px;
  color: #8d9c9f;
  font-size: 8.5px;
}

.ppq-footer b { color: #587178; font-weight: 850; }
.ppq-footer span:last-child { direction: ltr; letter-spacing: .35px; }
footer { display: none !important; }

@media (max-width: 900px) {
  .ppq-hero-grid { grid-template-columns: 1fr; gap: 27px; }
  .ppq-pipeline { max-width: none; }
  .ppq-stat-row { grid-template-columns: repeat(2, 1fr); }
  .ppq-stat-row > div:nth-child(3), .ppq-stat-row > div:nth-child(4) { border-top: 1px solid rgba(255,255,255,.08); }
}

@media (max-width: 700px) {
  .ppq-app-shell { width: min(100% - 18px, 1180px) !important; padding-top: 9px !important; }
  .ppq-hero { padding: 24px 20px 0; border-radius: 23px; }
  .ppq-hero h1 { margin-top: 23px; font-size: 36px; letter-spacing: -1px; }
  .ppq-brand-en { letter-spacing: .65px; }
  .ppq-stat-row { margin-right: -20px; margin-left: -20px; padding: 0 20px; }
  .ppq-stat-row > div { flex-direction: column; gap: 2px; padding: 13px 0; }
  .ppq-stat-row > div + div { padding-right: 13px; }
  .ppq-stat-row strong { font-size: 16px; }
  .ppq-workspace { padding: 18px 15px !important; border-radius: 20px !important; }
  .ppq-section-heading { align-items: flex-start; flex-direction: column; gap: 5px; }
  .ppq-key-hint { display: none; }
  #ppq-prompt textarea { min-height: 150px !important; font-size: 14px !important; }
  .ppq-output-grid { flex-direction: column !important; }
  .ppq-output-column {
    width: 100% !important;
    max-width: 100% !important;
    flex: 1 1 100% !important;
  }
  .ppq-result-card, .ppq-evidence-card { min-height: 225px; padding: 19px 18px; }
  .ppq-answer { font-size: 15.5px; }
  .ppq-footer { flex-direction: column; text-align: center; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
  .ppq-spinner { animation-duration: 1.6s; }
}
"""


def _theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="emerald",
        secondary_hue="teal",
        neutral_hue="slate",
        font=("Vazirmatn", "Segoe UI", "Tahoma", "Arial", "sans-serif"),
    )


def create_demo(assistant: object) -> gr.Blocks:
    """Create the UI without loading models, which keeps it independently testable."""
    counter = {"value": 0}

    def run_query(query: str):
        query = (query or "").strip()
        if not query:
            return EMPTY_QUERY_HTML, INITIAL_EVIDENCE_HTML, {}

        counter["value"] += 1
        started = time.perf_counter()
        try:
            output = assistant.handle(query, query_id=f"ui-{counter['value']:05d}")
            elapsed = time.perf_counter() - started
            return (
                _answer_html(output, elapsed),
                _evidence_html(output),
                _technical_payload(output, query, elapsed),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            error_html = f"""
            <section class="ppq-result-card danger" dir="rtl" aria-live="assertive">
              <div class="ppq-card-kicker">خطای اجرا</div>
              <div class="ppq-result-head">
                <div class="ppq-status-wrap"><span class="ppq-status-icon"></span><span class="ppq-status">خطا در پردازش درخواست</span></div>
                <span class="ppq-latency">{elapsed:.1f} ثانیه</span>
              </div>
              <div class="ppq-result-desc ppq-no-divider" dir="ltr">{_safe(type(exc).__name__)}: {_safe(exc)}</div>
            </section>
            """
            return (
                error_html,
                INITIAL_EVIDENCE_HTML,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(elapsed, 3),
                },
            )

    def pending_state():
        return PENDING_ANSWER_HTML, INITIAL_EVIDENCE_HTML, {}

    def clear_state():
        return "", INITIAL_ANSWER_HTML, INITIAL_EVIDENCE_HTML, {}

    demo = gr.Blocks(title="PPQ | سامانهٔ عبارات ثابت فارسی")

    with demo:
        with gr.Column(elem_classes=["ppq-app-shell"]):
            gr.HTML(HERO_HTML)

            with gr.Column(elem_classes=["ppq-workspace"]):
                gr.HTML(
                    """
                    <div class="ppq-section-heading" dir="rtl">
                      <div><h2>درخواست خود را بنویسید</h2><p>می‌توانید یک عبارت ناقص، معنای موردنظر، موقعیت یا پرسش دربارهٔ اصالت یک عبارت را وارد کنید.</p></div>
                      <div class="ppq-key-hint">ورودی فارسی · حداکثر ۲۰۰۰ نویسه</div>
                    </div>
                    """
                )
                with gr.Column(elem_classes=["ppq-prompt-frame"]):
                    query = gr.Textbox(
                        label="متن درخواست فارسی",
                        show_label=False,
                        placeholder="برای نمونه: برای کسی که همیشه کارهایش را به فردا می‌اندازد، یک ضرب‌المثل مناسب پیشنهاد کن…",
                        lines=4,
                        max_lines=8,
                        max_length=2000,
                        interactive=True,
                        autofocus=True,
                        rtl=True,
                        text_align="right",
                        container=False,
                        elem_id="ppq-prompt",
                        html_attributes=gr.InputHTMLAttributes(
                            lang="fa",
                            spellcheck=True,
                            autocapitalize="sentences",
                            enterkeyhint="send",
                        ),
                    )

                with gr.Row(elem_classes=["ppq-actions"]):
                    submit = gr.Button(
                        "پردازش درخواست  ←",
                        variant="primary",
                        size="lg",
                        scale=3,
                        elem_id="ppq-submit",
                        elem_classes=["ppq-primary"],
                    )
                    clear = gr.Button(
                        "پاک‌کردن",
                        variant="secondary",
                        size="lg",
                        scale=1,
                        min_width=120,
                        elem_id="ppq-clear",
                        elem_classes=["ppq-secondary"],
                    )

                with gr.Column(elem_classes=["ppq-examples-wrap"]):
                    gr.HTML('<div class="ppq-examples-label">نمونه‌های آماده برای نمایش سریع</div>')
                    with gr.Row(elem_classes=["ppq-examples-row"]):
                        ex1 = gr.Button(
                            "پیشنهاد بر اساس موقعیت",
                            size="sm",
                            min_width=190,
                            elem_classes=["ppq-example"],
                        )
                        ex2 = gr.Button(
                            "بررسی اصالت یک عبارت",
                            size="sm",
                            min_width=190,
                            elem_classes=["ppq-example"],
                        )
                        ex3 = gr.Button(
                            "بازسازی صورت ناقص",
                            size="sm",
                            min_width=190,
                            elem_classes=["ppq-example"],
                        )

            with gr.Row(elem_classes=["ppq-output-grid"]):
                with gr.Column(scale=6, elem_classes=["ppq-output-column"]):
                    answer = gr.HTML(
                        INITIAL_ANSWER_HTML,
                        elem_id="ppq-answer-output",
                        elem_classes=["ppq-html-output"],
                    )
                with gr.Column(scale=4, elem_classes=["ppq-output-column"]):
                    evidence = gr.HTML(
                        INITIAL_EVIDENCE_HTML,
                        elem_id="ppq-evidence-output",
                        elem_classes=["ppq-html-output"],
                    )

            with gr.Accordion(
                "جزئیات فنی و مسیر تصمیم", open=False, elem_classes=["ppq-accordion"]
            ):
                details = gr.JSON(value={}, label=None, elem_classes=["ppq-tech"])

            gr.HTML(
                """
                <div class="ppq-footer" dir="rtl">
                  <span><b>PPQ</b> · پروژهٔ کارشناسی مهندسی کامپیوتر · دانشگاه تهران</span>
                  <span>Corpus-grounded Persian fixed-expression retrieval</span>
                </div>
                """
            )

        submit_pending = submit.click(
            pending_state,
            inputs=None,
            outputs=[answer, evidence, details],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        submit_pending.then(
            run_query,
            inputs=[query],
            outputs=[answer, evidence, details],
            show_progress="minimal",
            scroll_to_output=True,
            api_name="process_query",
        )

        enter_pending = query.submit(
            pending_state,
            inputs=None,
            outputs=[answer, evidence, details],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        enter_pending.then(
            run_query,
            inputs=[query],
            outputs=[answer, evidence, details],
            show_progress="minimal",
            scroll_to_output=True,
            api_visibility="private",
        )

        clear.click(
            clear_state,
            inputs=None,
            outputs=[query, answer, evidence, details],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        ex1.click(
            lambda: "برای کسی که همیشه کارهایش را به فردا می‌اندازد یک ضرب‌المثل مناسب پیشنهاد کن",
            outputs=[query],
            queue=False,
            api_visibility="private",
        )
        ex2.click(
            lambda: "آیا «هر که بامش بیش برفش بیشتر» یک ضرب‌المثل معتبر است؟",
            outputs=[query],
            queue=False,
            api_visibility="private",
        )
        ex3.click(
            lambda: "فکر می‌کنم می‌گویند «از تو حرکت از خدا…»؛ صورت کاملش چیست؟",
            outputs=[query],
            queue=False,
            api_visibility="private",
        )

    return demo


def launch_demo(demo: gr.Blocks, *, port: int, share: bool) -> None:
    """Launch with Gradio 6 styling arguments in their supported location."""
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
        theme=_theme(),
        css=UI_CSS,
        head=UI_HEAD,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the PPQ Gradio interface")
    parser.add_argument(
        "--embedding-device",
        choices=["cuda:0", "cpu"],
        default="cuda:0",
        help="Device used by the BGE-M3 query encoder; Gemma still requires CUDA.",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a temporary public Gradio link."
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading the PPQ runtime. Models are initialized once before launch...")
    assistant = build_runtime(ROOT, args.embedding_device, log=print)
    print("PPQ runtime is ready. Launching the web interface...")

    launch_demo(create_demo(assistant), port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
