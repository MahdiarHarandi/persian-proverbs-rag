# Final 200 — PPQ vs Same-Gemma Performance Summary

Paired cases: **200 / 200**

## Headline steady-state latency

- Base median: **29.287 s**
- PPQ median: **17.589 s**
- Median paired difference (PPQ − Base): **-12.588 s**
- Median relative difference: **-41.8%**
- Median PPQ/Base ratio: **0.582×**
- Base P95: **31.737 s**
- PPQ P95: **114.912 s**

## Interpretation

The typical PPQ request was faster in this experiment, but PPQ has a substantially heavier latency tail.
Do not summarize this as “PPQ is always faster”. Use timing together with the blind quality comparison.
