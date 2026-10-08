# Decision 0002: automatic bounded draft preview

Status: superseded by Mike on 26 September 2026. The active interview app is Run-only; this file records the retired preview choice.

Mike removed automatic previews after finding the capped one-day result misleading while testing a seven-day full run. A setting edit now updates only the draft. Existing charts retain the previous explicit full result, clearly labelled as such, until **Run simulation** is clicked again. A fresh session has no sampled result until that click. The active default draft is the illustrative six-cohort forecast with 1,000 EVs, 100 evaluation worlds and seven study days. The legacy one-day models remain explicit alternatives, not the default.

The earlier preview design below is historical, not an active requirement. Do not restore it without a new product decision.

After a valid parameter change is committed to the viewer's draft, the app may automatically refresh a small, clearly labelled **Preview, not a full run**. The preview is derived from that draft identity, is bounded and cancellable, and cannot replace the active full result. Rapid subsequent edits invalidate stale previews; only the latest valid draft may publish its preview. Invalid or still-uncommitted widget values do not start work.

A full Monte Carlo starts **only** after an explicit valid **Run** click. Opening the app, editing, importing settings, navigating, Reset, downloading and ordinary Streamlit reruns never trigger a full run. A preview is not release evidence, not an approved full-result export, and not a replay/comparison parent.

The precise preview sample cap, debounce, latency/error behaviour and any optional watermarked preview export must be measured and frozen before implementation. If a bounded preview cannot give honest useful feedback, show a clear pending/unavailable state rather than silently running the full model.
