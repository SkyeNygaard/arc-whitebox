# Edge-DWS Prompt 7 handoff

This package is a fail-fast, actual-width edge-state DWS runner for a frozen low-dimensional Prompt 4–6 correction label.

Current terminal state: **EXTERNALLY BLOCKED** because the current Library snapshot does not contain the frozen label/corpus bundle or canonical split registry. See `report.md` and `LOCAL_HANDOFF.md`.

Quick integrity check:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

No broad weight-to-answer target is supported. The input contract rejects label dimensions above 16 and the model never emits 256 final answers.
