# Quickstart

This tutorial gets you from zero to first success in ~5 minutes. It is copy‑paste runnable and includes concrete success checks.

## Prerequisites

- Python `3.13` (required)
- A terminal with `pip` available
- Optional: Google Gemini API key (not required for dry‑run/mock mode)

## 1) Install

Pick one path:

```bash
# A) From Releases (recommended)
# Download the latest .whl from:
# https://github.com/seanbrar/gemini-batch-prediction/releases/latest
pip install ./gemini_batch-*.whl

# B) From source (this repo)
git clone https://github.com/seanbrar/gemini-batch-prediction.git
cd gemini-batch-prediction
pip install -e .

# Full steps are in How‑to → Installation
```

Optional: notebooks/visualization helpers

```bash
pip install "matplotlib~=3.10" "pandas~=2.3" "seaborn~=0.13"
```

Verify installation:

```bash
python -c "import gemini_batch as gb; print('✅', gb.__version__)"
```

## 2) First run (no API key needed)

The library defaults to a deterministic mock mode until you opt‑in to the real API. This means you can verify setup without any credentials or cost.

```python
# save as hello.py
import asyncio
from gemini_batch import run_simple, types

async def main():
    result = await run_simple(
        "Say hello to Quickstart",
        source=types.Source.from_text("Quickstart content"),
    )
    print(result["status"], result["answers"][0])

asyncio.run(main())
```

Run it:

```bash
python hello.py
```

Expected output (mocked):

```text
ok echo: Say hello to Quickstart
```

## 3) Use the real API (optional)

Enable real calls and configure your billing tier to avoid rate‑limit surprises.

```bash
# Set your key and enable real API (bash/zsh)
export GEMINI_API_KEY="<your key>"
export GEMINI_BATCH_TIER=free      # free | tier_1 | tier_2 | tier_3
export GEMINI_BATCH_USE_REAL_API=1

# Sanity check (redacted):
gb-config doctor
```

Windows PowerShell:

```powershell
$Env:GEMINI_API_KEY = "<your key>"
$Env:GEMINI_BATCH_TIER = "free"      # free | tier_1 | tier_2 | tier_3
$Env:GEMINI_BATCH_USE_REAL_API = "1"
gb-config doctor
```

Re‑run `hello.py`. You should still see `status: ok` with a non‑mock answer.

!!! warning "Costs & rate limits"
    Real API calls may incur costs and are subject to tier‑specific rate limits. Set `GEMINI_BATCH_TIER` to match your billing plan.

## 4) Next steps

- Tutorials → You’ve completed Quickstart; try batching with multiple prompts using `run_batch`.
- How‑to → Configuration for env/files, profiles, and audits.
- How‑to → Troubleshooting and How‑to → FAQ for common first‑run issues.
- Reference → API overview and types (`run_simple`, `run_batch`, `types.Source`).

## Onboarding checklist

- Install from Releases or source and verify import.
- Run the Hello example in mock mode and confirm expected output.
- Set `GEMINI_API_KEY`, `GEMINI_BATCH_TIER`, `GEMINI_BATCH_USE_REAL_API=1` when ready for real calls.
- Run `gb-config doctor` until no issues are reported.
- Re-run the example; iterate with How‑to → Troubleshooting if needed.
