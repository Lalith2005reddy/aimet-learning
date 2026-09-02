# AIMET Learning Curriculum (session-by-session)

## Context
User cloned Qualcomm AIMET repo locally, beginner to AIMET (knows ML already), wants full mastery built up over many sessions, not one dump. Track: PyTorch side, hands-on (install + run real notebooks/code each session), skip general quantization theory recap (assumed known).

Repo root: `aimet/` inside `Next_gen_AIMET`. Key facts gathered from exploration + `CLAUDE.md`:
- Core package: `aimet_torch` at `TrainingExtensions/torch/src/python/aimet_torch/`
- `v2/` = current API (`QuantizationSimModel`), `v1/` = legacy — teach v2 only
- C++ core (`ModelOptimizations/DlQuantization/`) exposed via pybind11 as `libpymo` — mention once, don't deep-dive unless asked
- Examples: `Examples/torch/quantization/*.ipynb`, `Examples/torch/compression/*.ipynb`, `Examples/torch/v2/`
- Docs: `Docs/ptq_techniques/`, `Docs/techniques/`, `Docs/tutorials/`
- Build: CMake or `pip install --no-build-isolation -e .` per `CLAUDE.md`

Each session = one method/concept, hands-on: read source + docs, run/modify an example notebook or small script against a real (small) PyTorch model, verify output.

## Session Plan

**Session 1 — Setup + orientation**
- Install AIMET from source (pip editable build per CLAUDE.md), verify `import aimet_torch` works
- Tour repo layout live (map `CLAUDE.md` architecture to actual folders)
- Run simplest example: `Examples/torch/v2/quickstart_guide.py`
- Goal: working environment + mental map of where things live

**Session 2 — QuantizationSimModel fundamentals (PTQ basics)**
- Read `Docs/tutorials/quantsim.rst` + `aimet_torch/v2/quantsim/`
- Quantize a small pretrained model (e.g. torchvision resnet18), compare FP32 vs simulated-quant accuracy
- Understand quant params: bitwidth, per-channel vs per-tensor, symmetric vs asymmetric

**Session 3 — Cross-Layer Equalization (CLE) + BatchNorm folding**
- `Docs/ptq_techniques/cle.rst`, `bn.rst`, `bnf.rst`
- Apply CLE to a model with bad per-channel weight ranges, show accuracy improvement pre-quant

**Session 4 — AdaRound**
- `aimet_torch/adaround/`, `Docs/ptq_techniques/adaround.rst`
- Run `Examples/torch/quantization/*adaround*.ipynb`, understand adaptive rounding optimization loop

**Session 5 — Sequential MSE (SeqMSE) + BN Reestimation**
- BN reestimation notebook, SeqMSE for ONNX-side concept comparison (brief), when to use vs AdaRound

**Session 6 — Quantization-Aware Training (QAT)**
- `Docs/techniques/qat.rst`, run `Examples/torch/v2/qat.ipynb` and QAT range-learning variant
- Fine-tune quantized model, compare recovery vs pure PTQ

**Session 7 — Automatic Mixed Precision (AMP)**
- `aimet_torch/amp/`, `v2/mixed_precision/`
- Run AMP notebook: assign per-layer bitwidths automatically under accuracy/perf tradeoff

**Session 8 — QuantAnalyzer + visualization tools**
- `aimet_torch/v2/visualization_tools/`
- Diagnose a model's quantization sensitivity layer-by-layer, read min-max/histogram plots

**Session 9 — Model compression: Spatial SVD, Weight SVD, Channel Pruning, Winnow**
- `svd/`, `channel_pruning/`, `winnow/` + `Examples/torch/compression/*.ipynb`
- Compress a model, measure MAC reduction vs accuracy

**Session 10 — Architecture Checker + Model Validator**
- `arch_checker/`, `model_validator/` — pre-flight checks before quantizing a custom model

**Session 11 — Experimental frontier: OmniQuant, SpinQuant, AdaScale, GPTVQ, LoRA quant**
- `experimental/` dirs — survey each briefly, run one if example exists (likely GenAI/LLM-oriented)

**Session 12 — Put it together: end-to-end pipeline on a real model**
- Pick a model (e.g. a small vision model or, given GenAILab exists, a small LLM), build full pipeline: validate → CLE/BN fold → AdaRound → QAT fine-tune → export quantized model
- This session tests everything learned combined

## How each session runs
1. Point to exact doc + source files for that session's topic
2. Read/walk the code together, explain design choices
3. Run the matching example notebook/script live, inspect real numbers
4. Small hands-on tweak (change bitwidth, swap model, etc.) to confirm understanding
5. End with a short recap of what was learned + preview of next session

## Verification
Each session's hands-on part is self-verifying: a runnable notebook/script producing before/after accuracy or MAC numbers. Session 1 verified once `import aimet_torch` succeeds and quickstart script runs without error.

## Notes
- Sessions are independent enough to pause/resume across days; revisit this plan file's list to track progress (check off completed sessions in conversation, not by editing this file's structure)
- If GPU unavailable, note CPU-only build flag (`ENABLE_CUDA=OFF`) needed at Session 1