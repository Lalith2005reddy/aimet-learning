# AIMET Learning Curriculum (Session-by-Session)

## Context

I cloned the Qualcomm AIMET repository locally and I am a beginner to AIMET, but I already know Machine Learning.

### Learning Goals

* Master AIMET progressively over multiple sessions.
* Focus specifically on the **PyTorch side** of AIMET.
* Learn through **hands-on implementation**, not just theory.
* Install AIMET and run real notebooks/scripts during each session.
* Skip general quantization theory because it is already understood.
* Learn **AIMET v2 APIs only**; avoid legacy v1 APIs unless specifically needed for comparison.

### Repository

The AIMET repository is located inside:

```text
Next_gen_AIMET/
└── aimet/
```

### Important Repository Facts

* **Core PyTorch package:**

  ```text
  TrainingExtensions/torch/src/python/aimet_torch/
  ```

* **Current API:**

  ```text
  aimet_torch/v2/
  ```

* **Legacy API:**

  ```text
  aimet_torch/v1/
  ```

  We will primarily learn **v2 only**.

* **C++ core:**

  ```text
  ModelOptimizations/DlQuantization/
  ```

  This is exposed to Python through `pybind11` as `libpymo`.

  We only need to understand this at a high level. We will not deep-dive into the C++ implementation unless required.

* **Examples:**

  ```text
  Examples/torch/quantization/
  Examples/torch/compression/
  Examples/torch/v2/
  ```

* **Documentation:**

  ```text
  Docs/ptq_techniques/
  Docs/techniques/
  Docs/tutorials/
  ```

* **Build/install:**

  According to `CLAUDE.md`, the editable installation can be done using:

  ```bash
  pip install --no-build-isolation -e .
  ```

  CMake can also be used when required.

---

# Session Plan

## Session 1 — Setup + Repository Orientation

### Topics

* Install AIMET from source.
* Verify that AIMET is correctly installed.
* Understand the repository structure.
* Map the architecture described in `CLAUDE.md` to the actual repository.
* Run the simplest AIMET example.

### Files/Resources

```text
CLAUDE.md

Examples/torch/v2/quickstart_guide.py

TrainingExtensions/torch/src/python/aimet_torch/
```

### Tasks

1. Install AIMET using:

   ```bash
   pip install --no-build-isolation -e .
   ```

2. Verify:

   ```python
   import aimet_torch
   ```

3. Explore the repository structure.

4. Run:

   ```text
   Examples/torch/v2/quickstart_guide.py
   ```

### Goal

By the end of this session:

* AIMET should be installed.
* `import aimet_torch` should work.
* The quickstart example should run.
* We should understand where the important AIMET components live.

### Verification

The session is successful if:

```python
import aimet_torch
```

works without errors and the quickstart script executes successfully.

### GPU Note

If a GPU is unavailable, use the CPU-only configuration:

```text
ENABLE_CUDA=OFF
```

---

# Session 2 — QuantizationSimModel Fundamentals (PTQ Basics)

## Topics

* Understand `QuantizationSimModel`.
* Understand the AIMET v2 quantization simulation flow.
* Apply simulated quantization to a small pretrained PyTorch model.
* Compare FP32 and simulated-quantized model accuracy.
* Understand the important quantization configuration options.

### Files/Resources

Documentation:

```text
Docs/tutorials/quantsim.rst
```

Source:

```text
TrainingExtensions/torch/src/python/aimet_torch/v2/quantsim/
```

### Hands-On

Use a small pretrained model such as:

```text
torchvision.models.resnet18
```

Create a `QuantizationSimModel` and compare:

```text
FP32 model
      ↓
QuantizationSimModel
      ↓
Simulated quantized model
```

### Concepts to Understand

* Bitwidth
* Per-tensor quantization
* Per-channel quantization
* Symmetric quantization
* Asymmetric quantization
* Quantization encodings
* Activation quantization
* Weight quantization
* Calibration

### Experiment

Change one configuration at a time:

```text
8-bit → 6-bit
per-tensor → per-channel
symmetric → asymmetric
```

Observe the impact on model accuracy.

### Goal

Understand the basic AIMET v2 PTQ workflow and how `QuantizationSimModel` represents quantization during simulation.

---

# Session 3 — Cross-Layer Equalization (CLE) + BatchNorm Folding

## Topics

* Cross-Layer Equalization.
* BatchNorm folding.
* Why these techniques are useful before quantization.
* How AIMET modifies model weights before quantization.

### Files/Resources

```text
Docs/ptq_techniques/cle.rst
Docs/ptq_techniques/bn.rst
Docs/ptq_techniques/bnf.rst
```

### Source

Explore the relevant AIMET PyTorch implementation under:

```text
TrainingExtensions/torch/src/python/aimet_torch/
```

### Hands-On

Use a model containing convolution/BatchNorm layers.

Apply:

```text
BatchNorm Folding
        ↓
Cross-Layer Equalization
        ↓
Quantization
```

### Experiment

Find layers with significantly different channel ranges.

Compare:

```text
Before CLE
vs
After CLE
```

Then observe the effect on simulated quantization accuracy.

### Goal

Understand how AIMET modifies the FP32 model before quantization to make the model more quantization-friendly.

---

# Session 4 — AdaRound

## Topics

* Understand AdaRound.
* Understand why standard rounding can hurt accuracy.
* Understand adaptive rounding.
* Understand the optimization loop used by AdaRound.

### Files/Resources

Source:

```text
TrainingExtensions/torch/src/python/aimet_torch/adaround/
```

Documentation:

```text
Docs/ptq_techniques/adaround.rst
```

Examples:

```text
Examples/torch/quantization/*adaround*.ipynb
```

### Hands-On

Run an AdaRound notebook.

Compare:

```text
Normal PTQ rounding
vs
AdaRound
```

### Things to Inspect

* Layer-wise optimization.
* Rounding decisions.
* Reconstruction loss.
* Optimization iterations.
* Impact on final model accuracy.

### Goal

Understand AdaRound as a PTQ technique that optimizes weight rounding rather than blindly rounding every weight.

---

# Session 5 — Sequential MSE (SeqMSE) + BatchNorm Reestimation

## Topics

* Sequential MSE.
* BatchNorm reestimation.
* When these techniques are useful.
* Comparison with AdaRound.

### Hands-On

Run the relevant BatchNorm reestimation example/notebook.

Study the Sequential MSE concept briefly, including its ONNX-side implementation where appropriate.

### Compare

```text
Basic PTQ
AdaRound
SeqMSE
BN Reestimation
```

Understand:

* What problem each technique solves.
* When each technique should be preferred.
* How they affect quantized model accuracy.

### Goal

Build intuition for choosing between different PTQ accuracy-recovery techniques.

---

# Session 6 — Quantization-Aware Training (QAT)

## Topics

* Understand AIMET QAT.
* Fine-tune a quantized model.
* Compare QAT against pure PTQ.
* Understand range learning.

### Documentation

```text
Docs/techniques/qat.rst
```

### Examples

```text
Examples/torch/v2/qat.ipynb
```

Also investigate:

```text
QAT range-learning variant
```

### Hands-On Pipeline

```text
FP32 Model
    ↓
PTQ
    ↓
Quantized Model
    ↓
QAT Fine-Tuning
    ↓
Recovered Accuracy
```

### Experiment

Compare:

```text
FP32
PTQ
QAT
```

Measure accuracy after each stage.

### Goal

Understand how QAT recovers quantization-induced accuracy loss through training.

---

# Session 7 — Automatic Mixed Precision (AMP)

## Topics

* Automatic Mixed Precision in AIMET.
* Layer-wise sensitivity.
* Assigning different bitwidths to different layers.
* Accuracy/performance trade-offs.

### Source

```text
TrainingExtensions/torch/src/python/aimet_torch/amp/
```

and:

```text
aimet_torch/v2/mixed_precision/
```

### Hands-On

Run the AIMET AMP notebook/example.

Allow AIMET to determine an appropriate precision configuration.

For example:

```text
Layer A → INT8
Layer B → INT8
Layer C → INT4
Layer D → INT8
Layer E → INT16
```

depending on the supported configuration.

### Goal

Understand how AIMET automatically chooses precision assignments instead of using one bitwidth for the entire model.

---

# Session 8 — QuantAnalyzer + Visualization Tools

## Topics

* Quantization sensitivity analysis.
* QuantAnalyzer.
* Visualization tools.
* Layer-by-layer analysis.
* Min/max statistics.
* Histograms.

### Source

```text
TrainingExtensions/torch/src/python/aimet_torch/v2/visualization_tools/
```

### Hands-On

Run QuantAnalyzer on a model.

Inspect:

```text
Layer sensitivity
Activation statistics
Weight statistics
Min/max values
Histograms
Quantization impact
```

### Goal

Learn how to answer:

> "Which layers are causing the accuracy loss after quantization?"

This session is focused on **diagnosing quantization problems** rather than simply applying quantization.

---

# Session 9 — Model Compression

## Topics

Study the major AIMET model-compression techniques:

1. Spatial SVD
2. Weight SVD
3. Channel Pruning
4. Winnow

### Source Directories

```text
TrainingExtensions/torch/src/python/aimet_torch/svd/
TrainingExtensions/torch/src/python/aimet_torch/channel_pruning/
TrainingExtensions/torch/src/python/aimet_torch/winnow/
```

### Examples

```text
Examples/torch/compression/*.ipynb
```

### Hands-On

Take a model and perform compression.

Measure:

```text
Original MACs
Compressed MACs

Original model size
Compressed model size

Original accuracy
Compressed accuracy
```

### Goal

Understand the trade-off between:

```text
Compression
    ↕
Computational cost
    ↕
Accuracy
```

---

# Session 10 — Architecture Checker + Model Validator

## Topics

* Architecture Checker.
* Model Validator.
* Detecting unsupported/problematic model structures.
* Preparing a custom model before quantization.

### Source

```text
TrainingExtensions/torch/src/python/aimet_torch/arch_checker/
TrainingExtensions/torch/src/python/aimet_torch/model_validator/
```

### Hands-On

Create/use a custom PyTorch model.

Run:

```text
Model Validator
        ↓
Architecture Checker
        ↓
Quantization
```

Intentionally introduce a model structure that may cause a problem and observe the validation/checker output.

### Goal

Learn how to perform **pre-flight checks** before attempting quantization on a custom model.

---

# Session 11 — Experimental Frontier

## Topics

Survey AIMET's experimental techniques:

* OmniQuant
* SpinQuant
* AdaScale
* GPTVQ
* LoRA quantization

### Source

Explore relevant:

```text
experimental/
```

directories.

### Approach

This session is primarily a survey.

For each technique:

1. Understand the problem it solves.
2. Understand the high-level idea.
3. Identify what type of model it targets.
4. Inspect the implementation.
5. Find an example if available.
6. Run one technique if the environment/example supports it.

### Focus

Pay particular attention to:

```text
GenAI
LLMs
Transformer quantization
Parameter-efficient fine-tuning
```

because the repository contains GenAI-related components.

### Goal

Understand where AIMET is heading beyond traditional CNN/vision-model quantization.

---

# Session 12 — End-to-End AIMET Pipeline

## Final Project

Put everything learned so far together on a real model.

### Model

Choose one:

```text
Small vision model
```

or, if the environment supports it:

```text
Small LLM / GenAI model
```

### Full Pipeline

```text
                ┌─────────────────┐
                │   FP32 Model    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Model Validator │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Architecture    │
                │ Checker          │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ BN Folding      │
                │ + CLE            │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ QuantSim / PTQ  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │   AdaRound      │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │      QAT        │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Export/Deploy   │
                │ Quantized Model │
                └─────────────────┘
```

### Measure

At every major stage record:

```text
Accuracy
Model size
MACs
Bitwidth
Quantization configuration
```

### Final Goal

Be able to take a new PyTorch model and independently answer:

* Is this model compatible with AIMET?
* How should I prepare it for quantization?
* Which PTQ technique should I use?
* Where is quantization hurting accuracy?
* Should I use AdaRound?
* Should I use QAT?
* Should I use mixed precision?
* How can I diagnose problematic layers?
* How much compression can I achieve?
* How do I export/deploy the final quantized model?

---

# How Each Session Will Run

Every session follows the same workflow.

## Step 1 — Locate the Relevant Code

Identify the exact AIMET source files related to the current topic.

Example:

```text
Topic
  ↓
Documentation
  ↓
AIMET source code
  ↓
Example notebook/script
```

---

## Step 2 — Read the Documentation

Read the relevant AIMET documentation first.

For example:

```text
Docs/tutorials/
Docs/ptq_techniques/
Docs/techniques/
```

The goal is to understand **what AIMET intends the feature to do**.

---

## Step 3 — Walk Through the Source Code

Read the actual implementation.

We should identify:

* Main classes.
* Main functions.
* Important parameters.
* Internal workflow.
* Important helper functions.
* How the v2 API connects the pieces.

The goal is not to memorize every line.

Instead, understand:

```text
Input
  ↓
Main API
  ↓
Internal processing
  ↓
Output
```

---

## Step 4 — Run a Real Example

Run the corresponding:

```text
.ipynb
```

or:

```text
.py
```

example.

Whenever possible, use a small PyTorch model so that execution is fast.

---

## Step 5 — Make a Small Modification

Change something yourself.

Examples:

```text
8-bit → 4-bit
```

or:

```text
ResNet18 → another small model
```

or:

```text
Change calibration samples
```

or:

```text
Change a quantization configuration
```

The purpose is to confirm that the concept is actually understood.

---

## Step 6 — Verify the Result

Record real numbers.

Depending on the session:

```text
Accuracy
Loss
MACs
Model size
Layer sensitivity
Quantization encodings
Compression ratio
```

The goal is always to connect:

```text
Code
  ↓
AIMET operation
  ↓
Numerical result
  ↓
Understanding
```

---

## Step 7 — Recap

At the end of every session:

### What we learned

Short summary of the important concepts.

### What we implemented

The actual AIMET API/code used.

### What we verified

The numerical/experimental result.

### Next session

Preview of the next topic.

---

# Verification Strategy

Every session should contain a self-verifying hands-on component.

Examples:

### Session 1

```text
import aimet_torch
```

must succeed.

Quickstart script must run successfully.

### Session 2

Compare:

```text
FP32 accuracy
vs
Simulated quantized accuracy
```

### Session 3

Compare:

```text
Before CLE
vs
After CLE
```

and observe quantization behavior.

### Session 4

Compare:

```text
Standard rounding
vs
AdaRound
```

### Session 6

Compare:

```text
PTQ
vs
QAT
```

### Session 7

Compare:

```text
Uniform precision
vs
AMP mixed precision
```

### Session 9

Compare:

```text
Original MACs
vs
Compressed MACs
```

---

# Progress Tracker

Use this checklist to track learning progress.

* [ ] Session 1 — Setup + Repository Orientation
* [ ] Session 2 — QuantizationSimModel Fundamentals
* [ ] Session 3 — CLE + BatchNorm Folding
* [ ] Session 4 — AdaRound
* [ ] Session 5 — SeqMSE + BN Reestimation
* [ ] Session 6 — QAT
* [ ] Session 7 — AMP
* [ ] Session 8 — QuantAnalyzer + Visualization
* [ ] Session 9 — Model Compression
* [ ] Session 10 — Architecture Checker + Model Validator
* [ ] Session 11 — Experimental Frontier
* [ ] Session 12 — End-to-End AIMET Pipeline

---

# Learning Principles

## 1. Learn From the Current AIMET API

Prefer:

```text
aimet_torch/v2/
```

over:

```text
aimet_torch/v1/
```

The v1 API should only be discussed when understanding an important compatibility or architectural difference.

---

## 2. Code Before Excessive Theory

Since general ML and quantization fundamentals are already known, prioritize:

```text
AIMET documentation
        ↓
AIMET source
        ↓
Real example
        ↓
Modification
        ↓
Result
```

rather than spending large amounts of time reviewing generic quantization theory.

---

## 3. Use Small Models During Learning

Prefer models that allow fast iteration.

For example:

```text
ResNet18
MobileNet
Small CNN
Small Transformer
```

The objective is learning AIMET, not training a large model from scratch.

---

## 4. Understand the Source, Not Just the API

For every major AIMET feature, understand:

```text
What problem does it solve?
        ↓
What does the public API do?
        ↓
What happens internally?
        ↓
What changes in the model?
        ↓
What changes in the output?
```

---

## 5. Always Verify With Experiments

Whenever possible, don't stop at:

> "This technique should improve accuracy."

Instead, actually measure:

```text
Before
vs
After
```

using a real model.

---

# Important Repository Areas

Keep this as a quick reference while learning.

```text
aimet/
│
├── CLAUDE.md
│
├── Examples/
│   └── torch/
│       ├── quantization/
│       ├── compression/
│       └── v2/
│
├── Docs/
│   ├── ptq_techniques/
│   ├── techniques/
│   └── tutorials/
│
├── ModelOptimizations/
│   └── DlQuantization/
│
└── TrainingExtensions/
    └── torch/
        └── src/
            └── python/
                └── aimet_torch/
                    ├── v1/
                    ├── v2/
                    ├── adaround/
                    ├── amp/
                    ├── svd/
                    ├── channel_pruning/
                    ├── winnow/
                    ├── arch_checker/
                    └── model_validator/
```

---

# Important Architecture Concept

At a high level, AIMET can be viewed as:

```text
                 PyTorch Model
                       │
                       ↓
              ┌─────────────────┐
              │  AIMET PyTorch  │
              │      APIs       │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       PTQ/QAT       AMP       Compression
          │            │            │
          └────────────┼────────────┘
                       ↓
              AIMET Optimization
                       │
                       ↓
              Quantized/Optimized
                    Model
                       │
                       ↓
                  Deployment
```

The lower-level C++ optimization components are exposed through:

```text
libpymo
```

using `pybind11`.

We will understand this connection at a high level but will not deep-dive into the C++ implementation unless needed.

---

# Expected Outcome After Session 12

After completing this curriculum, the target is to be comfortable with the AIMET PyTorch workflow:

```text
                 New PyTorch Model
                        │
                        ↓
                 Validate Model
                        │
                        ↓
              Architecture Checker
                        │
                        ↓
                 Model Preparation
                        │
                 ┌──────┴──────┐
                 ↓             ↓
             BN Folding       CLE
                 └──────┬──────┘
                        ↓
                     PTQ
                        ↓
               QuantizationSim
                        │
              ┌─────────┼─────────┐
              ↓         ↓         ↓
          AdaRound     SeqMSE    Other PTQ
              └─────────┼─────────┘
                        ↓
                 Accuracy Analysis
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
            QAT                  AMP
              │                   │
              └─────────┬─────────┘
                        ↓
                Final Quantized Model
                        │
                        ↓
                  Export/Deploy
```

The ultimate goal is not merely to know the names of AIMET techniques, but to be able to **read AIMET source code, use its v2 APIs, debug issues, run experiments, interpret results, and build a complete quantization/compression pipeline independently.**
