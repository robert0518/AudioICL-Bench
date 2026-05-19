# 🎧 AudioICL-Bench: A Benchmark for In-Context Learning in Audio

### The official GitHub page of the paper "AudioICL-Bench: Can Large Audio Language Models Solve Morse Code In Context? A Benchmark for Audio In-Context Learning"

- **Authors:** Jia-Hung Chen, Yi-Cheng Lin, Kai-Wei Chang, Ke-Han Lu, Hung-Yi Lee
- **Affiliations:** National Taiwan University, Taiwan; Massachusetts Institute of Technology, USA; NTU AI-CoRE, Taiwan
- **Accepted to Interspeech 2026**
- **Paper link:** *(coming soon)*

---

## Overview

![AudioICL-Bench Overview](figures/overview.png)

## Abstract

**TL;DR: We propose AudioICL-Bench, a benchmark for evaluating context-based rule learning (Task Learning) in audio models, and reveal a clear gap between acoustic perception and rule induction in current LALMs.**

Recent advances in large language models demonstrate strong in-context learning, yet it remains unclear whether audio models can learn new rules from contextual examples. Existing audio benchmarks are largely centered on task recognition, lacking rigorous assessment of a model's ability for contextual task learning. We introduce AudioICL-Bench, a benchmark designed to evaluate context-based rule learning in audio models. It includes controlled artificial tasks (e.g., counting, operator inference, dynamic remapping) and real-world scenarios such as Morse code decoding. Many tasks are constructed so that zero-shot performance is insufficient, requiring models to infer task-specific rules from demonstrations. By analyzing performance across varying numbers of examples, we reveal a clear gap between acoustic perception and rule learning, highlighting limitations of current audio models.

**🌟 Key Findings**

- Models can learn symbolic mapping rules from audio demonstrations (Fast Binding), reaching near-perfect accuracy on AudioOperator with sufficient shots.
- Models **consistently fail** at tasks requiring precise temporal measurement (AudioLength, MorseCode), revealing a fundamental temporal resolution bottleneck caused by audio token downsampling.
- Explicit instructions are mandatory — removing them causes catastrophic performance collapse across most tasks.
- Audio ICL is fundamentally harder than text ICL: audio lacks explicit unit boundaries, and temporal patterns span multiple time scales that current models cannot resolve.

---

## Dataset

Our dataset is available on HuggingFace 🤗:

> **[hongzz-18/AudioICL-Bench](https://huggingface.co/datasets/hongzz-18/AudioICL-Bench)**
