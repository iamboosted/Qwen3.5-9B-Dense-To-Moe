[README.md](https://github.com/user-attachments/files/28327124/README.md)
# Dense-to-MoE Conversion of Qwen 3.5 9B: A Negative Result

**Author:** [iAmBoosted](https://huggingface.co/iAmBoosted)

An attempt to convert Qwen 3.5 9B (a DeltaNet hybrid model) from a dense architecture to Mixture-of-Experts using post-hoc MoEfication. This would have been the first MoEfication of a DeltaNet hybrid model. The conversion succeeded structurally but failed to produce a usable model. This repository documents what was tried, what failed, and why — so others don't repeat the same mistakes.

## Model Background

Qwen 3.5 9B is a 9.6B parameter model with a hybrid attention architecture: 24 DeltaNet (linear attention) layers and 8 full attention layers across 32 total layers. Every layer uses a standard SwiGLU MLP with 12,288 intermediate neurons. The MLP accounts for ~50% of total parameters (4.83B).

### Parameter Distribution

| Component | Params | % of Total |
|-----------|--------|------------|
| MLP (32 layers) | 4.83B | 50.1% |
| DeltaNet attention (24 layers) | 1.62B | 16.8% |
| Embeddings | 1.02B | 10.6% |
| lm_head | 1.02B | 10.5% |
| Full attention (8 layers) | 0.47B | 4.9% |
| Other (norms, MTP, etc.) | 0.69B | 7.1% |

## What Was Attempted

### Approach 1: CMoE (Direct Slice-and-Route)

Based on [CMoE (Pei et al., ACL 2026 Main)](https://github.com/JarvisPei/CMoE), which profiles neuron activation patterns, clusters co-activating neurons via balanced k-means, slices MLP weights into expert groups, and initializes routers analytically from activation statistics.

**Configuration tested:** E16 S2 A4 — 16 total experts per layer, 2 shared (always active), 4 routed experts activated per token. Target: ~5.5B active parameters per token (down from 9.6B dense).

**Pipeline:**
1. Profiled neuron activations on 1024 calibration samples (fp16, A100 80GB)
2. Clustered 12,288 neurons into 16 expert groups per layer using balanced k-means
3. Sliced gate_proj, up_proj, and down_proj weights along the neuron dimension
4. Initialized routers from activation statistics

**Result:** Conversion completed successfully on all 32 layers. Pre-fine-tune output was completely degenerate (repetition loops), which is expected per the CMoE paper. The problem was that no amount of post-hoc fine-tuning could recover model quality.

### Fine-Tuning Attempts

**Router-only training (3.67M trainable params):**
- 2048 samples, 2 epochs on C4
- Loss plateaued at ~4.92
- Output: topically aware but incoherent ("Gravity is the force that pulls objects down to earth" was the best it managed)

**Expert + router training (4.83B trainable params):**
- Unfroze all expert weights with differential learning rate (experts: 5e-5, routers: 1e-3)
- 8-bit AdamW to fit in 80GB VRAM
- Loss dropped to ~3.68 at 400 steps, then to ~2.40 in continuation training
- **Loss began diverging after step 200 of continuation**, climbing back to 3.1+
- Output at lowest loss: fluent English but zero factual knowledge retained. The model learned to generate plausible-sounding text but lost everything the dense parent knew.

### Approach 2: D2DMoE (Sparsify-Then-Slice)

Based on [D2DMoE (NeurIPS 2024)](https://arxiv.org/abs/2405.14297), which adds L1 regularization to MLP intermediate activations before conversion. This forces the model to learn to use fewer neurons per token, creating natural sparsity that makes subsequent clustering far more effective.

**Implementation:** Registered forward hooks on each MLP's down_proj to capture the intermediate activation (SiLU(gate_proj(x)) × up_proj(x)), then added an L1 penalty term to the training loss.

**Result:** After 1000 steps with λ=1e-3, activation sparsity reached only 23.4% and plateaued. Increasing λ was the obvious next step, but model quality was already degrading (incorrect factual answers, incomplete reasoning) while sparsity remained flat. The L1 penalty was disrupting learned representations without successfully killing neurons.

## Why It Failed

### The Core Problem: SwiGLU Has Zero Natural Activation Sparsity

ReLU-based models naturally produce ~50% zero activations (negative inputs become zero). This means roughly half the neurons are already "off" for any given token, making it straightforward to identify which neurons matter and group them into experts.

SwiGLU (SiLU activation) never outputs exact zeros. Every single neuron contributes something to every token. When we profiled all 32 layers, the activation magnitude distribution was nearly flat for early and middle layers. There are no natural "dead" neurons to cluster around.

This means:
- **Clustering is noisy:** With no natural grouping signal, k-means produces arbitrary cuts through a continuous distribution rather than finding natural expert boundaries.
- **Weight slicing destroys distributed representations:** When every neuron participates in every computation, cutting the MLP into pieces breaks interdependencies that can't be recovered by training routers alone.
- **L1 regularization struggles to create sparsity:** The SwiGLU activation function resists being pushed to zero because both the gate and up projections must simultaneously become negligible for a neuron to truly "die." This requires substantially more training than the few thousand steps we attempted.

### What the Literature Missed

The CMoE paper reports successful conversion with only ~2K samples of router fine-tuning. However, their experiments were on architectures with more favorable sparsity characteristics. The paper's claim that relative importance (top-K per token) sidesteps the absolute sparsity problem is technically true for routing decisions, but doesn't address the fundamental issue: if the original model uses all neurons for every token, slicing those neurons into groups and only activating a subset per token destroys information regardless of how well the router selects.

The D2DMoE paper demonstrates successful sparsification on models like Gemma-2B, but achieving meaningful sparsity on SwiGLU at the 9B scale appears to require training budgets (billions of tokens, extended L1 schedules) far beyond what's feasible on consumer or moderate cloud compute.

## Key Findings

1. **Post-hoc MoEfication of SwiGLU models without prior sparsification does not work.** Router-only training cannot compensate for information loss in the weight slicing step. Full expert fine-tuning drives loss down but the model learns fluency without retaining knowledge.

2. **L1 sparsification of SwiGLU at λ=1e-3 is insufficient.** 1000 steps produced only 23.4% sparsity with quality degradation already visible. The activation function's dual-gate structure resists simple L1 pressure.

3. **Loss is not a reliable quality indicator for MoE-converted models.** Loss reached 2.40 (close to the dense parent's range) while output was completely incoherent. The model was fitting the training distribution (C4 web text) without retaining the factual knowledge and reasoning capabilities of the original.

4. **The DeltaNet attention backbone is not the problem.** DeltaNet layers were left untouched throughout all experiments. The failure is entirely in the MLP conversion, which uses standard SwiGLU identical to vanilla transformer models. The same issues would apply to any SwiGLU model.

## What Might Work

For anyone attempting dense-to-MoE conversion on SwiGLU models, the following approaches may be more promising:

- **Aggressive L1 with extended training:** λ=1e-1 or higher, trained for tens of thousands of steps or more. The D2DMoE paper's results suggest this works but requires substantially more compute than we allocated.
- **ReLU substitution:** Replace SiLU with ReLU across all MLPs, fine-tune to recover quality, then MoEfy. ProSparse (2024) demonstrated this but required 134B tokens for a 13B model — well beyond consumer budgets.
- **Target ReLU-based models instead:** Models with natural activation sparsity (ReLU, GELU variants) are dramatically better candidates for post-hoc MoEfication.
- **Use natively-MoE architectures:** Production MoE models (Qwen3 MoE, Mixtral, DeepSeek) achieve their efficiency because MoE was part of the architecture from the start, not retrofitted.

## Hardware and Compute Used

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Initial conversion + router training | RTX 3060 12GB (local) | ~16 hours | $0 |
| Full pipeline (fp16, 1024 calibration) | A100 SXM 80GB (RunPod) | ~4 hours | ~$6 |
| Expert fine-tuning | A100 SXM 80GB (RunPod) | ~3 hours | ~$5 |
| D2DMoE sparsification | A100 SXM 80GB (RunPod) | ~1.5 hours | ~$2 |
| **Total** | | **~24.5 hours** | **~$13** |

## Methodology Attribution

- **CMoE:** Pei et al., "CMoE: Fast and Efficient Construction of Mixture-of-Experts Models," ACL 2026 Main (Oral). [GitHub](https://github.com/JarvisPei/CMoE). MIT License.
- **D2DMoE:** Incomplete application of the D2DMoE methodology (NeurIPS 2024). The sparsification step was attempted but did not reach the convergence demonstrated in the original paper.

## Related Work by This Author

- [Falcon-H1 7B SLERP Merge](https://huggingface.co/iAmBoosted/falcon-h1-7b-instruct-x-h1r-slerp) — First published SLERP merge of Mamba-2 hybrid models
- [Falcon-H1 1.5B-Deep Reasoning](https://huggingface.co/iAmBoosted/falcon-h1-1.5b-deep-reasoning) — QLoRA math reasoning fine-tune (+30% relative improvement)
- [Zamba2 SLERP Merge](https://github.com/iamboosted/Zamba2-SLERP-Merge) — Demonstrated that weight-sharing architectures resist standard merge tooling

## License

This documentation is released under MIT License. No model weights are distributed.
