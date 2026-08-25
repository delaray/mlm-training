# LoRA and QLoRA

Low-Rank Adaptation (LoRA) freezes pretrained weights and learns a low-rank update:

$$
W' = W + \frac{\alpha}{r}BA
$$

where rank \(r\) is much smaller than the original matrix dimensions. QLoRA combines adapters with a quantized frozen backbone, reducing accelerator memory further.

## Project behavior

`setup_model_for_mlm_training` validates the requested device, optionally constructs a bitsandbytes 4/8-bit configuration on CUDA, prepares the model for k-bit training, and applies PEFT adapters to architecture-aware target modules. On CPU it logs a warning and falls back to an FP32 backbone.

| Mode | Backbone | Trainable state | Best use |
|---|---|---|---|
| Full tuning | FP32/FP16 | Most parameters | Maximum flexibility with ample compute |
| LoRA | FP32/FP16 frozen | Small adapters | Portable, memory-efficient experiments |
| QLoRA | 4/8-bit frozen | Small adapters | GPU memory is the main constraint |

## Important trade-offs

- Higher rank increases capacity and trainable memory.
- Target-module selection is architecture-specific and must be inspected.
- Quantization saves model memory but does not eliminate activation memory.
- Adapter-only artifacts require the correct base model and revision at load time.
- A feature-extraction or MLM task configuration should be validated against the PEFT/model versions in use.

## Extensions and improvements

- Log trainable parameter counts and peak allocated memory per trial.
- Tune rank, alpha, dropout, and target modules with constrained search spaces.
- Compare LoRA against IA3, AdaLoRA, prefix tuning, and full fine-tuning.
- Add gradient checkpointing and paged optimizers for longer contexts.
- Record quantization type, compute dtype, library versions, and GPU architecture.
- Test merged and unmerged adapter outputs for numerical parity before deployment.
