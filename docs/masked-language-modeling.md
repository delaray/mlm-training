# Masked Language Modeling

Masked language modeling corrupts a token sequence and trains an encoder to recover selected tokens from bidirectional context. It provides a label-free objective for continued pretraining on a specialist corpus.

For masked positions \(M\), the objective is:

$$
\mathcal{L}_{MLM} = -\sum_{i \in M}\log p_\theta(x_i \mid \tilde{x})
$$

where \(x\) is the original sequence and \(\tilde{x}\) is its corrupted version.

## Pipeline behavior

The project tokenizes documents once and applies masking dynamically in `DataCollatorForLanguageModeling`. A sequence can therefore receive different masks across epochs. Validation loss is used for checkpoint selection and Optuna optimization.

| Parameter | Effect | Typical failure mode |
|---|---|---|
| `mlm_probability` | Fraction selected for corruption | Too high removes needed context |
| `max_length` | Available context and memory cost | Padding waste or truncation |
| `learning_rate` | Adaptation speed | Instability or catastrophic forgetting |
| `epochs` | Exposure to the domain | Overfitting a small corpus |
| `weight_decay` | Parameter regularization | Under/over-regularization |

## What MLM can and cannot prove

A lower domain validation loss means better masked-token prediction on that distribution. It does not automatically imply better sentence embeddings, retrieval, fairness, robustness, or factuality. Those claims require separate frozen benchmarks.

## Extensions and improvements

- Tune masking probability and use whole-word or span masking.
- Mix general-domain replay data with specialist data to reduce forgetting.
- Compare MLM with replaced-token detection for DeBERTa/ELECTRA-style models.
- Track perplexity-like diagnostics carefully; masked-token losses are not directly comparable across tokenizers.
- Add early stopping, multiple seeds, learning-rate schedules, and gradient clipping studies.
- Follow MLM with contrastive sentence-embedding training when retrieval is the target.
