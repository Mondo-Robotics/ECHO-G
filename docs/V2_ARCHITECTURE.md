# V2 audio+text architecture

V2 is the direct-robot `qknorm_wordtime` model with schema
`release30-bounded-qk-global-local-wordtime-v2.1`. It retains tokenizer-level text keys and learns
a global/local attention mixture. It is distinct from fixed-prior B1, human-latent models, and
later experiments with word-pooled local keys.

## Backbone and acoustic alignment

The transformer has 12 blocks, hidden width 768, 8 heads, FFN width 2048, dropout 0, and learned
frame positions. The supported clip limit is 600 frames at 30 FPS; the stored position table has
capacity 608. Input and output motion dimension is 39.

```text
hidden = Linear(noised_normalized_motion)
       + Linear(LayerNorm(frame_aligned_audio))
       + learned_frame_position
text_hidden = Linear(LayerNorm(text_tokens))
```

Audio is already aligned to motion frames before entering the transformer. Each block performs
motion self-attention, text cross-attention, and an MLP, with flow-time-conditioned AdaLN and
residual gates. Both text branches share Q/K/V and output projections within a block; different
blocks retain their own parameters. The local mixture is different from the block residual gate.

## Per-head text attention

All formulas below use ordinary text. `masked_softmax` excludes padded tokens.

```text
Q_hat = L2_normalize(Q, eps=1e-6)
K_hat = L2_normalize(K, eps=1e-6)
content = temperature * (Q_hat @ transpose(K_hat))

distance = frame_seconds - token_center_seconds
raw_log_prior = -0.5 * ((distance - lag) / sigma) ** 2

global_attention = masked_softmax(content)
local_attention  = masked_softmax(content + clamp(raw_log_prior, min=-12))

support = max_over_valid_tokens(exp(raw_log_prior))
effective_mix = mix * support
attention = (1 - effective_mix) * global_attention + effective_mix * local_attention
output = output_projection(concatenated_heads(attention @ V))
```

Q/K normalization acts on each head's 96-dimensional vectors; V is not QK-normalized. The
Gaussian support uses the **unclipped** prior, while local attention uses the clipped log prior.
The attention distributions are mixed after softmax. Substituting one softmax over scaled time
biases changes the method.

A positive lag places the preferred motion frame after the token center; a negative lag places
it before. The sign of `distance` is therefore required. Frame position, real-time distance in
seconds, and flow time in [0, 1] are different inputs.

## Learned bounds

There is one raw scalar for each parameter, layer, and head:

| Parameter | Transformation | Range | Initialization |
|---|---|---|---:|
| Temperature | `1 + 15 * sigmoid(raw)` | 1–16 | 8 |
| Sigma, seconds | `0.25 + 1.75 * sigmoid(raw)` | 0.25–2 | 0.75 |
| Lag, seconds | `0.5 * tanh(raw)` | −0.5–0.5 | 0 |
| Mix | `0.5 * sigmoid(raw)` | 0–0.5 | 0.1 |

The checkpoint supplies trained values; initialization values are not inference constants.
Support decreases the local contribution when no token is near the frame. Global attention can
still read the entire transcript. Missing historical timing is represented by zero distances,
so local/global attention coincide mathematically; raw timed inference requires actual timestamps.

## Learning and sampling

Physical motion is normalized with the frozen drop-training statistics. For Gaussian noise `z`,
normalized motion `x`, and sampled flow time `u`, training uses `x_u = u*x + (1-u)*z` and target
velocity `x-z`. The objective is valid-frame flow MSE plus 0.5 times adjacent-frame velocity
difference MSE. This temporal loss is not world-space jerk.

Sampling starts from stem/seed-derived Gaussian noise and performs 8 Euler updates, using EMA
weights and CFG=1. Fixed seeds make individual runs repeatable; the model remains a stochastic
generative model. Physical references are obtained by one inverse normalization at export.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full recipe and checkpoint identity.
