# Qwen3.6 35B-A3B MoE conversion

`converter.qwen36_moe` converts RavenX's **BF16 or MLX affine 4-bit** checkpoint
into Splash's schema-4 `splash-packed-q4-moe` target weights. The MLX 4-bit
path repacks existing unsigned Q4 codes, BF16 scales and BF16 biases directly;
MLX's 8-bit router and shared-expert gate are also repacked without a second
quantization. The converter streams sharded safetensors one tensor, expert or
256 vocabulary rows at a time. The MLX 8-bit whole-model variant is not
supported.

## Inputs and commands

Download either RavenX checkpoint and the official Splash package, or point
at existing local copies:

```bash
hf download deadbydawn101/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-mlx \
  --local-dir /models/ravenx-bf16
hf download incoai/Qwen3.6-35B-A3B-Splash \
  --local-dir /models/qwen36-splash-reference

# Smaller MLX affine 4-bit source, if using that path:
hf download Ck-TnT/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-mlx-mlx-4Bit \
  --local-dir /models/ravenx-mlx-4bit

python -m converter.qwen36_moe inspect \
  --source /models/ravenx-bf16 --model-type qwen36-moe
python -m converter.qwen36_moe build \
  --source /models/ravenx-bf16 \
  --out /models/ravenx-splash-q4 \
  --reference /models/qwen36-splash-reference \
  --model-type qwen36-moe
python -m converter.qwen36_moe validate \
  --source /models/ravenx-bf16 --out /models/ravenx-splash-q4
```

For the smaller MLX 4-bit checkpoint, replace `/models/ravenx-bf16` with its
directory. `build --resume` skips target files whose size and header are
complete. A target-only build can be assembled later without repacking weights:

```bash
python -m converter.qwen36_moe build \
  --source /models/ravenx-mlx-4bit --out /models/ravenx-splash-q4 \
  --target-only --resume
python -m converter.qwen36_moe assemble \
  --source /models/ravenx-mlx-4bit --out /models/ravenx-splash-q4 \
  --reference /models/qwen36-splash-reference
python -m converter.qwen36_moe validate \
  --source /models/ravenx-mlx-4bit --out /models/ravenx-splash-q4
```

`build --target-only` creates the 42 target files without copying reference
assets. That is useful for inspecting weights, but current Splash cannot load
a target-only package: its installer and runtime require draft and vision
assets. A full build copies the compatible six-layer draft and vision tower
from the official package and retains RavenX's own tokenizer, config and chat
template. `tokenizer/vocab.json` is derived from RavenX's `tokenizer.json`.

## Source and binary mapping

Every layer starts with a 16-byte header: `MDFM0001`, little-endian layer
index, and type 0 (GDN) or 1 (full attention). Sections begin on 16 KiB
boundaries, in the exact order below. `target/head.bin` uses `MDFM0002` and
`target/embedding.bin` uses `MDFE0001`. The 30 GDN and 10 attention layer files
are 475,201,536 and 471,285,760 bytes respectively, matching the official
Qwen3.6 package. Head and embedding files are 286,097,408 and 286,081,024
bytes. Total target weights: **19,541,082,112 bytes (18.199 GiB)**.

| Splash section | RavenX tensor(s) | Encoding |
| --- | --- | --- |
| input-norm / post-attention-norm | `input_layernorm.weight` / `post_attention_layernorm.weight` | BF16 identity |
| gdn-input | `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` | Q4, concatenated in that order, 192 zero rows |
| gdn-convolution / decay / time-bias / norm | `linear_attn.conv1d.weight`, `A_log`, `dt_bias`, `norm.weight` | BF16, FP32 `-exp(A_log)`, BF16, BF16 |
| gdn-output | `linear_attn.out_proj.weight` | Q4 |
| attention-input | `self_attn.q_proj`, `k_proj`, `v_proj` | Q4, concatenated in that order |
| query-norm / key-norm / attention-output | `self_attn.q_norm`, `k_norm`, `o_proj` | BF16 identity, BF16 identity, Q4 |
| router | `mlp.gate.weight` | Q8 affine |
| experts-gate / up / down | `mlp.switch_mlp.{gate,up,down}_proj.weight` | 256 consecutive Q4 expert slabs in source index order |
| shared-expert-gate / up / down | `mlp.shared_expert.{gate,up,down}_proj.weight` | one Q4 slab each |
| shared-expert-scalar-gate | `mlp.shared_expert_gate.weight` | Q8, first row live, 255 zero rows |
| final-norm / logits | `language_model.model.norm.weight` / `language_model.lm_head.weight` | BF16 identity / Q4 |
| embedding | `language_model.model.embed_tokens.weight` | row-major Q4 components |

All layer tensors above have the prefix `language_model.model.layers.N.`.
Q4 is unsigned four-bit affine, grouped by 64 input values, with BF16 scale
and bias. Each projection stores `[codes][scales][biases]`; codes and
parameters are tile-major by 256 output rows, then input group, then row in
tile. One routed expert occupies a complete projection slab before the next
expert. Embedding codes and parameters are instead row-major, each component
in a separate aligned section. Router Q8 uses the same 256-row and 64-input
group traversal, with one byte per code and BF16 scale/bias.

For BF16 input, the existing Q4 encoder's zero-anchored affine rule is reused;
the router uses min/max Q8 affine. Real RavenX BF16 samples gave relative RMSE
0.080-0.108 for Q4 projection families and 0.0062 for the router. For MLX
4-bit input, the existing quantized values are preserved, so repacking should
have zero numeric error relative to MLX dequantization. Unlike Qwen3.8,
**Qwen3.6 norm tensors are copied directly**. Their bytes were checked
against the official package. GDN decay is FP32 `-exp(A_log)`.

## Validation and runtime limits

`inspect` checks all required names, dtypes and shapes in shard headers and
prints section offsets and estimated sizes. `verify` checks file sizes,
headers, reference `layout.json` offsets when present, and every artifact's
size/hash in a full package. `validate` adds independent row-offset Q4/Q8
decoding against source BF16 or dequantized MLX weights for embeddings, GDN,
attention, router, first, middle and last routed experts, shared experts and
output head; it also checks norm, convolution, bias and decay bytes. Run
`python -m unittest tests.test_qwen36_moe -v` for small format tests.

After starting Splash with the package on Apple Silicon, run:

```bash
python scripts/validate_qwen36_runtime.py \
  --base-url http://127.0.0.1:8000 \
  --out /models/ravenx-runtime-report.json
```

The script reads `/status`, sends short generation, system and tool-call
prompts, checks that replies are nonempty, and saves complete responses for
review. Use `--mlx-answers /path/to/answers.json` to place original MLX answers
beside the Splash responses in the report. The script cannot turn off drafting because
Splash does not expose that mode.

For a local package copied onto a Mac, the installed Splash runtime can be
started directly without uploading or downloading the weights through the Hub:

```bash
PKG="$HOME/Models/RavenX-Splash-Q4"
SPLASH_ROOT="$(brew --prefix splash)/libexec"
"$SPLASH_ROOT/python/bin/python3" -u "$SPLASH_ROOT/server/server.py" \
  "$PKG/target" "$PKG/draft" --tokenizer "$PKG/tokenizer" \
  --model local/RavenX-Splash-Q4 \
  --binary "$SPLASH_ROOT/engine/splash" --max-context 8K
```

This bypasses Splash's Hub installer but uses its native model loader. The
package must contain `target/`, `draft/`, `vision/` and `tokenizer/` under the
same root. In another terminal, request a short Chat Completion with the
model ID `local/RavenX-Splash-Q4`.

Current Splash requires a draft even though target-only generation would be
valuable for validation. It has no public switch for draft-free serving; this
converter cannot provide that runtime mode. The official Qwen3.6 draft is
shape-compatible and speculative verification should keep target tokens
authoritative, but its speed with RavenX is unmeasured. RavenX has no vision
weights in its 733-tensor BF16 index, so the full package reuses the official
vision tower to satisfy Splash's loader. Image quality with the fine-tuned
language model is untested.

The local MLX 4-bit checkpoint was converted and assembled in this workspace.
The 42 target files total **19,541,082,112 bytes (18.199 GiB)**; the complete
package is **20,947,991,647 bytes (19.509 GiB)**. Structural and tensor
validation passed, as did Splash's installer manifest and full artifact-hash
checks. All 43 sampled Q4/Q8 decoded rows matched the source MLX checkpoint
with zero numerical error. Because the input was already quantized, this does
not measure its error against the original BF16 checkpoint. The user reports
that the source model worked in LM Studio on an M5 Pro Mac with 48 GB RAM and
a 125K configured context limit. The user subsequently ran the **converted
package in Splash** through the local server command above with an 8K context
limit: Splash reached readiness and returned a text response to the short
prompt. The exact output, runtime memory use, longer prompts, and practical
context limit have not been recorded. Draft-free generation is unsupported by
the current Splash runtime. RavenX-specific tool calling and image quality
also remain unverified.

Format sources: Splash `runtime/model/QwenTarget.hpp`, `QwenTarget.cpp`,
`Qwen3_6Moe.cpp`, `WeightStore.cpp`, and the published `layout.json` and
`manifest.json` of `incoai/Qwen3.6-35B-A3B-Splash`.
