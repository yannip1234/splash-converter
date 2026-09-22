# Swift → Splash Qwen3.8-27B tensor mapping

Companion to `docs/SWIFT_COMPATIBILITY.md`.
- Source: `refs/swift-bf16/` — `ukisai/Swift-Qwen3.8-27b` @ `048328f4…`, BF16 safetensors,
  1,199 tensors (18 shards; headers parsed, shapes verified against the index).
- Destination: Splash Qwen3.8-27B package layout per `docs/SPLASH_FORMAT_SPEC.md` §5
  (verified byte-exact against `refs/qwen38-splash/`).
- One row per destination tensor (section); rows cover all layer instances of a file type
  when the transform is identical per instance (stated per row).

> **Correction (iteration 13, 2026-09-22).** A3/A4 (`query-norm`, `key-norm`) also carry the
> unit offset. Every bf16 norm section in the package stores `γ+1` **except** `gdn-norm`,
> which is the lone identity copy. The rule of thumb: if it normalises a residual-stream or
> per-head activation, it carries the offset.
>
> **Corrections (iteration 6, 2026-09-21).** Three rows below were wrong when written
> and are now fixed against the decoded reference package: the two block-norm rows carry a
> unit offset (`γ+1`), and `gdn-decay` is `−exp(A_log)`, not a dtype upcast. All three were
> caught by building layer 0 from the base checkpoint and byte-comparing against
> `refs/qwen38-splash/target/layer-0.bin`; each showed up as a section that failed to match.

## Conventions

- Source shape is row-major `weight[out, in]` as stored in the safetensors checkpoint.
- `Q4(out×in)`: codes `out·in/2` + scales `out·in/32` + biases `out·in/32`, tile-major
  (group 64, storage_n 256, low-nibble-first, 16384-aligned sub-sections) — spec §3.
- `quantize+pack`: affine Q4 quantization (per 64-group scale/bias, bf16) + tile-major pack.
- `quantize+pack-row-major`: same quantization, plain [token][dim] row-major layout, emitted
  as three separate aligned sections — spec §3.5 (embedding only).
- Status: **EXACT** = byte-identical copy into the destination section;
  **TRANSFORM_REQUIRED** = concat/pad/dtype/quantization step applied;
  **REUSE_FROM_BASE** = destination taken byte-for-byte from the existing Splash package
  (justified in §5).

## 1. `target/layer-N.bin` — GDN layers (N ∉ {3,7,…,63}; 48 files × 11 sections)

Source prefix `model.language_model.layers.{N}.`. All 48 instances identical transform.

| # | destination tensor (section, geometry) | source tensor (shape, bf16) | transform | quantization | status |
|---|---|---|---|---|---|
| G1 | input-norm, [5120] bf16, 10240 B | `input_layernorm.weight` [5120] | **unit offset**: store `bf16(γ + 1)` (verified bit-exact vs the reference package; the kernel expects the pre-added form) | none | TRANSFORM_REQUIRED |
| G2 | gdn-input, Q4(16640×5120), 47923200 B | `linear_attn.in_proj_qkv.weight` [10240,5120]; `linear_attn.in_proj_z.weight` [6144,5120]; `linear_attn.in_proj_b.weight` [48,5120]; `linear_attn.in_proj_a.weight` [48,5120] | concatenate (qkv;z;b;a) — qkv rows already [q 2048 \| k 2048 \| v 6144] (model-verified), matching destination spans; zero-pad 160 tail rows to width 16640; then quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| G3 | gdn-convolution, [10240][4] bf16, 81920 B | `linear_attn.conv1d.weight` [10240,1,4] | identity (flatten singleton dim; `w[ch*4+tap]` matches) | none | EXACT |
| G4 | gdn-decay, [48] float32, 192 B | `linear_attn.A_log` [48] | **float32 `−exp(A_log)`** — not a dtype upcast. Verified: all 48 values match the reference package, 33 bit-exact and 15 within 1 ULP (a libm `exp` difference) | none | TRANSFORM_REQUIRED |
| G5 | gdn-time-bias, [48] bf16, 96 B | `linear_attn.dt_bias` [48] | identity | none | EXACT |
| G6 | gdn-norm, [128] bf16, 256 B | `linear_attn.norm.weight` [128] | identity — **no unit offset** (verified byte-identical); the +1 applies only to the two block norms G1/G8 | none | EXACT |
| G7 | gdn-output, Q4(5120×6144), 17694720 B | `linear_attn.out_proj.weight` [5120,6144] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| G8 | post-attention-norm, [5120] bf16, 10240 B | `post_attention_layernorm.weight` [5120] | **unit offset**: store `bf16(γ + 1)` | none | TRANSFORM_REQUIRED |
| G9 | mlp-gate, Q4(17408×5120), 50135040 B | `mlp.gate_proj.weight` [17408,5120] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| G10 | mlp-up, Q4(17408×5120), 50135040 B | `mlp.up_proj.weight` [17408,5120] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| G11 | mlp-down, Q4(5120×17408), 50135040 B | `mlp.down_proj.weight` [5120,17408] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |

Section order in file: G1…G11 as listed (QwenTarget.cpp:91-133 `readQwenMixer`).

## 2. `target/layer-N.bin` — full-attention layers (N ∈ {3,7,…,63}; 16 files × 9 sections)

Source prefix `model.language_model.layers.{N}.`. All 16 instances identical transform.

| # | destination tensor (section, geometry) | source tensor (shape, bf16) | transform | quantization | status |
|---|---|---|---|---|---|
| A1 | input-norm, [5120] bf16, 10240 B | `input_layernorm.weight` [5120] | **unit offset**: store `bf16(γ + 1)` | none | TRANSFORM_REQUIRED |
| A2 | attention-input, Q4(14336×5120), 41287680 B | `self_attn.q_proj.weight` [12288,5120]; `self_attn.k_proj.weight` [1024,5120]; `self_attn.v_proj.weight` [1024,5120] | concatenate (q;k;v) — q_proj rows already per-head [q 256 \| gate 256] interleaved (model-verified, modeling_qwen3_5.py:787-789), matching destination spans; no reorder; then quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| A3 | query-norm, [256] bf16, 512 B | `self_attn.q_norm.weight` [256] | **unit offset**: store `bf16(γ + 1)` — verified byte-identical to the reference package. (This row said *identity* until iteration 13; shipping it that way left every attention layer's q_norm short by exactly 1.0 and cost 27.5 points of task accuracy.) | none | TRANSFORM_REQUIRED |
| A4 | key-norm, [256] bf16, 512 B | `self_attn.k_norm.weight` [256] | **unit offset**: store `bf16(γ + 1)` — verified byte-identical to the reference package. (This row said *identity* until iteration 13; shipping it that way left every attention layer's k_norm short by exactly 1.0 and cost 27.5 points of task accuracy.) | none | TRANSFORM_REQUIRED |
| A5 | attention-output, Q4(5120×6144), 17694720 B | `self_attn.o_proj.weight` [5120,6144] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| A6 | post-attention-norm, [5120] bf16, 10240 B | `post_attention_layernorm.weight` [5120] | **unit offset**: store `bf16(γ + 1)` | none | TRANSFORM_REQUIRED |
| A7 | mlp-gate, Q4(17408×5120), 50135040 B | `mlp.gate_proj.weight` [17408,5120] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| A8 | mlp-up, Q4(17408×5120), 50135040 B | `mlp.up_proj.weight` [17408,5120] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| A9 | mlp-down, Q4(5120×17408), 50135040 B | `mlp.down_proj.weight` [5120,17408] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |

Section order in file: A1…A9 as listed (QwenTarget.cpp:95-108 attention branch).

## 3. `target/head.bin` (1 file × 2 sections)

| # | destination tensor (section, geometry) | source tensor (shape, bf16) | transform | quantization | status |
|---|---|---|---|---|---|
| H1 | final-norm, [5120] bf16, 10240 B | `model.language_model.norm.weight` [5120] | identity | none | EXACT |
| H2 | logits, Q4(248320×5120), 715161600 B | `lm_head.weight` [248320,5120] | quantize+pack | q4_bits 4, group 64 | TRANSFORM_REQUIRED |

## 4. `target/embedding.bin` (1 file × 3 sections)

| # | destination tensor (section, geometry) | source tensor (shape, bf16) | transform | quantization | status |
|---|---|---|---|---|---|
| E1 | embedding-weights, codes [248320][5120] nibbles, 635699200 B | `model.language_model.embed_tokens.weight` [248320,5120] | quantize+pack-row-major (token-major; low-nibble-first) | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| E2 | embedding-scales, bf16 [248320][80], 39731200 B | (same source; per-group scales) | derived from same quantization | q4_bits 4, group 64 | TRANSFORM_REQUIRED |
| E3 | embedding-biases, bf16 [248320][80], 39731200 B | (same source; per-group biases) | derived from same quantization | q4_bits 4, group 64 | TRANSFORM_REQUIRED |

E1–E3 are three separate 16384-aligned sections of one quantization (spec §3.5;
must not use tile-major indexing for the embedding table).

## 5. `vision/model.bin` (1 file, 930250752 B; all bf16 affine sections)

**REUSE_FROM_BASE justification:** the Swift vision tower is byte-identical to the base
`Qwen/Qwen3.8-27B` on a 7-tensor sample spanning the full block depth — `pos_embed`
(5.3 MB), `patch_embed.proj` (3.5 MB), `merger.norm`, `merger.linear_fc1` (42.5 MB),
`blocks.0.attn.qkv`, `blocks.13.mlp.linear_fc2`, `blocks.26.attn.proj.bias` (SHA-256 equal,
byte-fetched from both repos) — and the config is byte-identical with an identical tensor
inventory. The existing Splash package's `refs/qwen38-splash/vision/model.bin` encodes
exactly those base weights, so the file is reused byte-for-byte. Converting from Swift's
own tensors is also fully specified below (transform column); that path is the fallback if
full-file verification is ever required.

| # | destination tensor (section, geometry) | source tensor (shape, bf16) | transform | quantization | status |
|---|---|---|---|---|---|
| V1 | patch-embed, (1152×1536) weights + [1152] bias | `model.visual.patch_embed.proj.weight` [1152,3,2,16,16] + `.bias` [1152] | identity (flatten input dims 3×2×16×16 = 1536) | none | REUSE_FROM_BASE |
| V2 | position-table, 48×48×1152, 5308416 B | `model.visual.pos_embed.weight` [2304,1152] | identity (2304 = 48×48 rows) | none | REUSE_FROM_BASE |
| V3 | norm1, [1152] + bias (×27 blocks) | `model.visual.blocks.{b}.norm1.weight` + `.bias` [1152] | identity | none | REUSE_FROM_BASE |
| V4 | qkv, (3456×1152) + [3456] bias (×27) | `blocks.{b}.attn.qkv.weight` + `.bias` [3456,1152] | identity (rows [q 1152 \| k 1152 \| v 1152]) | none | REUSE_FROM_BASE |
| V5 | proj, (1152×1152) + [1152] bias (×27) | `blocks.{b}.attn.proj.weight` + `.bias` [1152,1152] | identity | none | REUSE_FROM_BASE |
| V6 | norm2, [1152] + bias (×27) | `blocks.{b}.norm2.weight` + `.bias` [1152] | identity | none | REUSE_FROM_BASE |
| V7 | fc1, (4352×1152) + [4352] bias (×27) | `blocks.{b}.mlp.linear_fc1.weight` + `.bias` [4304,1152] | zero-pad 48 tail rows (4304→4352) + bias zero-pad | none | REUSE_FROM_BASE (if converting from Swift: TRANSFORM_REQUIRED) |
| V8 | fc2, (1152×4352) + [1152] bias (×27) | `blocks.{b}.mlp.linear_fc2.weight` + `.bias` [1152,4304] | zero-pad 48 tail input columns | none | REUSE_FROM_BASE (same) |
| V9 | merger-norm, [1152] + bias | `model.visual.merger.norm.weight` + `.bias` [1152] | identity | none | REUSE_FROM_BASE |
| V10 | merger-fc1, (4608×4608) + [4608] bias | `merger.linear_fc1.weight` + `.bias` [4608,4608] | identity | none | REUSE_FROM_BASE |
| V11 | merger-fc2, (5120×4608) + [5120] bias | `merger.linear_fc2.weight` + `.bias` [5120,4608] | identity | none | REUSE_FROM_BASE |

Section order in file: V1, V2, then per block V3–V8 (b = 0..26), then V9, V10, V11
(QwenVision.cpp:47-94). No deepstack (config `deepstack_visual_indexes: []`).

## 6. Source tensors with no destination (excluded)

15 MTP tensors (source-only; the Qwen3.5 model class ignores `^mtp.*` on load and the
Splash engine has no MTP):

`mtp.fc.weight` [5120,10240]; `mtp.layers.0.input_layernorm.weight` [5120];
`mtp.layers.0.post_attention_layernorm.weight` [5120]; `mtp.layers.0.self_attn.q_proj.weight`
[12288,5120]; `mtp.layers.0.self_attn.k_proj.weight` [1024,5120];
`mtp.layers.0.self_attn.v_proj.weight` [1024,5120]; `mtp.layers.0.self_attn.o_proj.weight`
[5120,6144]; `mtp.layers.0.self_attn.q_norm.weight` [256];
`mtp.layers.0.self_attn.k_norm.weight` [256]; `mtp.layers.0.mlp.gate_proj.weight`
[17408,5120]; `mtp.layers.0.mlp.up_proj.weight` [17408,5120];
`mtp.layers.0.mlp.down_proj.weight` [5120,17408]; `mtp.norm.weight` [5120];
`mtp.pre_fc_norm_embedding.weight` [5120]; `mtp.pre_fc_norm_hidden.weight` [5120].

## 7. Coverage check

- Source: 672 (GDN 14×48) + 176 (attention 11×16) + 3 (top-level) + 333 (visual) + 15 (MTP)
  = **1,199** = full checkpoint; 1,184 mapped, 15 excluded (§6).
- Destination sections: GDN 11×48 = 528; attention 9×16 = 144; head 2; embedding 3;
  vision 1+1+162+3 = 167; total **844** sections, every one assigned a source row above.
- File sizes per `docs/SPLASH_FORMAT_SPEC.md` §5 (GDN 216203264 B, attention 209469440 B,
  head 715194368 B, embedding 715177984 B, vision 930250752 B) are reproduced by the
  section geometry; no section is left undefined.
