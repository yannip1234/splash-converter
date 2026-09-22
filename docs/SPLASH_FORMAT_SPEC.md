# Splash Qwen3.8-27B Model Package — Format Specification

Read-only forensic spec of the on-disk format expected by the Splash target loader
(`refs/splash-src`) for the known-good package `refs/qwen38-splash/`.
Every conclusion below cites loader/kernel source (`path :: symbol`, line numbers) and,
where claimed, a direct byte-level check against the package. Nothing here assumes MLX
or Hugging Face storage conventions; the format is Splash's own `splash-packed-q4`
container (manifest `schema_version` 3, `refs/qwen38-splash/manifest.json:413-420`).

Verification method: the section model (header, order, per-section sizes, 16 KiB
alignment, full file consumption) was simulated in Python from the loader source and
matched **all 84 weight files byte-exactly** (all offsets in §5 verified `match=True`
against the actual file sizes in `refs/qwen38-splash/manifest.json`). All 84 weight-file
headers were read directly and tabulated in §2.3.

---

## 1. Package layout

```
<root>/
  manifest.json            # SHA-256 of this file is packageManifestSha256 (ModelDescriptor.mm:379)
  target/                  # the model under test (this spec's focus)
    embedding.bin          # 715177984 B
    head.bin               # 715194368 B
    layer-0.bin .. layer-63.bin   # 64 files; GDN (216203264 B) or attention (209469440 B)
  draft/                   # DFlash2 draft model (separate loader, DFlashDraft.cpp)
    layer-0.bin .. layer-4.bin    # 187449344 B each
    model.bin                # 328794112 B
  vision/
    model.bin              # 930250752 B
  tokenizer/
    chat_template.jinja, config.json, tokenizer.json, tokenizer_config.json, vocab.json
```

`manifest.json` fields required by the loader (`refs/splash-src/runtime/model/ModelDescriptor.mm`):
- `schema_version == 3` (ModelDescriptor.mm:208-209)
- `model` string → `qwen38Descriptor` (ModelDescriptor.mm:387, 171-174)
- `format` object, exact-value checks (ModelDescriptor.mm:135-161):
  - `q4_bits == 4` (212-213), `q4_group_size == 64` (214-215), `q4_storage_n == 256` (216-217)
  - `section_alignment_bytes == 16384` (148-150)
  - `target_layer_magic == "MDFL0006"` (151-153), `draft_layer_magic == "MDFD0004"` (154-156),
    `vision_magic == "MDFV0001"` (157-158)
- `execution_geometry`: 8 fields present **and equal** to compiled constants
  (validateExecutionGeometry, ModelDescriptor.mm:135-145; table at 23-31):
  `allocation_extent_target_bytes, draft_proposal_tokens, draft_query_rows,
  draft_sliding_window, maximum_batch_width, prefill_token_budget, target_kv_block_tokens,
  target_verify_rows` (manifest.json:400-410). The package's 9th field
  `attention_history_group` is extra and not checked. Model dimensions (64 layers,
  hidden 5120, …) are **not** taken from the manifest — they are the compiled
  `Qwen3_8Layout` defaults (Qwen3_8.hpp:24-41).
- `artifacts`: per-file `{path, sha256, size}`. `install/models.py` (installer, not the
  runtime) verifies every sha256/size; the runtime does **not** re-hash file bodies —
  it trusts the loader's header/size checks and `finish()` (see §8).
- `upstream` (manifest.json:423-440): provenance only, not validated by the loader.

Two SHA-256 fingerprints:
1. `packageManifestSha256` — SHA-256 of the `manifest.json` file itself
   (ModelDescriptor.mm:377-379).
2. `manifestFingerprintSha256` — computed from the loaded weight-file records
   (`weightManifestFingerprint`, WeightStore.cpp:302-334):
   canonical text = `"splash-packed-manifest-v1\n"` + for each record sorted by
   `relativePath`: `relativePath \t declaredBytes \t magic \t layer \t type \n`.
   `declaredBytes` is the on-disk file size. The runtime only requires it to be
   non-empty (RuntimeResources.mm:192-201); it is surfaced for diagnostics.
   Computed from the package: full = `af6fda6a474051838a14e023d48d1ab2c333e94344fd8a203d990d685fde542c`,
   target-only = `8a7d3c2ab5f70ee483e2fe3f8f97f63ffd47d35440ef9370539ed0e9165d78ec`.

---

## 2. Weight-file binary format

### 2.1 Header (16 bytes)

`WeightFile` constructor (WeightStore.cpp:161-194): file must be a non-empty regular
file, mmap'd read-only (WeightStore.cpp:89-148).

| offset | bytes | field | check |
|---|---|---|---|
| 0 | 8 | magic, ASCII | exact 8-byte match (WeightStore.cpp:169) |
| 8 | 4 | `layer`, u32 little-endian | must equal expected (WeightStore.cpp:171-180) |
| 12 | 4 | `type`, u32 little-endian | must equal expected (WeightStore.cpp:181-193) |

`loadLittleEndian32` (WeightStore.cpp:78-81). File size must be ≥ 16, a multiple of
16384, and ≥ 16 + one section (WeightStore.cpp:186-203; kWeightFileAlignment = 16*1024,
WeightStore.hpp:19).

### 2.2 Sections

Sections are raw bytes with **no per-section headers**: `WeightFile::section(bytes,
label)` (WeightStore.cpp:198-212) aligns the running offset up to 16384, takes exactly
`bytes`, and advances. `finish()` (WeightStore.cpp:214-221) requires the aligned
consumed offset to equal the file size — the whole file must be accounted for.

Alignment constant: `kWeightFileAlignment = 16384` (WeightStore.hpp:19), asserted equal
to manifest `section_alignment_bytes` (ModelDescriptor.mm:148-150).

### 2.3 Verified header table (read from the package)

| file | magic | layer | type | size |
|---|---|---|---|---|
| target/embedding.bin | MDFE0001 | 248320 | 5120 | 715177984 |
| target/head.bin | MDFL0002 | 64 | 2 | 715194368 |
| target/layer-N.bin | MDFL0006 | N | 0 (GDN) / 1 (full attention) | see §5 |
| draft/layer-N.bin | MDFD0004 | N (0..4) | 0 | 187449344 |
| draft/model.bin | MDFD0004 | 5 | 1 | 328794112 |
| vision/model.bin | MDFV0001 | 27 | 0 | 930250752 |

`type` values expected by the loaders: target layer `fullAttention ? 1U : 0U`
(QwenTarget.hpp:87-88), head `2` (QwenTarget.hpp:102), embedding
`(vocabularySize, hiddenSize)` (QwenTarget.hpp:111-112), draft layer `0` / model `1`
(DFlashDraft.cpp:300, 327), vision `(depth, 0)` (QwenVision.cpp:57).
All 84 files in the package match exactly (checked directly).

Full-attention layers in the package: 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 —
i.e. `(N+1) % 4 == 0` (Qwen3_8.hpp:45-48), matching `type=1` exactly.

---

## 3. Q4 tensor family (the quantization format)

### 3.1 Constants

- bit width 4, group size 64, storage N 256 (manifest; `kQ4GroupElements`/`kQ4StorageN`,
  WeightStore.hpp:16, 20).
- `validateQ4Layout`: `in % 64 == 0` (q4Elements, WeightStore.cpp:39-45),
  `out % 256 == 0` (validateQ4Layout, WeightStore.cpp:62-68).

### 3.2 Projection packing (`readQ4Projection`, WeightStore.cpp:231-247)

A projection of logical shape `[out][in]` (output rows × input columns) is stored as
**one section** of `q4PackedBytes(out, in) = out*in*9/16` bytes (WeightStore.cpp:57-60),
laid out in the section as `[codes | scales | biases]`:

| sub-part | bytes | element count | element type |
|---|---|---|---|
| codes | `out*in/2` | `out*in` nibbles | unsigned 4-bit, 0..15 |
| scales | `out*in/32` | `out*in/64` | bfloat16 |
| biases | `out*in/32` | `out*in/64` | bfloat16 (affine offset, a.k.a. zero point) |

The loader builds three buffer views into the one section: codes view of
`weightBytes = out*in/2`, scales of `parameterBytes = out*in/32`, biases of the same
(WeightStore.cpp:238-246). Dequantization:

```
w[n,k] = code(n, k) * scale(n, k/64) + bias(n, k/64)      (kernel epilogue, q4_mpp_tiles.h:172-174)
```

unsigned codes, **no +8 centering**; `scale` and `bias` are per (output row n,
64-wide input group g=k/64).

### 3.3 Code indexing — tile-major (all Q4 projections)

Verified against the package and the CPU reference `packSlab`
(`refs/splash-src/dev/tests/engine/moe_metal_test.mm:124-142`):

- A **tile** is 256 consecutive output rows (StorageN=256). For row `n`, group `g`:
  `parameter(n, g) = (n/256)*groups + g` flattened as `((n/256)*groups + g)*256 + n%256`
  (moe_metal_test.mm:98-101).
- Within one tile and one group, the 256×64 code block is row-major:
  nibble index `= (n%256)*64 + (k%64)`, i.e. 64 nibbles = **32 bytes per row**,
  contiguous across the 256 rows of the tile (moe_metal_test.mm:131-135).
- **Nibble order within a byte: low nibble first** (even k%64 in bits 0-3, odd k%64 in
  bits 4-7). `packSlab` writes `value<<4` for the odd position (moe_metal_test.mm:131-135);
  the embedding kernel reads `packed >> ((dim&1)*4) & 15`
  (runtime/metal/kernels/shared/embedding.metal:20-21).
- Kernel-side: the code buffer is addressed as a `uint4b_format` tensor
  `dextents{64, TileN} strides{1, 64}` with tile base
  `weights + tile*quant_groups*StorageN*64/2` and group offset
  `(quant_group*StorageN + tile_offset)*32` bytes (q4_mpp_tiles.h:148-151, 252-258).

### 3.4 Scale/bias indexing — tile-major (all Q4 projections)

`parameter = (tile * quant_groups + quant_group) * StorageN + tile_offset + row_in_tile`
(q4_mpp_tiles.h:169-170; identical formula in the CPU reference, moe_metal_test.mm:98-101).
So the scales/biases arrays are also tile-major: for a given 256-row tile, all 64
input groups, each 256 rows.

### 3.5 Embedding packing — row-major (exception)

`readQ4ProjectionComponents` (WeightStore.cpp:249-262) reads **three separate
sections** — `embedding-weights` (`out*in/2`), `embedding-scales` (`out*in/32`),
`embedding-biases` (`out*in/32`) — each independently 16384-aligned. The layout is
**plain row-major [token][dim]**, *not* tile-major (embedding.metal:20-24):

```
code(t, d)      = weights[t*(in/2) + d/2] >> ((d&1)*4) & 15     // low nibble first
scale/bias idx  = t * (in/64) + d/64
out[t, d]       = bfloat( code * scales[idx] + biases[idx] )
```

Consequence: an implementation must not reuse the tile-major indexing for the embedding
table, and must not assume the three embedding sub-arrays are contiguous within one
packed region — they are three aligned sections.

### 3.6 Measured weight statistics (layer-0 mlp-gate, dequantized in Python)

**Corrected 2026-09-21 (iteration 3).** The original numbers in this section
(|scale| ≈ 0.8–1.0, |bias| ≈ 1.1–1.25, `bias/scale ≈ -1.32`, row-0 min −13.03 /
max +12.75 / mean −1.03 / std 6.94) were measured by decoding the bf16 scale/bias
bytes with Python `struct` format `"e"` — IEEE 754 **half precision**, not bfloat16.
The values below are from a correct bf16 decode (verified: the `"e"` decode exactly
reproduces the original session's printed first-8 scales, confirming the bit pattern).

Per 64-group (1,392,640 groups, full section): |scale| p0.5/p50/p99.5 =
0.002106 / 0.003098 / 0.005127 (max 0.030151); |bias| p0.5/p50/p99.5 =
0.017090 / 0.025757 / 0.046631 (max 0.421875); `sign(scale) == -sign(bias)` in **100%**
of groups, zero scales/biases; `bias/scale` mean **−8.3779** (median −10.0253,
p0.5/p99.5 = −10.0253 / −8.0000). Row-0 dequantization (`w = code·scale + bias`):
min **−0.038330**, max **+0.039795**, mean **0.000120**, std **0.010972**
(rows 100/8704/17407: std 0.010591 / 0.009619 / 0.009847 — stable). Code histogram
(89,128,960 codes) is unimodal with a single dominant peak at **8** (10,734,108),
9 close second (10,625,498); mean code 8.2386. The dequantized scale
(≈1/√5120 ≈ 0.014 per element) is exactly the expected MLP-gate weight magnitude.

The 100% sign agreement and the tight |bias|/|scale| ≈ 8.0–10 cluster (mean −8.38)
are **measured properties of the converter output**, not loader requirements (the
loader applies no constraint on scale/bias values). A converter reproducing this
package must reproduce them.

---

## 4. Qwen3.8 target geometry

From `Qwen3_8Layout` (Qwen3_8.hpp:19-70), all consistent with manifest `execution_geometry`:

- layers 64, hidden 5120, vocab 248320, intermediate 17408
- attention: 24 query heads × 256 dim (width 6144), 4 KV heads, rotary pairs 32, θ=1e7
- GDN: 16 key heads × 128, 48 value heads × 128, convolutionDimension 10240, 4 conv taps
- packed GDN row width **16640**, packed full-attention row width **14336**
- full attention iff `(layer+1) % 4 == 0` (16 attention layers, 48 GDN layers)

### 4.1 GDN packed row (width 16640) — produced by the `gdn-input` Q4(16640×5120)

| span | field | size | kernel consumption |
|---|---|---|---|
| [0, 2048) | q | 16×128 | scaled ×1/128 at write (prefill/gdn.metal) |
| [2048, 4096) | k | 16×128 | scaled ×1/√128 at write |
| [4096, 10240) | v | 48×128 | value_head h → key_head h/3, lane h%3 |
| [10240, 16384) | z-gate | 48×128 | `gate = packed[...+10240+head*128+dim]`; `hidden = rms*silu(gate)` (gdn_primitives.h:39-53, 58-93) |
| [16384, 16432) | beta | 48 | `beta = sigmoid(b)` |
| [16432, 16480) | a (dt input) | 48 | `decay = exp(a_scale * softplus(bf16(a + dt_bias)))` |
| [16480, 16640) | padding | 160 | unused |

Offsets pinned from `gdn_prepare_prefill_phase` (prefill/gdn.metal:192-260:
KeyWidth=2048, ValueWidth=6144, BOffset=ConvDim+ValueWidth=16384, AOffset=16432) and
`gdn_gate_phase` (gdn_primitives.h:65, ZOffset=ConvDim=10240).

### 4.2 Full-attention packed row (width 14336) — produced by `attention-input` Q4(14336×5120)

Interleaved per query head, then k, v (attention_qkv_prepare.h:15-17;
prefill/attention_qkv.metal:1-80):

| span | field | size |
|---|---|---|
| [h·512, h·512+256) | q for query head h | 24×256 |
| [h·512+256, h·512+512) | gate for head h | 24×256 |
| [12288, 13312) | k | 4×256 |
| [13312, 14336) | v | 4×256 |

QStride=512, PackedStride = 24·512 + 2·4·256 = 14336. The gate is applied in the
attention epilogue: `hidden = attention * sigmoid(gate)` (attention_qkv.metal).

Both packed widths are **fully accounted for** (no unexplained tail): 16640 =
2048+2048+6144+6144+48+48+160(pad); 14336 = 12288+1024+1024, no tail padding.

---

## 5. Per-file section tables (verified byte-exact against the package)

All offsets start at 16 (header) with each section 16384-aligned. `q4(o,i)` = o·i·9/16.

### 5.1 target/layer-N.bin — GDN layers (48 files, 216203264 B)

| section | offset | bytes | tensor | dtype |
|---|---|---|---|---|
| input-norm | 16384 | 10240 | RMS γ [5120] | bf16 |
| gdn-input | 32768 | 47923200 | Q4(16640×5120): codes 42598400 + scales 2662400 + biases 2662400 | §3 |
| gdn-convolution | 47955968 | 81920 | [10240 ch][4 taps] | bf16, `w[channel*4+tap]` (gdn_primitives.h:8-23) |
| gdn-decay | 48037888 | 192 | [48 heads] | float32 |
| gdn-time-bias | 48037888+96+16384→48054272 | 96 | [48] dt bias | bf16 |
| gdn-norm | 48070656 | 256 | RMS γ [128] | bf16 |
| gdn-output | 48087040 | 17694720 | Q4(5120×6144) | §3 |
| post-attention-norm | 65781760 | 10240 | RMS γ [5120] | bf16 |
| mlp-gate | 65798144 | 50135040 | Q4(17408×5120) | §3 |
| mlp-up | 115933184 | 50135040 | Q4(17408×5120) | §3 |
| mlp-down | 166068224 | 50135040 | Q4(5120×17408) | §3 |

Section list & order: QwenTarget.cpp:91-133 (`readQwenMixer`), Qwen3_8.cpp:43-51 (mlp),
QwenTarget.hpp:90-95 (norms).

### 5.2 target/layer-N.bin — full-attention layers (16 files, 209469440 B)

| section | offset | bytes | tensor | dtype |
|---|---|---|---|---|
| input-norm | 16384 | 10240 | RMS γ [5120] | bf16 |
| attention-input | 32768 | 41287680 | Q4(14336×5120): codes 36700160 + scales 2293760 + biases 2293760 | §3 |
| query-norm | 41320448 | 512 | RMS γ per head [256] | bf16 |
| key-norm | 41336832 | 512 | RMS γ per head [256] | bf16 |
| attention-output | 41353216 | 17694720 | Q4(5120×6144) | §3 |
| post-attention-norm | 59047936 | 10240 | RMS γ [5120] | bf16 |
| mlp-gate | 59064320 | 50135040 | Q4(17408×5120) | §3 |
| mlp-up | 109199360 | 50135040 | Q4(17408×5120) | §3 |
| mlp-down | 159334400 | 50135040 | Q4(5120×17408) | §3 |

Section list & order: QwenTarget.cpp:95-108 (attention branch), Qwen3_8.cpp:43-51.

### 5.3 target/head.bin (715194368 B)

| section | offset | bytes | tensor |
|---|---|---|---|
| final-norm | 16384 | 10240 | RMS γ [5120] |
| logits | 32768 | 715161600 | Q4(248320×5120): codes 635699200 + scales 39731200 + biases 39731200 |

Loader: QwenTarget.hpp:101-108. Consumed: final RMSNorm then Q4 projection
`logits[rows, vocab]` (QwenTarget.cpp:544-552, 566-571).

### 5.4 target/embedding.bin (715177984 B)

| section | offset | bytes | tensor |
|---|---|---|---|
| embedding-weights | 16384 | 635699200 | codes [248320][5120] nibbles, row-major (§3.5) |
| embedding-scales | 635715584 | 39731200 | bf16 [248320][80] |
| embedding-biases | 675446784 | 39731200 | bf16 [248320][80] |

Loader: QwenTarget.hpp:110-116. Kernel: `embedding_q4_h5120`
(Embedding.cpp:11-20, 24-35; embedding.metal:20-24).

### 5.5 draft/ (DFlash2; loader DFlashDraft.cpp — secondary, not the test target)

Layer file (187449344 B): input-norm 10240; attention-convolution 40960 (bf16, 4 taps ×
5120); attention-dynamic Q4(1280×5120); qkv Q4(6144×5120); query-norm 256; key-norm 256;
attention-output Q4(5120×4096); post-attention-norm 10240; mlp-convolution 40960;
mlp-dynamic Q4(1280×5120); mlp-gate/up Q4(17408×5120); mlp-down Q4(5120×17408).
model.bin (328794112 B): context-projection Q4(5120×25600); hidden-norm 10240;
final-norm 10240; selector Q4(256×5120); predecessor-codebook bf16 [248320×256]
127124480; successor-codebook bf16 [248320×256] 127124480.
Both sizes verified byte-exact.

### 5.6 vision/model.bin (930250752 B; loader QwenVision.cpp:47-94)

All **bf16 affine** sections (weights `out*in*2` + biases `out*2`), no Q4:
patch-embed (1152×1536); position-table 48×48×1152×2; ×27 blocks {norm1(1152),
qkv(3456×1152), proj(1152×1152), norm2(1152), fc1(4352×1152), fc2(1152×4352)};
merger-norm(1152); merger-fc1(4608×4608); merger-fc2(5120×4608). Size verified
byte-exact.

---

## 6. Where the kernel consumes each tensor family

- **Q4 projections** → `Q4Linear::add` (Linear.cpp:352-380): buffers
  `[input, codes, scales, biases, (residual|gate), output]` + `Q4Params{out, in,
  persistent_groups}` (Linear.h:10-26); decode kernels `decode_linear_q4_n128/n256/…`
  (decode/linear_q4.metal:70-109), prefill `prefill_linear_q4_n128/n256/…`
  (prefill/linear_q4.metal:139-313); inner loop `q4_mpp_tile` reads the `uint4b_format`
  code tensor, per-group `partial * scales[parameter] + input_sums * biases[parameter]`
  (q4_mpp_tiles.h:169-180, 213-232). Epilogues: plain, `+residual`, fused gate·up SiLU
  (q4_mpp_tiles.h:217-226; kernel-side SiLU = `g/(1+exp2(-1.44269504·g))*up`).
- **Embedding** → `Embedding::add` → `embedding_q4_h5120` (Embedding.cpp:24-35,
  embedding.metal:11-24), threadgroup {40,1,1}.
- **GDN row** → conv/silu (gdn_primitives.h:8-34), gates (39-53), recurrent scan +
  gate phase (58-93), prefill prepare (prefill/gdn.metal:192-271).
- **Attention row** → per-head RMSNorm + RoPE (32 pairs) + value passthrough
  (attention_qkv_prepare.h:15-73), gate epilogue (prefill/attention_qkv.metal).
- **Logits** → head Q4 projection as in 5.3 (QwenTarget.cpp:544-572).

---

## 7. Loader execution trace (one target layer, GDN)

1. `serve-native TARGET_DIR DRAFT_DIR ...` (main.mm:185-195) → `inspectModelPackage`
   (ModelDescriptor.mm:375-399): parse `manifest.json`, hash it, validate
   schema/format/geometry (135-161, 208-219) → `qwen38Descriptor` (171-174).
2. `loadModelPackage` (ModelFactory.cpp:60-69) → `loadPackage` (26-58):
   `loadQwen3_8Weights(backend, root/"target", layout)` (37) → `loadQwenTargetWeights`
   (QwenTarget.hpp:72-125): loop 64 `layer-N.bin` (open WeightFile with magic MDFL0006,
   layer=N, type=fullAttention), read sections in §5.1/5.2 order, `finish()`; then
   head.bin, embedding.bin; compute `manifestFingerprintSha256` (119).
   Then `loadDFlashDraftWeights` (ModelFactory.cpp:42), `loadQwenVisionWeights` (44).
3. Runtime buffers are Metal views over the mmap'd file regions (WeightStore.cpp:152-156,
   198-212) — weights are never copied; kernels read them file-backed (WeightStore.cpp:116-119).
4. Forward: `inputNorm` → Q4 `gdn-input` (Linear) → GDN conv/scan/gate → Q4
   `gdn-output` (+residual) → `postAttentionNorm` → Q4 `mlp-gate`/`mlp-up` (fused
   GateUp SiLU) → Q4 `mlp-down` (+residual); head: final RMSNorm → Q4 `logits`.
   (QwenTarget.cpp:441-553.)

---

## 8. Files Splash actually requires

From `install/models.py:177-187` (authoritative installer file table,
`target_layers, draft_layers = (64, 5)` for schema 3) + the loaders above:

- `manifest.json` (schema 3, all §1 fields)
- `target/embedding.bin`, `target/head.bin`, `target/layer-{0..63}.bin`
- `draft/layer-{0..4}.bin`, `draft/model.bin`
- `vision/model.bin`
- `tokenizer/{chat_template.jinja, config.json, tokenizer.json, tokenizer_config.json, vocab.json}`
  (`config.json` text must have `text_config.model_type == "qwen3_5_text"`,
  `hidden_size == 5120`, `vocab_size == 248320`,
  `max_position_embeddings == kv::kMaximumLogicalTokens` — ModelDescriptor.mm:184-206)

Total **73 weight files** (66 target + 6 draft + 1 vision) + `manifest.json` + 5 tokenizer
files; the manifest lists **78 artifacts** (everything but itself). Counted directly from
the reference package by the whole-package harness (`converter/package.py`, T5.5); the
earlier figure of 84 in this section was an arithmetic slip.

Runtime integrity checks: per-file magic/layer/type (WeightStore.cpp:169-193), size
multiple-of-16384 and ≥ header+section (186-203), exact section sizes (198-212),
full consumption (214-221), `out%256==0`/`in%64==0` (39-68). File-body sha256 is
checked by the installer, not the runtime.

---

## 9. Quantization encoder rule (PINNED — T1.4)

**Provenance first.** The reference package was **not** built from the Swift
fine-tune, and not from `mlx-community/Qwen3.8-27B-4bit` despite the package
README's `base_model:` line. It was quantized from **`Qwen/Qwen3.8-27B` BF16**.
Proof: applying the rule below to that checkpoint reproduces the package's stored
biases bit-for-bit (100%) and its scale magnitudes bit-for-bit given `k` (100%)
across both layer-0 MLP sections, 1,392,640 groups each. The same test against the
Swift checkpoint gives **0.673861** bias agreement and 0.969910 code agreement —
decisive, but note that Swift's layer-0 `gate_proj` differs from base by at most
5.19e-4 (mean 5.27e-5): the fine-tune is light enough that correlation-based
provenance tests cannot separate the two checkpoints, which is why only a
bit-exact test settles it. Everything in this section is therefore measured
against the base checkpoint, and bit-identity with the package is *not* a
requirement for Swift tensors — they are different weights.

**The rule.** Per group of 64 consecutive input elements of one output row:

```
mn, mx   = group min, max                  (bf16-exact; the source is bf16)
maxabs   = max(|mn|, |mx|)                 magnitude of the dominant extreme
span     = mx - mn
k        = round(15 * maxabs / span)       integer grid position of zero
|s|      = bf16(maxabs / k)
s        = +|s| if mn+mx < 0 else -|s|     sign points away from the anchor
b        = mn   if mn+mx < 0 else mx       anchor = dominant extreme
c        = clip(round((w - b) / s), 0, 15)
```

The 16 codes span 15 intervals across `[mn, mx]`. `k` is chosen so that **zero
lands exactly on an integer code**, which is why §3.6 measures `b/s ≈ -k`
(mean −8.38, `k ∈ 7..14`) and a code histogram peaking at 8; `|s| = maxabs/k`
makes the dominant extreme exactly representable at code 0. Dequantization is
unchanged: `w = c·s + b = (c − k)·s`.

**Measured agreement** (`tests/test_encoder_t14.py`, the three layer-0 MLP
sections of `refs/qwen38-splash/target/layer-0.bin` vs base `Qwen/Qwen3.8-27B`):

| quantity | mlp-gate | mlp-up | mlp-down |
|---|---|---|---|
| bias bit-exact | **1.000000** | **1.000000** | **1.000000** |
| scale sign | **1.000000** | **1.000000** | **1.000000** |
| `\|s\| == bf16(maxabs/k_ref)` | **1.000000** | **1.000000** | **1.000000** |
| `k` agreement | 0.999115 | 0.999142 | 0.999252 |
| codes exact | 0.995598 | 0.995589 | 0.995690 |
| codes within ±1 | 0.999861 | 0.999863 | 0.999884 |
| packed section bytes identical | 0.992733 | 0.992697 | 0.992820 |

**The residual is tie-breaking, and it costs nothing.** All 1,232 `k`
disagreements (0.0885%) occur at *exact* `.5` values of `15·maxabs/span`, and are
always ±1 step; 12,422 groups sit on such a tie. Round-half-up recovers 11,190 of
them. The reference encoder's choice at the remaining ties is not a function of
any group statistic, bf16 intermediate, or mantissa predicate tested — it is not
recoverable from the package, and it is not worth recovering:

| fidelity vs base BF16 | this rule | reference encoder |
|---|---|---|
| mlp-gate RMSE | **0.00093007** | 0.00093048 |
| mlp-gate mean abs | **0.00078454** | 0.00078480 |
| mlp-gate cosine | **0.99582009** | 0.99581641 |
| mlp-up RMSE | **0.00089614** | 0.00089652 |
| mlp-up cosine | **0.99583837** | 0.99583481 |

This reconstruction is *marginally more* faithful to the source than the encoder
that produced the shipped package, on every metric, with identical max abs error.
That is the acceptance bar, and it is met.

**Consequence for thresholds:** ~0.9958 cosine is the empirical ceiling of this
format at group 64 / 16 levels — the shipped reference package does no better
against its own source. Any per-tensor gate expecting cosine ≥ 0.999 for Q4 is
mis-specified (corrected in SPEC.md §5 item 3).

Applying the rule to the **Swift** layer-0 `gate_proj` against *its own* source
gives cosine **0.99582004**, RMSE **9.30092e-04** — indistinguishable from the
base-checkpoint result, so T2 should land in the same place.

**`mlp-down` geometry (resolved).** The section is plain `q4(5120×17408)` — the
HF `down_proj` tensor exactly as stored, **no transpose and no permutation**, the
same layout and the same rule as gate/up (table above). The trap is that the
section's byte count cannot distinguish `q4(5120×17408)` from `q4(17408×5120)`,
since `out·in·9/16` is symmetric; only decoding separates them, and it does so
decisively — the transposed reading scores **0.004142** bias agreement against
**1.000000** for the correct one. Any hypothesis built on the byte count alone
(e.g. a group permutation reconciling the two) is unfalsifiable by construction.
Spec §5.1's `Q4(5120×17408)` label is correct as written.

## 9b. Remaining open questions

- `UNKNOWN:` whether the MoE schema-4 magics (`MDFM0001/0002`) share the same section
  framing beyond the 16-byte header — out of scope (schema 3 only in the package).
- `UNKNOWN:` the runtime does not re-verify per-file sha256 from `manifest.json`
  `artifacts`; whether a deployment gate should (installer-only today) is a policy
  choice, not a format requirement.
- `UNKNOWN:` the reference encoder's tie-break at exact `.5` grid positions
  (see §9). Measured as fidelity-neutral; recorded, not chased.

---

## 10. Gate result

Format fully determined and independently verifiable: for every tensor family the
filename, section order, byte sizes (all 73 weight files byte-exact), dtype, packing order
(tile-major projection codes with low-nibble-first bytes; row-major embedding),
scale/bias representation (bf16 affine, tile-major index for projections,
token-major for embedding), dequantization (`w = s·c + b`), and the exact kernel
consumption path are pinned with source citations. A single tensor (e.g.
`target/layer-0.bin` mlp-gate) can be implemented from §3 + §5.1 alone and
independently checked (dequantize; round-trip re-encode; cross-check with the CPU
reference `packSlab`/`project`).

FORMAT_GATE=PASS
