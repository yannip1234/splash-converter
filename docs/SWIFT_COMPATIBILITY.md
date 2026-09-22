# Swift-Qwen3.8-27B ↔ Splash Qwen3.8-27B Engine Compatibility

Date: 2026-07-11. Scope: determine whether the high-precision Swift-Qwen3.8-27B checkpoint is
architecturally compatible with Splash's existing Qwen3.8-27B target engine, tensor by tensor.
No converter code is written. Companion file: `docs/TENSOR_MAPPING.md` (one row per Splash
destination tensor).

## 0. Workspace note (finding)

The task referenced `SPEC.md` and `PROGRESS.md` at the workspace root. **Neither file exists**
(verified by exhaustive `find` on 2026-07-11). This investigation proceeds from
`docs/SPLASH_FORMAT_SPEC.md` (424 lines, `FORMAT_GATE=PASS`), engine source under
`refs/splash-src/`, the Splash package under `refs/qwen38-splash/`, and Swift metadata under
`refs/swift-bf16/`.

## 1. Canonical source selection

- `refs/swift-bf16/` is the **only** Swift source in the project and is **BF16**, the
  highest-precision representation available. No oQ8/oQ4e quantized conversion is used as the
  canonical source (task constraint satisfied).
- Provenance: Hugging Face `ukisai/Swift-Qwen3.8-27b`, commit
  `048328f4059015b63f860a453bf94834af0db683` (located via web search; untrusted source, used
  only to identify the remote repo — every fact below is from local files and fetched bytes).
- Local metadata integrity vs remote tree
  (`refs/swift-bf16/.cache/huggingface/trees/048328f…json`): 12 regular files match `blob_id`
  git-blob SHA-1; `tokenizer.json` (LFS) matches `lfs_sha256`
  `0997f410c57a…229b9f3`.
- The 18 weight shards were not downloaded (55.6 GB). All 18 safetensors headers were fetched
  with HTTP Range requests at the pinned commit and parsed: **1,199 tensors, all BF16**; the
  header name set is exactly equal to `model.safetensors.index.json` `weight_map`, and
  per-shard membership matches (0 mismatches). Total declared weight bytes: 55,562,855,904.
- The same header fetch was performed for the base `Qwen/Qwen3.8-27B` (resolve/main) to enable
  byte-level base-vs-Swift comparison (§5).

## 2. Architecture parameter comparison (verified directly)

Swift values: `refs/swift-bf16/config.json` (byte-identical to base — §5.1). Splash values:
`refs/splash-src/runtime/model/Qwen3_8.hpp:19-70` (`Qwen3_8Layout`), `runtime/metal/abi/ExecutionGeometry.h:8`,
and `docs/SPLASH_FORMAT_SPEC.md` §4.

| Parameter | Swift (config.json) | Splash | Match |
|---|---|---|---|
| model type | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` (text: `qwen3_5_text`), `language_model_only: false` | Qwen3.8-27B hybrid GDN/full-attention target | ✓ same design (§3) |
| layers | 64 | 64 | ✓ |
| hidden size | 5120 | 5120 | ✓ |
| intermediate size | 17408 | 17408 | ✓ |
| query heads | 24 | 24 | ✓ |
| KV heads | 4 | 4 | ✓ |
| head dimension | 256 | 256 (attentionWidth 6144) | ✓ |
| vocab | 248320 | 248320 | ✓ |
| embeddings | `embed_tokens` [248320,5120], untied (`tie_word_embeddings: false`) | embedding.bin Q4(248320×5120), separate | ✓ |
| LM head | `lm_head.weight` [248320,5120] separate | head.bin logits Q4(248320×5120) | ✓ |
| normalization | RMSNorm γ only, no bias; eps 1e-06; per-head q/k RMSNorm(256) in attention; gated RMSNorm(128) in GDN | bf16 γ sections: input-norm, post-attention-norm, final-norm, query-norm, key-norm, gdn-norm | ✓ |
| RoPE | θ 1e7; `partial_rotary_factor: 0.25` → 64 rotated dims = 32 pairs; `mrope_interleaved: true`, `mrope_section: [11,11,10]` | rotaryPairs 32, θ 1e7 | ✓ (mrope is a runtime position scheme; for text-only positions t=h=w it degenerates to standard 32-pair RoPE; no weight consequence) |
| recurrent/GDN | 16 key heads ×128; 48 value heads ×128; conv kernel 4; `mamba_ssm_dtype: float32` | GDN 16×128 k, 48×128 v, convDim 10240, 4 taps; gdn-decay stored float32 | ✓ |
| layer pattern | `layer_types`: full_attention at 3,7,…,63 → 16 attention + 48 GDN | full attention iff `(layer+1)%4==0` → same indices | ✓ |
| vision model | depth 27; hidden 1152; intermediate 4304; out_hidden 5120; 16 heads (dim 72); patch 16×16; temporal patch 2; grid 48×48 (`num_position_embeddings: 2304`); 3 channels; `deepstack_visual_indexes: []` | VisionLayout (ops/Vision.hpp:11-26): depth 27, hidden 1152, intermediate 4304 (padded 4352), output 5120, grid 48, patchDim 1536, merged 4608 | ✓ (Splash pads 4304→4352, §3.4) |
| multimodal projector | `model.visual.merger`: LayerNorm(1152) + fc1 [4608,4608] + fc2 [5120,4608], biases | merger-norm(1152), merger-fc1(4608×4608), merger-fc2(5120×4608) | ✓ |
| MTP | `mtp_num_hidden_layers: 1` → 15 `mtp.*` tensors in checkpoint | no MTP in engine; `mtp.*` ignored by model class (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, modeling_qwen3_5.py:925) | source-only, excluded (§4) |
| max position | 262144 | `SPLASH_MAXIMUM_CONTEXT_TOKENS = 262144u` (ExecutionGeometry.h:8) | ✓ |

## 2.1 Tokenizer / special tokens / chat template

| Item | Swift (= base, byte-identical) | Splash package (`refs/qwen38-splash/tokenizer/`) |
|---|---|---|
| vocab.json | 248,320 tokens | **byte-identical** (SHA-256 `ce99b4cb2983…`) |
| tokenizer.json | identical to base | **differs**: pre_tokenizer Split regex `[\p{L}\p{M}]+` (base/Swift) vs `\p{L}+` (Splash — drops the combining-mark class); decoder flags flipped (`add_prefix_space`/`trim_offsets`/`use_regex`); `added_tokens` lists equal |
| tokenizer_config.json | identical to base | MLX-style re-keying; same special-token values; `processor_class: Qwen3VLProcessor`; no `added_tokens_decoder`/`additional_special_tokens` blocks |
| chat_template.jinja | identical to base | 12-byte diff: trailing non-first system message is emitted instead of `raise_exception` |
| config.json | identical to base | base fields + MLX additions (`quantization_config` bits 4/group 64/affine, sampling params, `eos_token_id`) |

Special token IDs (shared vocabulary): image 248056, video 248057, vision start/end 248053/248054,
audio token set. The Splash package ships the identical vocabulary; the MLX repack differences (sec 2.1) do not affect token IDs.

## 3. Layout-level verification (from model code)

Source of truth for row layouts: `transformers==5.17.0` wheel, `models/qwen3_5/modeling_qwen3_5.py`
(config declares `transformers_version: 5.8.0.dev0`; 5.17.0 is the installed superset — shapes
in the checkpoint match the class definitions exactly, so the layout code applies).

### 3.1 GDN layer (`Qwen3_5GatedDeltaNet`, lines 504-615)

- `in_proj_qkv` = Linear(5120 → 2·key_dim + value_dim = 10240), weight [10240, 5120]. Forward
  splits the output as `torch.split([key_dim, key_dim, value_dim])` = **[q 2048 | k 2048 |
  v 6144]** (lines 603-611) — exactly the first three spans of Splash's packed GDN row
  (spec §4.1: q[0,2048), k[2048,4096), v[4096,10240)).
- `in_proj_z` [6144, 5120] → z-gate span [10240,16384).
- `in_proj_b` [48, 5120] → beta span [16384,16432) (`beta = sigmoid(b)`, line 617 — Splash
  applies the sigmoid in-kernel, gdn_primitives.h).
- `in_proj_a` [48, 5120] → a span [16432,16480).
- `A_log` [48] (stored as `log(A)`, line 537; Splash gdn-decay float32 — converter upcasts),
  `dt_bias` [48] (line 533; Splash gdn-time-bias bf16).
- `conv1d` = Conv1d(10240, 10240, kernel 4, groups 10240, no bias) → weight [10240, 1, 4];
  Splash gdn-convolution `w[channel*4+tap]` bf16 — byte-identical layout after dropping the
  singleton dim.
- `norm` = gated RMSNorm(128) → γ [128]; Splash gdn-norm [128].
- `out_proj` = Linear(value_dim 6144 → 5120) → weight [5120, 6144]; Splash gdn-output
  Q4(5120×6144) — same out/in.

### 3.2 Full-attention layer (`Qwen3_5Attention`, lines 749-821)

- `q_proj` = Linear(5120 → 24·256·2 = 12288), weight [12288, 5120]. Forward:
  `self.q_proj(x).view(*, -1, head_dim*2)` then `torch.chunk(…, 2, dim=-1)` (lines 787-789):
  the 12288 rows are **per-head interleaved [q 256 | gate 256] for heads 0..23** — exactly
  Splash's packed attention spans [h·512, h·512+256)=q, [h·512+256, h·512+512)=gate (spec
  §4.2). Gate applied in the epilogue as `attn_output * sigmoid(gate)` (line 818) — same
  semantics as the Splash attention epilogue.
- `k_proj`, `v_proj` [1024, 5120] → spans [12288,13312), [13312,14336).
- `q_norm`/`k_norm` = RMSNorm(256) → γ [256] each; Splash query-norm/key-norm [256] bf16.
- `o_proj` = Linear(6144 → 5120) → [5120, 6144]; Splash attention-output Q4(5120×6144).
- No biases anywhere (`attention_bias`/`mlp_bias` false) — matches Splash's bias-free Q4
  projections.

### 3.3 RoPE (`Qwen3_5TextRotaryEmbedding`, lines 143-218)

θ = 1e7; `dim = int(head_dim * partial_rotary_factor) = int(256·0.25) = 64` → 32 pairs
(line 181-186). mrope sections [11,11,10] interleave t/h/w position ids; for text-only
positions (t=h=w) the recomposition (lines 209-217) leaves the standard 32-pair RoPE.
Splash kernels use 32 pairs, θ 1e7 — match.

### 3.4 Vision (`Qwen3_5Vision*`, lines 948-1210)

- `patch_embed.proj` = Conv3d(3 → 1152, kernel [2,16,16]) + bias → weight [1152,3,2,16,16],
  i.e. 1152×1536; Splash patch-embed (1152×1536) + bias.
- `pos_embed.weight` [2304, 1152] = 48×48×1152; Splash position-table 48×48×1152.
- Per block: `attn.qkv` = Linear(1152→3456) + bias, rows [q 1152 | k 1152 | v 1152] (forward
  reshape `seq,3,heads,-1`, lines 1037-1039) — consumed whole by Splash qkv(3456×1152);
  `attn.proj` [1152,1152] + bias; `norm1`/`norm2` LayerNorm(1152) with bias; `mlp.linear_fc1`
  = Linear(1152→**4304**) + bias, `mlp.linear_fc2` = Linear(4304→1152) + bias. Splash stores
  fc1/fc2 at the **padded** intermediate 4352 (QwenVision.cpp: `readAffine(paddedIntermediateSize, …)`)
  → converter zero-pads 48 rows (fc1) / 48 columns (fc2).
- `merger.norm` LayerNorm(1152) + bias; `merger.linear_fc1` Linear(1152·4→4608) [4608,4608] +
  bias; `merger.linear_fc2` Linear(4608→5120) [5120,4608] + bias. All match Splash merger
  sections. No deepstack (empty index list).

### 3.5 MTP

The checkpoint carries one MTP hidden layer (15 tensors: `mtp.fc` [5120,10240],
`mtp.layers.0.*` full-attention layer, `mtp.norm`, `mtp.pre_fc_norm_embedding`,
`mtp.pre_fc_norm_hidden`). The Qwen3.5 model class ignores `^mtp.*` on load (line 925); the
Splash engine has no MTP destination. **These 15 tensors are source-only and excluded from
the mapping** (listed in TENSOR_MAPPING.md §4).

## 4. Base-vs-Swift tensor-level comparison

Base: `Qwen/Qwen3.8-27B` (resolve/main), same 18-shard layout.

1. **config.json: byte-identical** (4,312 B, same SHA-256 `191e0af23210…`). Swift is a pure
   weight fine-tune of the base; the architecture description did not change.
2. **Tensor inventory: identical** — 1,199 names, 0 shape diffs, 0 dtype diffs (all BF16), 0
   shard-assignment diffs.
3. **Vision tower: byte-identical** — upgraded from a sample to a complete check on
   2026-09-22: **all 333 `model.visual.*` tensors** compared byte-for-byte against
   `Qwen/Qwen3.8-27B` shard 1 (which holds every one of them), 333/333 identical,
   0 differing, 0 absent. Asserted in `tests/test_t5_package.py`. Original 7-tensor
   sample spanning the full block depth:
   `pos_embed` (5.3 MB), `patch_embed.proj` (3.5 MB), `merger.norm`, `merger.linear_fc1`
   (42.5 MB), `blocks.0.attn.qkv`, `blocks.13.mlp.linear_fc2`, `blocks.26.attn.proj.bias` —
   SHA-256 equal base vs Swift. (Sample, not all 222 visual tensors — see §7.)
4. **LM values: partially altered.** Full sweep of every cheap tensor in all 64 layers
   (304 tensors, byte-fetched from both repos):
   - `post_attention_layernorm`: differs in **64/64** layers.
   - `self_attn.q_norm`: differs in **16/16** attention layers.
   - `input_layernorm`: differs in 12/64 layers (1,2,3,4,8,12,20,28,36,44,52,60).
   - `self_attn.k_norm`: differs in 2/16 layers (3, 15).
   - `linear_attn.A_log`: differs in 3/48 layers (1, 2, 4).
   - `linear_attn.dt_bias`: identical 48/48; `linear_attn.norm`: identical 48/48.
   - Spot probes: `layers.63.mlp.down_proj` (178 MB) differs; `model.language_model.norm`,
     `layers.0.input_layernorm`, `layers.0.linear_attn.{A_log,conv1d}`, `mtp.norm` identical.
   The large projection weights (all `in_proj_*`, `out_proj`, `o_proj`, `k/v_proj`,
   `mlp.gate/up/down`, `embed_tokens`, `lm_head`) were not byte-fetched (multi-hundred-MB
   each); per-tensor value identity for them is UNKNOWN, and the converter must read every
   tensor from the Swift checkpoint regardless.
5. **Tokenizer files**: base == Swift byte-identical for all five files + chat template
   (§2.1); the Splash package's tokenizer is a separate MLX-style repack.

## 5. Conclusions — the seven questions

**1. Can the existing Splash Qwen3.8 kernels execute Swift without architectural changes?**
**Yes.** Every Splash destination tensor is produced from Swift tensors by the transforms
listed in TENSOR_MAPPING.md (identity bf16 copies, vertical concatenation of the GDN
projection quartet / attention qkv trio, zero padding, bf16→fp32 upcast of A_log, and
quantize+pack for every Q4 section). The kernel-consumed geometries — Q4(16640×5120),
Q4(14336×5120), Q4(5120×6144) ×2, Q4(17408×5120) ×2, Q4(5120×17408), Q4(248320×5120) ×2,
row-major Q4 embedding, bf16 norm/conv/decay sections, bf16 vision affines — are unchanged.
No kernel, layout constant, or header format needs modification.

**2. Which tensors differ only in values?**
Structurally, **all 1,199** Swift tensors are identical to the base in name, shape, dtype, and
shard placement (value changes do not alter structure). By value, proven-different:
`post_attention_layernorm` (64/64), `q_norm` (16/16), `input_layernorm` (12 layers),
`k_norm` (layers 3, 15), `A_log` (layers 1, 2, 4), `layers.63.mlp.down_proj` (spot).
Proven-identical: `dt_bias` (48/48), `gdn-norm` (48/48), `input_layernorm` (52 layers),
`k_norm` (14 layers), the 7-tensor vision sample, final norm, layer-0 GDN scalars/conv, MTP
norm. Unknown per tensor: the large projection weights (not byte-fetched).

**3. Which differ structurally?**
**None.** Zero structural differences: no added/removed/renamed tensors, no shape or dtype
change, same layer pattern, same head counts, same vision geometry, same vocabulary.
(Packaging-only deltas such as Splash's 160-row GDN tail pad and 4304→4352 vision padding
are destination conventions, not source-structure differences.)

**4. Can the official Qwen vision tower be reused?**
**Yes.** Swift's vision tower is byte-identical to the base on the sampled tensors (incl.
42.5 MB `merger.linear_fc1`) and config-identical in every geometry parameter; the existing
Splash `refs/qwen38-splash/vision/model.bin` encodes exactly those base weights and can be
reused byte-for-byte (REUSE_FROM_BASE in TENSOR_MAPPING.md). Converting from Swift's own
tensors is also fully specified (identity + fc1/fc2 zero padding).

**5. Can the tokenizer be reused byte-for-byte?**
**Swift vs base: yes** — all five tokenizer files + chat template are byte-identical.
**Swift vs the current Splash package: no** — the package ships an MLX-style repack whose
`tokenizer.json` pre_tokenizer regex drops the `\p{M}` combining-mark class, whose decoder
flags differ, and whose chat template changes the trailing-system-message branch.
Recommendation: ship the Swift/base byte-identical files in the Swift package; both satisfy
Splash's `validate_tokenizer` (vocab ⊂ [0,248320)

**6. Can the existing DFlash2 drafter plausibly be tested against Swift without retraining?**
**Plausible, as a smoke/mechanical test — not as a quality claim.** The DFlash2 draft
(`incoai/Qwen3.8-27B-DFlash2`, trained for the base) consumes the target's hidden state
(draft model.bin `context-projection` Q4(5120×25600)) and the shared vocabulary (248320).
Swift's hidden dimension, vocabulary, and special tokens are identical, so the draft runs
unchanged against a Swift target. Its acceptance behavior is trained on the base target's
hidden-state distribution, and Swift's LM weights are altered (§4.4), so acceptance rates
will differ and are **unknown until measured**.

**7. What remains UNKNOWN?**
- Per-tensor value identity for all large projection tensors (not byte-fetched; §4.4).
- Vision-tower identity beyond the 7-tensor sample (222 visual tensors not all hashed).
- DFlash2 acceptance quality against Swift (architecture compatible; quality unmeasured).
- Multimodal RoPE (mrope [11,11,10]) behavior inside Splash's own vision runtime — a runtime
  positioning concern, not a weight/packaging concern.
- The remote repo card's performance claims (untrusted; not relied upon).
- Which tokenizer file set the Splash server actually consumes at runtime for the current
  package (recommendation above avoids the question for the Swift package).

## 6. Gate

No demonstrated architectural reason prevents a one-layer converter over the
TENSOR_MAPPING.md tables.

COMPATIBILITY_GATE=PASS
