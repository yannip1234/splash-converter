# splash-converter

Convert Qwen3.8-27B BF16 or Qwen3.6-35B-A3B BF16/MLX 4-bit checkpoints into
[Splash](https://github.com/incoai/splash) Q4 packages.

Splash documents the package *shape* in `DEVELOPMENT.md` but ships no converter, so the
byte-level layout — section offsets, tile-major packing order, the quantization rule, the
norm conventions — was worked out from the engine source and verified against the one
reference package that exists.

Built to package [Swift-Qwen3.8-27B](https://huggingface.co/ukisai/Swift-Qwen3.8-27b);
the result is at
[SiliconSpecies/Swift-Qwen3.8-27B-Splash](https://huggingface.co/SiliconSpecies/Swift-Qwen3.8-27B-Splash).

## Usage

For Qwen3.6 MoE and RavenX, see [the schema-4 guide](docs/QWEN36_MOE.md).
The new backend is `python -m converter.qwen36_moe`; the commands below remain
the Qwen3.8 dense backend.

```bash
python -m converter.build layers    --source /path/to/bf16 --out output/pkg/target
python -m converter.build head      --source /path/to/bf16 --out output/pkg/target
python -m converter.build embedding --source /path/to/bf16 --out output/pkg/target
python -m converter.package assemble --out output/pkg --swift /path/to/bf16 \
    --reference /path/to/an/existing/splash/package      # draft/ + vision/ are reused
python -m converter.package verify   --root output/pkg
```

Needs `numpy`. A full 64-layer conversion takes ~3 minutes; every file is validated as it
is produced and the build stops at the first failure.

## Scope

- **Works**: any checkpoint with the Qwen3.8-27B layout (64 layers, hidden 5120,
  48 GDN + 16 full-attention), schema 3 / `splash-packed-q4`.
- **Qwen3.6 MoE**: schema 4 / `splash-packed-q4-moe` has a separate backend in
  `converter/qwen36_moe.py`. It accepts BF16 or MLX affine 4-bit input; the
  MLX 8-bit whole-model variant is not accepted.
- `draft/` and `vision/` are byte-copied from an existing package — this converts the
  target model, not the DFlash 2 drafter or the vision tower.

## Qwen3.8 caveats

**Every bf16 norm section stores γ + 1**, including `query-norm` and `key-norm`.
`gdn-norm` is the lone exception and stores γ unchanged. Getting the attention norms wrong
leaves the model fluent while destroying multi-step reasoning (17×23 → 196, √169 → 5), and
**per-section error metrics will not show it** — the section contains exactly what you
intended to write; the intent was wrong. Cost: 27.5 points of task accuracy, invisible to
five separate numerical checks.

**The format has nowhere to carry `generation_config.json`.** The installer accepts exactly
five tokenizer files, and Splash deliberately keeps sampling out of the model. So a packaged
model silently loses its recommended sampling: Splash's chat page defaults to
temp 1.0 / top_p 0.95 / top_k 20, but the `/v1` API defaults to greedy. This affects every
packaged model, not just this one. `converter/sampling_proxy.py` is a ~50-line workaround
for clients that cannot send the values themselves.

## The quantization rule

Per group of 64 input elements of one output row, derived by decoding the reference package
and confirmed bit-exact against the checkpoint it was built from:

```
mn, mx  = group min, max          maxabs = max(|mn|, |mx|)
k       = round(15 * maxabs / (mx - mn))     integer grid position of zero
|s|     = bf16(maxabs / k)
s       = +|s| if mn+mx < 0 else -|s|        sign points away from the anchor
b       = mn   if mn+mx < 0 else mx          anchor = dominant extreme
c       = clip(round((w - b) / s), 0, 15)
```

Sixteen codes span fifteen intervals of `[mn, mx]`; `k` places zero exactly on an integer
code. Dequantization is `w = c·s + b`. Reproduces the reference package's own biases
bit-for-bit and its scales bit-for-bit given `k`; the residual disagreement is tie-breaking
at exact `.5` boundaries and is fidelity-neutral.

## Verification

`tests/` (73 tests) checks against ground truth rather than self-consistency:

- Byte-identity against a verbatim C++ port of the engine's own `packSlab` reference.
- A layer rebuilt from the base checkpoint is byte-compared to the shipped reference file —
  this is what localises a wrong transform to a single section.
- A decode path that shares no index logic with the packer (scalar `struct` offset reads).
- Loader- and installer-equivalent whole-package checks, ported from `WeightStore.cpp` and
  `install/models.py`, and validated against the reference package before being trusted.
- Norm conventions are checked against the reference package, not against our own source.

Tests skip cleanly when `refs/` (a reference package and a checkpoint) are absent.

## Docs

`docs/SPLASH_FORMAT_SPEC.md` — the format as measured: headers, section tables, packing
order, the encoder rule. `docs/TENSOR_MAPPING.md` — source tensor → destination section, per
layer type. `docs/SWIFT_COMPATIBILITY.md` — the source-side compatibility analysis.

Apache 2.0, matching Splash. Unofficial and unaffiliated.
