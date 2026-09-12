# DeepSeek V4.1-Flash port — architecture delta vs V4 (colibri `deepseek_v4` engine)

Source: `deepseek-ai/DeepSeek-V4.1-Flash` config.json (`model_type: deepseek_v41`,
arch `DeepseekV41ForCausalLM`) vs `deepseek-ai/DeepSeek-V4-Flash-0731`
(`model_type: deepseek_v4`, 43 layers / hidden 4096 / 256 experts + 1 shared / top-6).
Reference implementation cross-checked against the official inference sources
(`modeling.py`, `engram.py`, `vision.py`, quoted below as `ref:LINE`).

## Shape changes (easy)

| key | V4 | V4.1 |
| --- | --- | --- |
| `num_hidden_layers` | 43 | 40 |
| `hidden_size` | 4096 | 5120 |
| `n_routed_experts` | 256 | 384 |
| `num_experts_per_tok` | 6 | 6 |
| `moe_intermediate_size` | ? | 2304 |
| `num_attention_heads` | ? | 64 |
| `num_key_value_heads` | ? | 1 |
| `head_dim` | ? | 512 |
| `qk_rope_head_dim` | ? | 64 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | present | 1280 / 1024 / 8 |
| `sliding_window` | present | 128 |
| `vocab_size` | ? | 129280 |
| `max_position_embeddings` | ? | 1048576, YaRN factor 16 |
| `compress_ratios` | 43 entries | **43 entries for 40 layers** (2 for layers 2..19, 1 for 20..39) |
| expert weight dtype | native fp4 | **fp4** (`expert_dtype`), dense fp8-e4m3 UE8M0 — same layout family |
| config shape | flat, `expert_dtype`/`scoring_func` at the root | **nested**: architecture keys live in `text_config`, plus a `vision_config`; `quantization_config` has no `fmt` |
| `attn.attn_sink` | present | present (per-head f32 sinks, `ref:639`) |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | present | 4 / 20 / 1e-06 |

## Already covered by the V4 engine (verified, not assumed)

Two items from the first draft of this document were wrong; reading `deepseek_v4.c`
disproves them, and they are the reason this port is a fork rather than a rewrite:

1. **Router — done.** `coli_v4_route()` (`deepseek_v4.c:1654`) already implements
   exactly V4.1's router: `scores = sqrt(softplus(gate·h))`, selection by
   `scores + e_score_correction_bias` (`topk_method: noaux_tc`), then
   `norm_topk_prob` over the selected scores and `routed_scaling_factor`.
   `coli_v4_config_parse` even *requires* `scoring_func: sqrtsoftplus` and
   `topk_method: noaux_tc`. Nothing to port.
2. **SwiGLU clamp — done.** `coli_v4_swiglu()` (`deepseek_v4.c:1713`) clamps
   `up` to ±limit and `gate` to +limit, matching `ref:845-847`, and
   `swiglu_limit: 10.0` is a required key. Nothing to port.

## New mechanisms (the real work)

1. **Shared KV and index layers.** `kv_source_layer_ids: [2, 8, 14, 20]`,
   `index_source_layer_ids: [2, 8, 14, 20, 24, 28, 32, 36]` (8 entries).
   `compress_ratio > 0` does **not** mean a layer compresses its own KV: only the
   source layers own a compressor and an indexer (`ref:618`, `ref:654-660`), and
   every other layer reuses that layer's compressed KV and its DSA index. The V4
   engine plans one KV per layer (`context_bytes`, cache slots, `coli_v4_layer_plan`
   keying indexer/compressor weights on `compress_ratios[layer] > 0`), so both the
   planner and the per-layer runtime need a source-layer indirection.
   Supporting keys: `candidate_source_layer_id: 20`, `candidate_topk_blocks: 2048`,
   `candidate_block_size: 8`.
2. **Engram (NEW).** `engram_layer_ids: [1, 14]`, `engram_num_embeddings:
   [384006168, 384016682]`, `engram_max_ngram_size: 4`, `engram_n_heads: 8`,
   `engram_head_dim: 256`, `engram_vocab_size: 16000000`,
   `engram_compressed_vocab_size: 99092`. Each engram layer adds an n-gram hash
   lookup into the residual stream: per-layer **prime-sized bucket ranges** drawn
   in order (never reused), hashes built by XOR-ing `token_id * multiplier` over
   the `max_ngram_size - 1` previous tokens, ids going through a compressed token
   map (NFKC/NFD/strip-accents/lowercase + whitespace collapse) whose size the
   multipliers derive from (`ref`/engram.py). The token map is *not* a checkpoint
   tensor: it is rebuilt from the tokenizer, so the port needs it generated (the
   engine owns `tok_unicode*` tables, the generator is a build step).
3. **DSpark / MTP (NEW layout).** `num_nextn_predict_layers: 3`,
   `dspark_target_layer_ids: [37, 38, 39]`, `dspark_block_size: 5`,
   `dspark_markov_rank: 256`, `dspark_n_routed_experts: 128`,
   `dspark_num_experts_per_tok: 3`. The V4 engine's DSpark include
   (`deepseek_v4_dspark.inc`) probes `mtp.N.markov_head.markov_w1/w2.weight`; V4.1
   ships `mtp.N.markov_head.embed.weight` + `mtp.N.markov_head.head.weight`, plus
   `mtp.N.main_proj`, `mtp.N.main_norm`, `mtp.N.confidence_head.proj`. Port, don't
   assume reuse.
4. **Vision tower + aligner.** `deepseek_v41_vision`: 32 layers, hidden 1024, 16
   heads, patch 14, `downsample_ratio: 3`, `max_image_tokens: 1024`. Weights:
   `vision.*`, `aligner.w1/w2`, `image_start`/`image_end`/`image_newline`.
   Text-only inference does not need it, so the engine may refuse image input
   instead of running it wrongly (`ffn.gate.bias_vl` is only applied when an image
   mask is present — `ref:819-820`).
5. **Asymmetric active params** (8B input / 16B output): prefill vs decode route
   different widths; the planner's prefill/decode split must model it.

## Tensor inventory (official safetensors index, 96,085 tensors)

Patterns relative to `layers.N.`; counts are per-checkpoint totals.

| tensor | count | note |
| --- | --- | --- |
| `ffn.experts.{w1,w2,w3}.{weight,scale}` | 15360 each | 384 experts × 40 layers, fp4 + per-32 scales |
| `ffn.shared_experts.{w1,w2,w3}.{weight,scale}` | 40 each | one shared expert per layer |
| `ffn.gate.weight` / `ffn.gate.bias` / **`ffn.gate.bias_vl`** | 40 each | `bias_vl` is the vision-masked routing bias (new) |
| `attn.{wkv,wq_a,wq_b,wo_a,wo_b}.{weight,scale}` | 40 each | fp8-e4m3 + UE8M0 |
| `attn.attn_sink`, `attn.q_norm.weight`, `attn.kv_norm.weight` | 40 each | per-layer sinks |
| **`attn.indexer.wk.weight`** | 4 | new: indexer key projection, on source layers only |
| `attn.indexer.{wq_b,weights_proj}` / `attn.indexer.k_norm` | 8 / 4 | present only where an index is computed |
| `attn.compressor.{norm,wkv}` | 4 | **only the 4 `kv_source_layer_ids` own a compressor** |
| `attn.compressor.wgate.weight` | 3 | 3 of those 4 compress (ratios 2/2/2/1) |
| **`engram.{embed.weight,embed.scale,k_weight,q_weight,wkv.weight,wkv.scale}`** | 2 each | new tensor family, `engram_layer_ids` |
| `mtp.N.*` (attn, ffn, gate, hc_*) | 3 each | DSpark layers, 128 experts × top-3 |
| **`mtp.N.{main_proj,main_norm,confidence_head.proj,markov_head.embed,markov_head.head,norm}`** | 1 each | new DSpark head layout |
| `hc_{attn,ffn}_{base,fn,scale}` | 40 each | hyper-connections |
| `vision.*`, `aligner.*`, `image_*` | 32 blocks | vision tower (text-only runs ignore it) |
| `embed.weight`, `head.weight`, `norm.weight` | 1 each | untied embeddings |

## Fail-closed gates (the port must never compute a different model)

Anchored on the checkpoint's own contents, so each gate flips off as its mechanism
lands — never on an assumption:

| gate | trigger | behaviour today |
| --- | --- | --- |
| engram tensors present + `engram_layer_ids` set | `layers.N.engram.*` in the index | **refuse**: dropping the n-gram contribution changes the residual stream |
| image input | prompt contains an image token | **refuse**: no vision tower, and `gate.bias_vl` would be skipped |
| DSpark | `mtp.*` present | **speculation off, warn**: the greedy path is unaffected, so this is a speed loss, not a wrong answer |
| CUDA tier | always, for now | V4.1 on the V4 tier (`coli_cuda_dsv4*.dll`) would upload V4.1 weights into V4 maths; `COLI_V4_GPU_TIER` is deliberately left undefined |
| prefix checkpoint | snapshot magic | V4.1 writes `COLIV41C`, V4 writes `COLIV4CK`: a V4 snapshot can never resume a V4.1 session |

## Verified checkpoint facts (headers read, no weights)

`tools/check_deepseek_v41_checkpoint.py` validates a checkpoint against the
engine's contract from the safetensors headers alone: `--from-hf` pulls ~2 MB and
takes a minute, `--headers-json` runs offline (what CI can use), `--model` reads a
local tree. Against the released checkpoint it reports **96,085 tensors /
475.2 GiB** -- the index's `metadata.total_size` matches the header sums byte for
byte -- and the contract passes.

Byte budget the planner has to live with:

| family | size | note |
| --- | --- | --- |
| routed experts (fp4 packed as I8) | 253.1 GiB | 84.4 GiB per projection, + 15.8 GiB of E8M0 scales |
| engram tables | 188.8 GiB | **two tables, 94.4 GiB each** |
| DSpark experts | 6.3 GiB + 0.4 scales | 128 experts top-3 across three stages |
| attention dense (fp8) | ~10 GiB | wq_b/wo_b 1.56 GiB each, wo_a 1.25, wq_a 0.24 |
| vision tower + aligner | ~2.3 GiB | text-only runs ignore it |
| embed / head | 2.5 GiB | untied |

Shapes that decide the port:

- `engram.embed.weight` is F8_E4M3 `[384006168, 256]` with an E8M0 scale per 32
  columns `[rows, 8]`: **94.4 GiB per engram layer, 40% of the model**, touched as
  ~24 rows per token per layer ((max_ngram_size - 1) x n_heads). It cannot be
  resident -- mmap it and let the page cache hold the rows a conversation touches,
  exactly the pattern AirLLM uses for Qwen3.8's n-gram table. Unlike routed
  experts the addresses are **token-deterministic** (they depend only on token
  ids), so a whole prompt's rows can be prefetched before the layer runs.
- `engram.{q,k}_weight` BF16 `[4, 5120]`, `engram.wkv` fp8 `[25600, 6144]`
  = `[5 x hidden, hidden + o_lora_rank]`.
- experts are packed fp4 in **I8** (`[moe, hidden/2]` for w1/w3, `[hidden, moe/2]`
  for w2) with per-32 E8M0 scales; dense weights are fp8-e4m3 with 32x32 UE8M0
  scales. Indexer, compressor and index-K are **BF16** in the checkpoint and
  quantized at runtime -- and with two different regimes: compressed KV is fp4 in
  groups of 16 with **E4M3** scales, index K fp4 in groups of 32 with **E8M0**.
- ownership, confirmed layer by layer: `compressor.wkv/norm` and
  `indexer.wk/k_norm` on `kv_source_layer_ids [2,8,14,20]`; `indexer.wq_b` +
  `weights_proj` on the eight `index_source_layer_ids`; `compressor.wgate` **only**
  where the compress ratio is 2 (layers 2, 8, 14 -- layer 20 has ratio 1); `engram.*`
  on `[1, 14]`; `attn_sink`/`q_norm`/`kv_norm`/`gate.bias_vl` on all 40 layers.
- `compress_ratios` is 43 long for 40 backbone + 3 DSpark layers: `0` for layers
  0-1 (pure sliding window, base rope), `2` for 2-19, `1` for 20-39, `0` for the
  three DSpark layers -- which is why they carry no compressor and no indexer.
- DSpark stages are **heterogeneous**: stage 0 owns `main_proj` fp8 `[5120, 15360]`
  and `main_norm` (15360 = 3 x hidden: the three target layers' captured hidden
  states), the last stage owns `markov_head.embed`/`head` BF16 `[129280, 256]`,
  `confidence_head.proj` BF16 `[1, 5376]` (5376 = hidden + markov_rank) and `norm`.
  Assuming three identical draft layers is the port bug this asserts against.
- vision: `patch_embed` `[1024, 14*14*3]`, MLP `w1` is the **fused gate+up**
  `[5632, 1024]` = 2 x `intermediate_size`, `w2` `[1024, 2816]`, `aligner.w1`
  `[5120, 9216]` = hidden x (1024 x 3x3) for `downsample_ratio: 3`.
- **zero `tid2eid` tensors**: V4.1 dropped V4's token-id hash routing, so its config
  has no `num_hash_layers` while the V4 parser *requires* that key -- a config-shape
  delta, not a footnote.

## The experts, read from the vendor's own converter

The routed experts ship as fp4 with **per-row scales over 32-wide blocks along K** -- the
shape rule is stated as an assertion in the released `inference/convert.py`:

```python
assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size   # 32
```

so `layers.0.ffn.experts.0.w1.scale` `[2304, 160]` is a `[2304, 5120]` weight and
`...w2.scale` `[5120, 72]` is `[5120, 2304]`: `moe_intermediate_size` is 2304, not the 73728
the shapes suggest when read the wrong way round. Their converter then casts the fp4 payload
into **fp8 e4m3** with the fp4 range folded into the scale (`cast_e2m1fn_to_e4m3fn`: a
per-block offset, `scale_max_offset_bits = scale.amax / 2**MAX_OFFSET_BITS`, re-expanded over
32), which is what lets the experts travel through an fp8 GEMM at the same 32-block geometry
as the dense weights.

Two consequences for the port:

- The 32x32 block geometry is not only the dense path's problem: it is the export format of
  the experts too, so one fix covers both and the fp4-specific machinery is avoidable if the
  loader performs the vendor's own cast.
- "Native load, no converter" stays true about *names and layout* -- but the experts do need
  this documented fp4-to-fp8 cast at load time. It is a transformation, not an external tool,
  and the delta doc should say so rather than let the earlier wording stand.

### The pin: the engine's fp4 read *is* the tensor the converter describes

`tests/v41_ops_probe.c` reads a synthetic expert payload through the engine's own
`coli_fp4_matvec_ref` (32 rows x 128 columns, every nibble code in both nibble positions,
per-row 32-column exponents spanning `2**-2 .. 2**2`), and `tools/check_deepseek_v41_ops.py`
dequantizes the *same* payload with the vendor's own `cast_e2m1fn_to_e4m3fn` -- lifted verbatim
out of `convert.py`'s AST, so safetensors and tqdm (which their converter imports) are not
needed to run their cast. The two sides must describe one tensor, so the comparison is an
equality, not a tolerance:

    fp4_expert_matvec   32x128 max |delta| 0.000e+00 (scales exercised: [0.25, 0.5, 1, 2, 4])
    fp4_tables          e2m1 grid identical to the vendor's, e8m0 decode max |delta| 0.000e+00

Scale 1 is one case among five, which is the pattern that hides a wrong rule. Three deliberate
mistakes move the same comparison off zero -- high nibble on the even column 9.0, the scale
block shifted by one 2.6, the fold ignored 189.0 -- so the check can fail. Verified on both
hosts: MinGW-w64 gcc 16.2 and Ubuntu gcc 15.2 produce byte-identical records for these two ops
(the only differences between the two captures are libm's last bits in `sinf`, in the
hyper-connection record).

So the checkpoint's fp4 payload with its per-row, per-32-column E8M0 exponent is exactly the
layout the engine's expert store already streams (`block_rows == 1`, `block_columns == 32`), and
`convert.py`'s fp4 -> fp8 cast is a converter *option* for the fp8-expert variant, not something
a loader owes the checkpoint.

The same reading finds the one thing that does have to change for the experts: the engine's fp4
matvec quantizes the *activation* at 128 (`deepseek_v41.c:17074`, inherited from V4), while the
reference passes `act_block_size=fp8_block_size` -- 32 -- for fp4 weights exactly as it does for
fp8 ones (`model.py`, `linear()`), and `fp4_gemm`'s host wrapper asserts the activation scale
count `== M * (K // act_block_size)`. The expert path needs the same 32-wide treatment as the
dense path: one sweep, not a second one.

## The layout map, held to the checkpoint and to the loader

`tools/deepseek_v41_layout.py` is the map the expert pin and the tiny fixture generator are
both written from: family -> shape -> encoding, derived from the config, with a `--check` mode
that holds it to the released headers and an `--engine` mode that diffs it against the names the
ported engine's loader actually declares (the per-layer plan's `add_*` calls, the expert store's
`layers.%d.ffn.experts.%d.%s.weight` builder, and every other name literal in the source).

Three encodings exist in this checkpoint and only three:

| encoding | families | scale |
| --- | --- | --- |
| fp8 e4m3 | every dense weight, the engram's `wkv`, `shared_experts.*`, DSpark `main_proj` | UE8M0, one exponent per **32x32** block |
| fp8 e4m3, per-row scale | the engram table `engram.embed.weight` | UE8M0, one exponent per row per 32 columns (`[rows, 8]`) |
| packed fp4 (I8, 2 values/byte along K) | the routed experts, backbone and DSpark | UE8M0, one exponent per row per 32 columns |

Held to the released checkpoint it classifies **96,085 tensors (48,496 weights + 47,589
scales)** with no shape or dtype disagreement and no unclassified name.

The same run settles two claims the port rests on:

- **"Native load, no converter"** is about names: the checkpoint already carries the
  converter's own vocabulary. `convert.py`'s seven `mapping` keys do match 333 tensor names,
  but every one of those is an *identity* rename (`wq_b` -> `wq_b`), and `mlp`->`ffn`,
  `self_attn`->`attn`, `weight_scale_inv`->`scale`, `e_score_correction_bias`->`bias` never
  fire at all. The one transformation a loader must perform is the experts' fp4 range (above).
- **The loader is still asking V4's questions.** Eleven families the engine declares do not
  exist in a V4.1 checkpoint:

  | the loader asks for | why it is not in the checkpoint |
  | --- | --- |
  | `attn.compressor.ape` | V4's compressor had a learned positional table; V4.1's has none |
  | `attn.indexer.compressor.{ape,norm,wgate,wkv}` | V4.1's compressor is `attn.compressor.*`, owned by the KV source layer, not by the indexer |
  | `ffn.gate.tid2eid` | token-id hash routing is gone (zero `tid2eid` tensors) |
  | `hc_head_{base,fn,scale}` | the head cache still reads V4's name for the final norm's hyper-connection |
  | `mtp.N.markov_head.markov_w1/w2` | V4.1's DSpark head is `markov_head.{embed,head}` |

  and thirty families it never asks for -- the mechanisms still to wire: `engram.*`,
  `attn.indexer.{wk,k_norm}`, `ffn.gate.bias_vl`, the DSpark head's `main_norm` / `norm` /
  `markov_head.*`, plus the vision tower and its aligner (deliberate).

  The plan's scale rule is the other half of the same finding: `add_fp8` declares
  `(rows + 127) / 128`, a 128x128 block, for every fp8 tensor it names.

## The tiny fixture, and what it falsified in the map

`tools/make_deepseek_v41_tiny.py` generates the family's oracle input: 168 tensors, 1.7 MB, in
`c/deepseek_v41_tiny/` (`config.json`, `tokenizer.json`, `model.safetensors`, `ref.json`). Its
source of truth is the vendor's own architecture -- the released `model.py`, imported unmodified
through the shim of `tools/deepseek_v41_reference.py`, filled with a fixed seed, quantized into
the checkpoint's encodings, and written *back into the model* before any reference generation
runs. So `ref.json`'s tokens are what those bytes produce. It needs no `transformers` (the V4
generator does): `ModelArgs` already carries a tiny default config, and the narrow scope is
vision off, `num_nextn_predict_layers: 0`, every `compress_ratio` 0 with no source layers, one
engram layer, and the dense path in the checkpoint's real 32x32 geometry.

Every tensor is checked against `tools/deepseek_v41_layout.py` **in both directions** -- the
vendor model may not hold a parameter the map lacks, and the map may not name a tensor the
fixture fails to produce -- and the map is then held to the released checkpoint again, unchanged
(96,085 tensors). Two runs are byte-identical, which is what lets `ref.json` be a fixture rather
than a record of one afternoon.

The first tiny geometry falsified three of the map's rules, all of them *accidentally right* on
the released checkpoint:

| rule | was | is (the vendor's own construction) | why the checkpoint could not tell |
| --- | --- | --- | --- |
| `engram.q_weight`, `k_weight` | `[max_ngram_size, hidden]` | `[hc_mult, hidden]` (`nn.Parameter(torch.ones(args.hc_mult, args.dim))`) | `hc_mult == max_ngram_size == 4` there |
| `engram.wkv` | `[5 * hidden, hidden + o_lora_rank]` | `[(hc_mult + 1) * hidden, (max_ngram_size - 1) * n_heads * head_dim]` | `5120 * 5 = 5 * hidden` and `3 * 8 * 256 = 6144 = hidden + o_lora_rank` |
| `ffn.gate.bias_vl` | always present | only where a vision tower is (`bias_vl` masks image spans) | the released checkpoint has a `vision_config` |

That is the fixture earning its keep before the engine can even load it: a shape rule read off
the checkpoint's numbers is a hypothesis, and a geometry where the candidate formulas disagree is
what turns it into a fact. The rules now come from the vendor's `__init__` calls, noted as such
in the tool.

`ref.json` carries three cases (`short`, `window` at 40 tokens, `engram` with repeated tokens)
as per-position expectations, both greedy continuations and the argmax after every prefix. Those
per-position expectations come from the *incremental* path -- one token per call with `start_pos`
advancing, which is what the engine does -- because the vendor's head returns only the last
position's logits by default (`Head.forward(..., full_logits=False)`). The generator asserts that
a second walk over the same prefix reproduces the first, so a cache that carried state across
walks would fail the fixture rather than the oracle.

## Shared KV / index: the exact mechanism

From the reference (`ref:500-580`, `ref:722-778`):

- `kv_source_layers` own the compressor and publish `compress_kv` + `index_k`; every
  other layer reads them. `compress_ratio > 0` does not make a layer an owner.
- `index_source_layers` run an indexer and publish `topk_idxs`; the layers between
  two sources reuse the published result instead of scoring again.
- two-level top-k: only `candidate_source_layer_id` (20) computes level-1 candidate
  blocks (2048 blocks of 8 positions, block score = its best position, newest partly
  filled block pinned in), and only layers *after* it mask their own scores by those
  blocks; earlier source layers do a plain global top-k.
- visibility: a compressed position is visible to a query only once the query has
  passed its last token -- the per-token mask a batched prefill needs. Pulsar reports
  the same trap on V4 (batched prefill decaying to single-token steps once the
  indexer engages).
- RoPE on compressed latents: group `j` is rotated at position `j * ratio`, with YaRN
  and `compress_rope_theta` when the ratio is non-zero, base `rope_theta` and no YaRN
  for the ratio-0 layers.

## Engram: the verified layout, and the exact forward

`tools/make_deepseek_v41_engram.py` reconstructs the two derived pieces of the
engram path and checks each against a number the checkpoint itself declares --
no weights involved, and both pass:

| derived artifact | check | result |
| --- | --- | --- |
| compressed token map | distinct classes == `engram_compressed_vocab_size` | **99,092 = 99,092** (15,502 classes collapse more than one token, largest 163) |
| prime bucket layout | the 24 moduli of a layer sum to that layer's declared table rows | **384,006,168** and **384,016,682** = `engram_num_embeddings` |

A wrong normalizer chain or a wrong prime-drawing order would miss those by a
wide margin, so both reconstructions are pinned to the reference.

**And then to the reference code itself.** Golden vectors produced by our own
transcription only prove the C agrees with the transcription: if the transcription
misreads the reference, both agree and both are wrong. So the C unit is held to the
*official* engram implementation instead
(`tools/check_deepseek_v41_engram_reference.py`, which runs `engram.py` under
torch):

| check | result |
| --- | --- |
| the reference builds its own compressed map and asserts its class count | passes (99,092) |
| primes / bucket offsets it derives internally | identical |
| multipliers from its own PCG64 seeding | identical |
| hash ids, prefill path, every sequence | match |
| hash ids, decode path (a single token against a filled cache) | match |

The addressing is therefore pinned end to end -- reference implementation, our
reconstruction, and the C unit -- before any weight is read. The map is
built with the checkpoint's own tokenizer (129,280 ids) over the normalizer chain
NFKC -> NFD -> strip accents -> lowercase -> collapse whitespace, with a
private-use sentinel so a one-space token survives `Strip()` (`--tokenizer`,
`--config`, `--output`).

**Frozen, not recomputed.** The per-layer hash multipliers come from numpy's PCG64
seeded with `10007 * layer_id`; numpy does not promise stream stability across
releases, and a different multiplier rehashes every row of a 94 GiB table. They
are therefore generated once and pinned in the layout file (four odd int64 per
layer, bounded so `token_id * multiplier` cannot overflow int64), together with
the primes and the flat bucket offsets (`--layout`).

### The addressing

Per position, on the *compressed* ids: the `max_ngram_size - 1` previous tokens
(start clamped to 0, any dead/image token blocking the n-gram) each get
`token * multiplier` XOR-ed into a rolling value; after step `i` that value is the
hash of the `(i+1)`-gram, taken `% prime` for each of the `n_heads` heads and
shifted by the layer's flat offset. That is `(max_ngram_size - 1) x n_heads` = 24
ids per position per layer, and the ids depend only on the token ids -- so they can
be computed (and their table rows prefetched) for a whole prompt before any layer
runs, unlike routed experts.

### The forward (`ref:296-365`)

1. `embed`: gather 24 rows of `head_dim` fp8 per position and dequantize with the
   row's E8M0 scales, one per 32 columns (row = 256 bytes + 8 scale bytes), then
   flatten to `24 * 256 = 6144`.
2. `wkv`: one matmul `[24 * head_dim -> dim * (hc_mult + 1)]` -- the released
   checkpoint stores it as fp8 `[25600, 6144]`, i.e. `5120 * 5 x 6144`, which is
   how the shape was confirmed.
3. split into `key` (`hc_mult x dim`) and `value` (`dim`); `weight` is
   `q_weight * k_weight`, only ever used as a product.
4. per (token, hc copy), `rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)`
   and `dot = sum(h * weight * key) * rstd * dim^-0.5` -- normalized per copy, not
   jointly.
5. `gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))`: a signed square root
   before the sigmoid, matching the training kernel.
6. `h + gate * value`, cast back to the stream's dtype. A token mask forces the
   gate to 0, which is what makes an image span pass through untouched.

### The two pieces implemented so far, held to the reference

`tests/engram_math_probe.c` prints its inputs and outputs, and
`tools/check_deepseek_v41_engram_math.py` rebuilds the reference from what it reads
-- extracting `Linear`, `ParallelEngramEmbedding` and `Engram` verbatim from the
reference model and running them under torch:

| piece | comparison | result |
| --- | --- | --- |
| row gather + fp8/E8M0 dequant | vs `ParallelEngramEmbedding.forward` | max delta **0** in fp32, identical after its bf16 rounding |
| gate maths (`rstd`, signed-sqrt sigmoid, per-copy) | vs `Engram.forward`, with the projection injected | max delta **1.2e-07** (reduction order) |

The projection itself is the engine's shared fp8 matvec, not new code. Both pieces
are also covered by hand-derived cases in `tests/test_deepseek_v41.c` (80 checks),
including the fail-closed edges: a hash row past the end of the mapping, a negative
row id, a `head_dim` that is not a multiple of 32, a negative eps, an empty hc
count.

Consequence for the port: the 94.4 GiB per layer is *gathered*, 24 rows per token,
and the working set of an interactive session is a few tens of MB of rows -- so the
tables are memory-mapped and left to the page cache, never loaded, never LRU-managed
like the expert store. `wkv` and the rest of the path reuse the engine's existing
fp8 machinery.

### Where the token map lives (decided with measurements)

The engine needs the 129,280-entry id -> class map at load. Measured on the real
map (99,092 classes, 30,188 repeats):

| encoding | bytes | as C source |
| --- | --- | --- |
| raw uint32 | 517,120 | ~2.5 MB |
| 17-bit packed | 274,720 | ~1.3 MB |
| varint deltas | 223,635 | ~1.1 MB |
| **bitmap of first appearances + 17-bit representative per repeat** | **80,310** | **503 KB** |
| (zlib floor on any of the above) | ~77,000 | – |

Classes are handed out in token-id order and a repeat always names an *earlier*
id, so the map is fully described by which ids are first appearances plus, for
every other id, which earlier id it repeats. That is what the generated
`deepseek_v41_engram_tokens.h` carries, and it is what the engine rebuilds.

Committed as a generated header rather than a runtime file, so the engine still
needs nothing but the checkpoint: `qwen38_nfc_tables.h` already ships 673 KB of
generated table in this tree, so 503 KB is within the established shape, and no
user has to run a tool before running a model. Regeneration is two commands, both
recorded at the top of the header.

### What the rebuild guarantees

| check | result |
| --- | --- |
| our map vs the reference's own map, entry by entry | 129,280 compared, **0 differing** |
| the C rebuild's class count, samples and FNV-1a fingerprint | match (64 checks in `tests/test_deepseek_v41.c`) |
| a payload that disagrees with itself (repeat before its representative, too-small buffer) | refused, not guessed |

## Dense fp8 blocks: 32x32, where the V4 engine hardcodes 128

Every fp8 weight in the released checkpoint is scaled in **32x32 blocks**, because
`quantization_config.weight_block_size` is `[32, 32]`. The header read confirms it
independently of the config: `attn.wq_a.scale` is `[40, 160]` for a `[1280, 5120]`
weight, and `engram.wkv.scale` is `[800, 192]` for `[25600, 6144]` -- rows/32 by
columns/32 in both cases. V4's config declares `[128, 128]`, and the engine's shared
matvec is written for that: the 128 appears in `fp8_matvec_validate` (both the
accepted `block_columns` and the scale size it demands), in the AVX2 tile path, in
the activation qdq width it passes, and inside `matmul_fp8`.

The reference settles both sides in one line, and names the activation explicitly:

```python
fp8_block_size = 32  # one fp8 scale per 32x32 weight block / 32 activations
```

It is the constant behind `act_quant(x, fp8_block_size, ...)` **and** the `block_size`
handed to `fp8_gemm`, so V4.1 quantizes the *activations* in 32-wide blocks too -- the
engine's 128 is wrong on that side as well, not only for the weight scales. The engram
table's geometry is the same constant: `ParallelEngramEmbedding` dequantizes with
`values.unflatten(-1, (-1, self.block_size))`, which is why the checkpoint stores 8
E8M0 scales for a 256-wide row.

The config now carries it rather than assuming it: `weight_block_size` is read from the
checkpoint into `fp8_block_rows`/`fp8_block_columns`, and any width but the verified 32 is
refused at parse time with the reason spelled out -- the vectorized matvec and the loader
still hardcode V4's 128, so accepting 128 here would dequantize every dense weight with the
wrong scale and say nothing. The gate goes away with the loader and the matvec taking the
width from the view.

So the dense path is a **blocker for any V4.1 layer**: run it as-is and the engine
would read the wrong scale for every weight it touches. The engram projection does
not: `coli_v41_fp8_matvec_blocked` takes the geometry from the view
(`block_columns`, and `block_rows` for the scale row), quantizes the activation over
the same width exactly as the reference's `Linear` does before its fp8 GEMM, and is
covered by a hand-checked case (a 1x32 block of two non-zero weights against a
computed product, plus the width and scale-size refusals). The dense layers still
need the same treatment, and that is the next thing that must land before the engine
can run one V4.1 layer at all.

### Where, exactly (and what became of it)

The sweep landed in `c/deepseek_v41.c` at `bcff526`'s child; the counts below are what it touched,
and the line numbers were read at `0c1b504` and drift.

| what | where | count |
| --- | --- | --- |
| `packed_rows8 ? 8 : 128, 128` -- the per-unit builds of a view's block geometry | 2387, 2849, 3997, 4737, 6848, 7331 | 6 |
| `fp8_view` -- the per-unit helpers that fill it in (three, not two: the first count came from a truncated grep) | 2369, 2831, 7313 | 3 |
| the grouped `wo_a` view -- both the scale geometry *and* the per-group scale offsets were at 128 | 2646, 3053, 7535 | 3 |
| `v41_fp8_pack_rows8_inplace` -- V4's runtime packing, which *redefines* `block_rows` to 8 to describe its own on-disk form | 794 | 1 |
| the shared activation buffer (`input_act`) that fed wq_a and wkv at 128 | 2449, 2911, 7398 | 3 |
| the shared matvec calls in the dense path (attention projections, grouped wo_a, indexer query projection, shared experts) | 12 + 3 + 3 + 6 | 24 |
| `coli_v41_fp8_matvec_blocked` -- the family matvec that takes the geometry from the view | 1930 | the dispatch target |

Two rules made those edits one sweep rather than six: the packing flag is now off for V4.1 (it
describes V4's own on-disk form, and leaving it on would tell the view a geometry the checkpoint
does not have), and every site reaches `coli_v41_fp8_matvec_blocked` -- a site left on the shared
matvec keeps V4's 128 and says nothing. The shared matvec now *refuses* a 32-wide view
(`fp8_matvec_validate` demands `block_columns == 128`), which is what turns a partial sweep into a
loud failure instead of a quiet one: the probe asks both matvecs the same question and records
`shared refused (-1), family accepted (0)`, with the family product agreeing with torch to
`0.000e+00`.

### The table is mapped, never read, and the driver stays per position

`coli_v41_engram_table_open` maps the two tensor ranges straight out of the shard with
`compat_map_readonly` -- the same primitive the engine already uses for weights, whose
comment describes exactly this case (pages faulted on read, reclaimable file-backed
cache). Nothing is copied and nothing is dequantized up front, which is what makes a
94.4 GiB table per layer a non-event: the scale bytes stay E8M0 and are decoded per
row, because expanding them to f32 would multiply the footprint by eight for a table
this size. The engine's own view convention for *dense* weights is the opposite (the
loader expands E8M0 into f32 before the layer runs) -- the two are not interchangeable,
and the table path deliberately keeps the raw bytes.

`coli_v41_engram_workspace_init` owns the per-position buffers (the fetched rows, the
hash ids, the projection, the fused `q_weight * k_weight`), and the workspace is sized
for the longest span it may serve: `hash_ids` writes `hash_cols` entries **per
position**, so a workspace built for one position silently overran its hash buffer on
a two-position span. It now carries `max_positions` and `span` refuses a longer span
instead. `coli_v41_engram_apply` does one position (hashes -> gather -> projection ->
gate -> residue) and `coli_v41_engram_span` walks a run of classes, refusing an
overlong span, a missing workspace, a row past the table end, and a `wkv` whose columns
do not match the hash width.

The contract test keeps the two halves apart on purpose: the gate and the gather are
held to the reference's own classes, while the driver is held to the plumbing it alone
owns (the split offsets, per-position routing, the fail-closed shapes). A synthetic
4-row table cannot hold real hash ids -- they are offsets over 24 primes, millions of
rows -- so `span` refusing it is asserted as the fail-closed property, and the routing
is checked with a `wkv` whose rows pick different input columns: shifted hashes must
leave different residuals.

## Port plan

1. `c/family_registry.py`: `deepseek_v41` descriptor + `_dsv41_geometry` planner. **done**
2. Native load (no converter: same fp4-experts/fp8-dense layout). **done — verdict recorded**
3. `c/deepseek_v41.c`: fork of the V4 engine — `tools/fork_deepseek_v41.py`
   (renames only, `--check` in sync), `Makefile.deepseek-v41(.units)`, host gate and
   fail-closed portable artifact. **done**
4. V4.1 config shape: `text_config` unwrapping + the new keys (source-layer ids,
   candidate/engram/dspark blocks), with the gates above. **next**
5. Shared KV/index indirection in `coli_v41_layer_plan`, `context_bytes` and the
   layer runtime; then engram (token map generator + hash layout), then the DSpark
   head layout.
6. Oracles: `tools/make_deepseek_v41_tiny.py` + `tools/dsv41_ref_numpy.py` (a numpy
   transcription of the reference model, since the V4 tiny path needs
   `DeepseekV41ForCausalLM` from Transformers) and `tests/test_deepseek_v41.py`;
   teacher-force 32/32 before any perf work. `tests/test_deepseek_v41.c` (config,
   routing, gates) is the C-side contract and its Makefile rule is already in place.
7. `docs/deepseek-v41.md`, serve path, segment/edge adapters, site/release prose.

## Status

- [x] Branch `deepseek-v41`, upstream v1.10.2
- [x] V4.1 config captured, delta analysis, tensor inventory, native-load verdict
- [x] Registry descriptor + planner; registry tests green
- [x] Engine fork + build wiring; CPU-only build verified with mingw-w64 gcc 16.2
      (27 units, LTO; warning profile identical to a tier-less V4 build, checked by
      `tools/v41_warning_parity.py`)
- [x] Checkpoint contract verified from headers alone (96,085 tensors / 475.2 GiB,
      ownership sets, engram sizes) by `tools/check_deepseek_v41_checkpoint.py`
- [x] V4.1 config shape + fail-closed gates in the engine
- [x] Engram table mapped out of the shard (no copy, E8M0 scales kept raw)
- [x] Engram per-position driver + workspace, fail-closed on every shape
- [x]  Dense fp8 block geometry: the views take the width from the checkpoint's scale shape, V4's
      rows8 packing is off, and every dense site -- the attention projections, the grouped `wo_a`,
      the indexer's query projection, the shared experts -- dispatches to
      `coli_v41_fp8_matvec_blocked`. The shared matvec refuses a 32-wide view instead of reading
      it as 128, and the path is pinned against torch (32x128, delta `0.000e+00`)
- [x] Verified on Linux as well as Windows: from a clean clone of the pushed commit,
      on gcc 15.2 / Ubuntu 26.04, the engine builds (0 errors), the contract test
      passes 110/110 checks with 0 warnings of its own, the registry test passes
      48/48, and the fork adds no warning over the V4 base unit by unit
      (`c/tools/v41_warning_parity.py`: 5 and 5, 0 units differing)
- [x] Engram layout verified against the checkpoint's own numbers (token map
      classes, table rows) by `tools/make_deepseek_v41_engram.py`
- [x] Engram hash addressing implemented in C and pinned to the *official*
      implementation (prefill and decode), `COLI_V41_UNIT_ENGRAM`
- [x] Token map storage decided by measurement and rebuilt in C, byte-identical to
      the reference's map
- [x] Engram row gather/dequant and the gate maths in C, verified against the
      reference's own classes (max delta 0 and 1.2e-07)
- [ ] Shared KV/index, engram path, DSpark
- [ ] Tiny oracle 32/32
