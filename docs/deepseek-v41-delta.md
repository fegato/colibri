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
wide margin, so both reconstructions are pinned to the reference. The map is
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

Consequence for the port: the 94.4 GiB per layer is *gathered*, 24 rows per token,
and the working set of an interactive session is a few tens of MB of rows -- so the
tables are memory-mapped and left to the page cache, never loaded, never LRU-managed
like the expert store. `wkv` and the rest of the path reuse the engine's existing
fp8 machinery.

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
- [ ] V4.1 config shape + fail-closed gates in the engine
- [x] Engram layout verified against the checkpoint's own numbers (token map
      classes, table rows) by `tools/make_deepseek_v41_engram.py`
- [ ] Shared KV/index, engram path, DSpark
- [ ] Tiny oracle 32/32
