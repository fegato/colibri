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
- [ ] V4.1 config shape + fail-closed gates in the engine
- [ ] Shared KV/index, engram, DSpark
- [ ] Tiny oracle 32/32
