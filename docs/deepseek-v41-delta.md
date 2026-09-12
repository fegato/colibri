# DeepSeek V4.1-Flash port — architecture delta vs V4 (colibri `deepseek_v4` engine)

Source: `deepseek-ai/DeepSeek-V4.1-Flash` config.json (`model_type: deepseek_v41`,
arch `DeepseekV41ForCausalLM`) vs `deepseek-ai/DeepSeek-V4-Flash-0731`
(`model_type: deepseek_v4`, 43 layers / hidden 4096 / 256 experts + 1 shared / top-6).

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
| expert weight dtype | native fp4 | **fp4** (`expert_dtype`), dense fp8-e4m3 UE8M0 — same layout family |

## New mechanisms (the real work)

1. **Router**: `scoring_func: sqrtsoftplus`, `topk_method: noaux_tc`,
   `norm_topk_prob: true`, `routed_scaling_factor: 1.5`.
   The V4 engine's router (sigmoid/softmax + aux-loss-free variant) needs a new
   scoring path + oracle check.
2. **KV sharing across layers**: `kv_source_layer_ids: [2, 8, 14, 20]` —
   some layers reuse another layer's KV instead of computing their own.
   The engine's per-layer KV planner (`context_bytes`, cache slots) assumes
   one KV per layer; this needs a source-layer indirection.
3. **Index sharing**: `index_source_layer_ids` (9 entries), `index_n_heads: 32`,
   `index_head_dim: 128`, `index_topk: 512`, plus `candidate_source_layer_id: 20`,
   `candidate_topk_blocks: 2048`, `candidate_block_size: 8` — the DSA sparse
   index is shared/computed from designated layers, not per-layer.
4. **Engram (NEW)**: `engram_layer_ids: [1, 14]`, n-gram embedding tables
   (`engram_vocab_size: 16000000`, `engram_compressed_vocab_size: 99092`,
   `engram_max_ngram_size: 4`, 8 heads × 256 dim). Entirely new tensor family
   the converter must classify and the engine must embed + attend.
5. **DSpark (speculative-ish, NEW)**: `num_nextn_predict_layers: 3`,
   `dspark_target_layer_ids: [37, 38, 39]`, `dspark_block_size: 5`,
   `dspark_markov_rank: 256`, `dspark_n_routed_experts: 128`,
   `dspark_num_experts_per_tok: 3`. Related to the existing V4 `dspark`/`MTP`
   include (`deepseek_v4_dspark.inc`) but with its own expert set and
   markov-rank projection — port, don't assume reuse.
6. **swiglu_limit: 10.0** — output clamp on the SwiGLU, one line in the FFN
   path plus oracle coverage.
7. **Vision tower** (`deepseek_v41_vision`: 32 layers, hidden 1024, 16 heads,
   patch 14, `max_image_tokens: 1024`): mirrors the `qwen38_vision.h` /
   `glm53_image.py` pattern — needs `tools/v41_image.py` + tower code.
8. **Asymmetric active params** (8B input / 16B output): prefill vs decode
   route different widths — the planner's prefill/decode split must model it.

## Port plan (mirrors how glm53/qwen38 were added)

1. `c/family_registry.py`: new `deepseek_v41` descriptor (model_types,
   geometry, planner budgets). Loader rejects unknown tensors loudly.
2. `c/tools/convert_v41.py` (or native-load if layout allows): classify every
   tensor name explicitly; engram tables + dspark experts are new kinds.
3. `c/deepseek_v41.c` (+ `.h`): fork from `deepseek_v4.c` — router scoring,
   KV/index source indirection, engram embed, dspark layers, swiglu clamp.
4. Oracles: `tools/make_deepseek_v41_tiny.py` + `dsv41_*_oracle.py` following
   the `make_deepseek_v4_tiny.py` / `dsv4_*_oracle.py` pattern; teacher-force
   32/32 before any perf work.
5. `docs/deepseek-v41.md` + registry tests + release wiring (`coli`, Makefile,
   Windows launcher).

## Status

- [x] Branch `deepseek-v41` created, upstream v1.10.2
- [x] V4.1 config captured (`model_type: deepseek_v41`)
- [x] Delta analysis (this file)
- [ ] Registry descriptor
- [ ] Converter / native-load mapping
- [ ] Engine fork
- [ ] Tiny oracle + validation
