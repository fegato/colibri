#ifndef COLIBRI_DEEPSEEK_V41_INTERNAL_H
#define COLIBRI_DEEPSEEK_V41_INTERNAL_H

/*
 * Internal DeepSeek-V4 API. Not a stability commitment.
 * External callers should use deepseek_v41.h (engine / config / prompt).
 */
#include "deepseek_v41.h"

#include <stdio.h>
#include "tensor.h"
#include "expert_store.h"
#include "deepseek_v41_expert_store_registry.h"
#include "native_quant.h"
#include "native_quant_batch.h"
#include "native_quant_dual.h"
#include "native_quant_fp4_rows16.h"
#include "st.h"

/* Worker count for the persistent expert-loader pool in the block pipeline
 * (deepseek_v41_block_pipeline.c). Shared here so the CLI can size the OpenMP
 * team around the loaders instead of scheduling compute onto their CPUs. */
#ifndef COLI_V41_EXPERT_LOADER_COUNT
#define COLI_V41_EXPERT_LOADER_COUNT 3
#endif

#define COLI_ST_MAX_RANK ST_MAX_RANK
#define COLI_ST_BF16 0
#define COLI_ST_F16 1
#define COLI_ST_F32 2
#define COLI_ST_U8 3
#define COLI_ST_I8 3
#define COLI_ST_F8_E4M3 4
#define COLI_ST_F8_E8M0 5
#define COLI_ST_I64 6

typedef int ColiSafetensorsDType;
typedef st_tensor ColiSafetensorsTensor;
typedef shards ColiSafetensorsIndex;

typedef struct {
    ColiTensorView view;
    void *data_allocation;
    void *scale_allocation;
} ColiOwnedTensor;

typedef struct {
    float *data;
    uint64_t count;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
} ColiFloatTensor;

int coli_st_index_open(ColiSafetensorsIndex **out, const char *directory,
                       char *error, size_t error_size);
void coli_st_index_close(ColiSafetensorsIndex *index);
size_t coli_st_tensor_count(const ColiSafetensorsIndex *index);
size_t coli_st_shard_count(const ColiSafetensorsIndex *index);
const char *coli_st_shard_path(const ColiSafetensorsIndex *index, int shard);
const ColiSafetensorsTensor *coli_st_find(const ColiSafetensorsIndex *index,
                                         const char *name);
int coli_st_tensor_shard(const ColiSafetensorsIndex *index,
                         const ColiSafetensorsTensor *tensor);
int coli_st_read_tensor(const ColiSafetensorsIndex *index,
                        const ColiSafetensorsTensor *tensor, void *destination);
int coli_st_read_at(const ColiSafetensorsIndex *index, int shard,
                    uint64_t offset, size_t length, void *destination);
/* Large, transient SSD read: prefer the index's O_DIRECT twin and use an
 * aligned bounce buffer, falling back to the ordinary buffered path. */
int coli_st_read_at_streaming(const ColiSafetensorsIndex *index, int shard,
                              uint64_t offset, size_t length,
                              void *destination);
int coli_st_streaming_direct_available(const ColiSafetensorsIndex *index,
                                       int shard);
int coli_st_prefetch_at(const ColiSafetensorsIndex *index, int shard,
                        uint64_t offset, size_t length);
const char *coli_st_dtype_name(ColiSafetensorsDType dtype);

/* ==== begin dual-SSD mirror (COLI_MODEL_MIRROR / SNAP_MIRROR) ==== */

/* Registers the read replicas listed in COLI_MODEL_MIRROR (or SNAP_MIRROR) on
 * `index` and derives the per-drive expert read split from COLI_DISK_WEIGHTS
 * or a startup bandwidth probe (colibri.c mirror_setup semantics). Runs after
 * the index is open and before any expert load. Returns 1 when a usable mirror
 * is active, 0 when none, -1 on error. */
int coli_st_mirror_setup(ColiSafetensorsIndex *index, const char *model_dir,
                         int experts_per_layer);
int coli_st_streaming_direct_available_rep(const ColiSafetensorsIndex *index,
                                           int shard, int rep);

/* Replica (0 = primary, 1..nrep-1 = mirrors) serving expert (layer, eid). */
int coli_st_expert_route(int layer, int eid);
int coli_st_mirror_active(void);
int coli_st_mirror_nrep(void);

/* Rep-aware reads: route to the replica fd (falling back to the primary when
 * the shard is absent there or on read error) and account bytes per drive. */
int coli_st_read_at_rep(const ColiSafetensorsIndex *index, int shard, int rep,
                        uint64_t offset, size_t length, void *destination);
int coli_st_read_at_streaming_rep(const ColiSafetensorsIndex *index, int shard,
                                  int rep, uint64_t offset, size_t length,
                                  void *destination);
int coli_st_prefetch_at_rep(const ColiSafetensorsIndex *index, int shard,
                            int rep, uint64_t offset, size_t length);

/* Per-drive I/O telemetry (bytes / read count), index [0] primary, [r] mirror. */
extern uint64_t g_v41_mir_bytes[1 + ST_MAX_MIR];
extern uint64_t g_v41_mir_nread[1 + ST_MAX_MIR];
/* ==== end dual-SSD mirror ==== */

int coli_tensor_load_fp8(ColiOwnedTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *prefix, char *error, size_t error_size);
void coli_owned_tensor_free(ColiOwnedTensor *tensor);
int coli_tensor_load_f32(ColiFloatTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *name, char *error, size_t error_size);
void coli_float_tensor_free(ColiFloatTensor *tensor);

typedef struct ColiV41Engine ColiV41Engine;

/* Runtime-selected full DSpark profile, shared with the separately compiled
 * generation unit. */
extern int coli_v41_full_dspark_wanted;
double coli_v41_dspark_cache_gb(void);

/* ==== begin deepseek_v41_math.h ==== */

#ifdef __cplusplus
extern "C" {
#endif

int coli_v41_hc_split_sinkhorn(float *pre, float *post, float *comb,
                              const float *mixes, const float scale[3],
                              const float *base, int hc, int iterations,
                              float eps);

int coli_v41_hc_pre(float *output, float *post, float *comb,
                   const float *input, const float *hc_fn,
                   const float scale[3], const float *base,
                   int hc, int dimension, int iterations,
                   float norm_eps, float hc_eps);

int coli_v41_hc_post(float *output, const float *branch,
                    const float *residual, const float *post,
                    const float *comb, int hc, int dimension);

int coli_v41_rmsnorm(float *output, const float *input, const float *weight,
                    int dimension, float eps);

int coli_v41_rope_precompute(float *cosines, float *sines,
                            int dimension, int sequence_length,
                            int original_sequence_length, float base,
                            float factor, int beta_fast, int beta_slow);

int coli_v41_rope_precompute_range(float *cosines, float *sines,
                                  int dimension, int start_position,
                                  int sequence_length,
                                  int original_sequence_length, float base,
                                  float factor, int beta_fast, int beta_slow);

int coli_v41_rope_position(float *cosines, float *sines,
                          int dimension, int position,
                          int original_sequence_length, float base,
                          float factor, int beta_fast, int beta_slow);

int coli_v41_rope_apply(float *vectors, int vector_count, int dimension,
                       const float *cosines, const float *sines, int inverse);

/* V41 DELTA: engram addressing. The tables are memory-mapped (94.4 GiB per layer),
 * so this is the derived part: token id -> compressed class, and the n-gram hash
 * ids that index a layer's table. Pinned to the reference by
 * tests/deepseek_v41_engram_vectors.h. */
/* The engram table: two read-only mappings (weight rows and their E8M0 scales) of
 * the same shard, plus the row count they agree on. */
typedef struct {
    compat_ro_map weight_map;
    compat_ro_map scale_map;
    const uint8_t *rows;
    size_t rows_bytes;
    const uint8_t *scales;
    size_t scales_bytes;
    int64_t row_count;
} ColiV41EngramTable;

/* Reusable per-layer buffers, so a prefill chunk allocates nothing per position. */
typedef struct {
    float *rows;
    int64_t *hashes;
    float *projection;
    float *weight;
    int hash_cols;
    int head_dim;
    int hc_mult;
    int dim;
    int max_positions;
} ColiV41EngramWorkspace;

int coli_v41_engram_table_open(ColiV41EngramTable *table, const char *shard_path,
                               int64_t weight_offset, size_t weight_bytes,
                               int64_t scale_offset, size_t scale_bytes,
                               int head_dim, char *error, size_t error_size);
void coli_v41_engram_table_close(ColiV41EngramTable *table);
int coli_v41_engram_workspace_init(ColiV41EngramWorkspace *work, int hash_cols,
                                   int head_dim, int hc_mult, int dim,
                                   int max_positions);
void coli_v41_engram_workspace_free(ColiV41EngramWorkspace *work);
int coli_v41_engram_apply(float *stream, int hc_mult, int dim,
                          const int64_t *hashes, const ColiV41EngramTable *table,
                          int head_dim, const ColiTensorView *wkv,
                          const float *weight, float eps,
                          ColiV41EngramWorkspace *work);
int coli_v41_engram_span(float *stream, int hc_mult, int dim, const int *classes,
                         int count, const int *history, int history_count,
                         int pad_class, int layer_position,
                         const ColiV41EngramTable *table, int head_dim,
                         const ColiTensorView *wkv, const float *weight, float eps,
                         ColiV41EngramWorkspace *work);
int coli_v41_fp8_matvec_blocked(float *output, const ColiTensorView *weight,
                                const float *input);
int coli_v41_engram_build_token_map(uint32_t *map, int map_count);
uint64_t coli_v41_engram_token_map_digest(const uint32_t *map, int count);
int coli_v41_engram_fetch_rows(float *output, const int64_t *rows, int row_count,
                               int head_dim, const uint8_t *table,
                               size_t table_bytes, const uint8_t *scale_table,
                               size_t scale_bytes);
int coli_v41_engram_gate(float *output, const float *stream, const float *key,
                         const float *value, const float *weight, int hc_mult,
                         int dim, float eps);
int coli_v41_engram_layer_position(int layer_id);
int coli_v41_engram_compress(int *classes, const int *ids, int count,
                             const uint32_t *token_map, int map_count, int dead);
int coli_v41_engram_hash_ids(int64_t *output, int layer_position,
                             const int *classes, int count,
                             const int *history, int history_count,
                             int pad_class);

int coli_v41_route(float *weights, int *indices, const float *hidden,
                  const float *gate, const float *bias,
                  const int *forced_indices, int experts, int dimension,
                  int topk, float route_scale);

int coli_v41_swiglu(float *output, const float *gate, const float *up,
                   int dimension, float limit);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_math.h ==== */

/* ==== begin deepseek_v41_layer.h ==== */

#include <stddef.h>
#include <stdint.h>

/* amalgamated: deepseek_v41_config.h */

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_V41_MAX_LAYER_TENSORS 48
#define COLI_V41_MAX_TENSOR_NAME 160

typedef struct {
    char name[COLI_V41_MAX_TENSOR_NAME];
    ColiSafetensorsDType dtype;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
    /* Dense FP8 weights may be transposed inside each 8-row tile after load.
     * This is an in-memory execution layout only; checkpoint bytes and scales
     * remain unchanged. */
    int packed_rows8;
} ColiDeepSeekV41TensorSpec;

typedef struct {
    int layer;
    int compression_ratio;
    int uses_hash_router;
    int has_compressor;
    int has_indexer;
    size_t tensor_count;
    ColiDeepSeekV41TensorSpec tensors[COLI_V41_MAX_LAYER_TENSORS];
} ColiDeepSeekV41LayerPlan;

typedef struct {
    size_t tensor_count;
    uint64_t total_bytes;
    uint64_t bf16_bytes;
    uint64_t f32_bytes;
    uint64_t fp8_weight_bytes;
    uint64_t fp8_scale_bytes;
    uint64_t i64_bytes;
} ColiDeepSeekV41LayerStats;

typedef struct {
    ColiDeepSeekV41LayerPlan plan;
    ColiDeepSeekV41LayerStats stats;
    void *data[COLI_V41_MAX_LAYER_TENSORS];
    /* Optional per-tensor backend-resident mirrors (Dsv4CudaTensor* on the CUDA
     * tier). Aligned 1:1 with plan.tensors[]; owned by the engine's GPU tier. */
    void *gpu[COLI_V41_MAX_LAYER_TENSORS];
} ColiDeepSeekV41LayerWeights;

int coli_v41_layer_plan(ColiDeepSeekV41LayerPlan *plan,
                       const ColiDeepSeekV41Config *config, int layer,
                       char *error, size_t error_size);
int coli_v41_layer_validate(const ColiDeepSeekV41LayerPlan *plan,
                           const ColiSafetensorsIndex *index,
                           ColiDeepSeekV41LayerStats *stats,
                           char *error, size_t error_size);
int coli_v41_layer_load(ColiV41Engine *engine,
                       ColiDeepSeekV41LayerWeights *weights,
                       const ColiDeepSeekV41Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size);
void coli_v41_layer_free(ColiV41Engine *engine,
                        ColiDeepSeekV41LayerWeights *weights);
const void *coli_v41_layer_data(const ColiDeepSeekV41LayerWeights *weights,
                               const char *name,
                               const ColiDeepSeekV41TensorSpec **spec);

/* Backend-mirror accessors: the gpu handle attached to the tensor named
 * "layers.<N>.<suffix>.weight", or NULL when the tier did not upload it. */
void *coli_v41_layer_gpu(const ColiDeepSeekV41LayerWeights *weights,
                        const char *suffix);
int coli_v41_layer_gpu_set(ColiDeepSeekV41LayerWeights *weights,
                          const char *suffix, void *handle);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_layer.h ==== */

/* ==== begin deepseek_v41_sparse_attention.h ==== */

#ifdef __cplusplus
extern "C" {
#endif

int coli_v41_sparse_attention_ref(float *output, const float *queries,
                                 const float *kv, const float *sinks,
                                 const int *indices, int heads,
                                 int head_dimension, int kv_count, int topk,
                                 float softmax_scale);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_sparse_attention.h ==== */

/* ==== begin deepseek_v41_kv_cache.h ==== */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV41KVCache ColiDeepSeekV41KVCache;

int coli_v41_kv_cache_create(ColiDeepSeekV41KVCache **cache,
                            int window_size, int compression_ratio,
                            int head_dimension, int max_context);
void coli_v41_kv_cache_reset(ColiDeepSeekV41KVCache *cache);
void coli_v41_kv_cache_destroy(ColiDeepSeekV41KVCache *cache);
int coli_v41_kv_cache_put_window(ColiDeepSeekV41KVCache *cache,
                                int position, const float *kv);
int coli_v41_kv_cache_put_compressed(ColiDeepSeekV41KVCache *cache,
                                    int position, const float *kv);
int coli_v41_kv_cache_indices(const ColiDeepSeekV41KVCache *cache,
                             int position, int *indices, size_t capacity);
const float *coli_v41_kv_cache_values(const ColiDeepSeekV41KVCache *cache);
int coli_v41_kv_cache_value_count(const ColiDeepSeekV41KVCache *cache);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_kv_cache.h ==== */

/* ==== begin deepseek_v41_attention_cache.h ==== */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV41AttentionCache ColiDeepSeekV41AttentionCache;

int coli_v41_attention_cache_create(ColiDeepSeekV41AttentionCache **cache,
                                   int window_size, int compression_ratio,
                                   int head_dimension, int max_context);
void coli_v41_attention_cache_reset(ColiDeepSeekV41AttentionCache *cache);
void coli_v41_attention_cache_destroy(ColiDeepSeekV41AttentionCache *cache);

/* query is [heads, head_dimension]. window_kv and compressed_kv have one
 * head_dimension vector each. compressed_kv is required at ratio boundaries. */
int coli_v41_attention_cache_step(ColiDeepSeekV41AttentionCache *cache,
                                 float *output, const float *query,
                                 const float *window_kv,
                                 const float *compressed_kv,
                                 const float *sinks, int heads,
                                 int position, float softmax_scale);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_attention_cache.h ==== */

/* ==== begin deepseek_v41_attention.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v41_config.h */
/* amalgamated: deepseek_v41_layer.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV41WindowAttentionState
    ColiDeepSeekV41WindowAttentionState;

int coli_v41_window_attention_prepare(ColiDeepSeekV41WindowAttentionState *state,
                                     const ColiDeepSeekV41LayerWeights *weights,
                                     const ColiDeepSeekV41Config *config,
                                     char *error, size_t error_size);
int coli_v41_window_attention_create(ColiDeepSeekV41WindowAttentionState **state,
                                    const ColiDeepSeekV41Config *config);
void coli_v41_window_attention_reset(ColiDeepSeekV41WindowAttentionState *state);
void coli_v41_window_attention_destroy(ColiDeepSeekV41WindowAttentionState *state);

/* Correctness-first single-KV attention. Compressed layers may use this at
 * position zero, before any compressed KV/indexer candidate exists. */
int coli_v41_attention_token_ref(float *output,
                                const ColiDeepSeekV41LayerWeights *weights,
                                const ColiDeepSeekV41Config *config,
                                const float *input, int position,
                                char *error, size_t error_size);
int coli_v41_attention_window_token_ref(
    float *output, ColiDeepSeekV41WindowAttentionState *state,
    const ColiDeepSeekV41LayerWeights *weights,
    const ColiDeepSeekV41Config *config, const float *input, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_attention.h ==== */

/* ==== begin deepseek_v41_attention_batch.h ==== */

/* amalgamated: deepseek_v41_attention.h */

int coli_v41_attention_window_batch_ref(
    float *outputs, ColiDeepSeekV41WindowAttentionState *state,
    const ColiDeepSeekV41LayerWeights *weights,
    const ColiDeepSeekV41Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size);
/* ==== end deepseek_v41_attention_batch.h ==== */

/* ==== begin deepseek_v41_attention_transaction.h ==== */

/* amalgamated: deepseek_v41_attention.h */

typedef struct ColiV41AttentionSnapshot ColiV41AttentionSnapshot;

int coli_v41_attention_snapshot_create(
    const ColiDeepSeekV41WindowAttentionState *state,
    ColiV41AttentionSnapshot **output);
int coli_v41_attention_snapshot_restore(
    ColiDeepSeekV41WindowAttentionState *state,
    const ColiV41AttentionSnapshot *snapshot);
void coli_v41_attention_snapshot_destroy(ColiV41AttentionSnapshot *snapshot);
/* Disk (de)serialization of a snapshot (little-endian raw fields + arrays).
 * write returns 0 on success; read allocates *output (0 on success). */
int coli_v41_attention_snapshot_write(const ColiV41AttentionSnapshot *snapshot,
                                     FILE *stream);
int coli_v41_attention_snapshot_read(FILE *stream,
                                    ColiV41AttentionSnapshot **output);
/* ==== end deepseek_v41_attention_transaction.h ==== */

/* ==== begin deepseek_v41_compressor.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v41_config.h */
/* amalgamated: deepseek_v41_layer.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV41CompressorState ColiDeepSeekV41CompressorState;

typedef struct {
    const char *prefix;
    int head_dimension;
    int rotate_fp4;
} ColiDeepSeekV41CompressorOptions;

int coli_v41_compressor_create(ColiDeepSeekV41CompressorState **state,
                              const ColiDeepSeekV41LayerWeights *weights,
                              const ColiDeepSeekV41Config *config,
                              char *error, size_t error_size);
int coli_v41_compressor_create_with_options(
    ColiDeepSeekV41CompressorState **state,
    const ColiDeepSeekV41LayerWeights *weights,
    const ColiDeepSeekV41Config *config,
    const ColiDeepSeekV41CompressorOptions *options,
    char *error, size_t error_size);
void coli_v41_compressor_reset(ColiDeepSeekV41CompressorState *state);
int coli_v41_compressor_bind_weights(ColiDeepSeekV41CompressorState *state,
                                    const ColiDeepSeekV41LayerWeights *weights,
                                    char *error, size_t error_size);
void coli_v41_compressor_destroy(ColiDeepSeekV41CompressorState *state);

/* Processes one decode token. produced is set to one only when a complete
 * compression window emits a KV vector. output may be NULL on other steps. */
int coli_v41_compressor_step(ColiDeepSeekV41CompressorState *state,
                            float *output, int *produced,
                            const float *input, int position,
                            char *error, size_t error_size);

/* Like coli_v41_compressor_step, but consumes precomputed wkv/wgate matvec
 * rows (projection_dim floats each, e.g. batched on the GPU) instead of
 * projecting input itself. State updates are identical. */
int coli_v41_compressor_advance(ColiDeepSeekV41CompressorState *state,
                               float *output, int *produced,
                               const float *kv_proj, const float *gate_proj,
                               int position, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_compressor.h ==== */

/* ==== begin deepseek_v41_compressor_snapshot.h ==== */

/* amalgamated: deepseek_v41_compressor.h */

typedef struct ColiV41CompressorSnapshot ColiV41CompressorSnapshot;

int coli_v41_compressor_snapshot_create(
    const ColiDeepSeekV41CompressorState *state,
    ColiV41CompressorSnapshot **output);
int coli_v41_compressor_snapshot_restore(
    ColiDeepSeekV41CompressorState *state,
    const ColiV41CompressorSnapshot *snapshot);
void coli_v41_compressor_snapshot_destroy(ColiV41CompressorSnapshot *snapshot);
int coli_v41_compressor_snapshot_write(const ColiV41CompressorSnapshot *snapshot,
                                      FILE *stream);
int coli_v41_compressor_snapshot_read(FILE *stream,
                                     ColiV41CompressorSnapshot **output);
/* ==== end deepseek_v41_compressor_snapshot.h ==== */

/* ==== begin deepseek_v41_indexer.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v41_config.h */
/* amalgamated: deepseek_v41_layer.h */

typedef struct ColiDeepSeekV41Indexer ColiDeepSeekV41Indexer;

int coli_v41_indexer_create(ColiDeepSeekV41Indexer **state,
                           const ColiDeepSeekV41LayerWeights *weights,
                           const ColiDeepSeekV41Config *config,
                           int max_context, char *error, size_t error_size);
int coli_v41_indexer_bind_weights(ColiDeepSeekV41Indexer *state,
                                 const ColiDeepSeekV41LayerWeights *weights,
                                 char *error, size_t error_size);
void coli_v41_indexer_reset(ColiDeepSeekV41Indexer *state);
void coli_v41_indexer_destroy(ColiDeepSeekV41Indexer *state);

/* Updates the overlap compressor, then returns compressed-cache ordinals in
 * descending index score order. query_rank is the normalized q_lora vector. */
int coli_v41_indexer_step(ColiDeepSeekV41Indexer *state, int *indices,
                         int index_capacity, const float *query_rank,
                         const float *input, int position,
                         char *error, size_t error_size);
/* Like coli_v41_indexer_step, but consumes precomputed rows of the indexer
 * compressor's wkv/wgate projections for this position. */
int coli_v41_indexer_step_projected(ColiDeepSeekV41Indexer *state, int *indices,
                                   int index_capacity, const float *query_rank,
                                   const float *input, int position,
                                   const float *kv_proj, const float *gate_proj,
                                   char *error, size_t error_size);
/* Batched prefill split of the step: advance per token (in order), then
 * select for the whole chunk at once. */
int coli_v41_indexer_advance(ColiDeepSeekV41Indexer *state, const float *input,
                            int position, const float *kv_proj,
                            const float *gate_proj, char *error,
                            size_t error_size);
int coli_v41_indexer_select_batch(ColiDeepSeekV41Indexer *state, int *indices,
                                 int index_capacity, const float *query_ranks,
                                 const float *inputs, int start_position,
                                 int batch, const int *counts, int *selected,
                                 char *error, size_t error_size);
const float *coli_v41_indexer_compressed_values(
    const ColiDeepSeekV41Indexer *state);
int coli_v41_indexer_compressed_count(const ColiDeepSeekV41Indexer *state);
/* ==== end deepseek_v41_indexer.h ==== */

/* ==== begin deepseek_v41_indexer_snapshot.h ==== */

/* amalgamated: deepseek_v41_indexer.h */

typedef struct ColiV41IndexerSnapshot ColiV41IndexerSnapshot;

int coli_v41_indexer_snapshot_create(const ColiDeepSeekV41Indexer *state,
                                    ColiV41IndexerSnapshot **output);
int coli_v41_indexer_snapshot_restore(ColiDeepSeekV41Indexer *state,
                                     const ColiV41IndexerSnapshot *snapshot);
void coli_v41_indexer_snapshot_destroy(ColiV41IndexerSnapshot *snapshot);
int coli_v41_indexer_snapshot_write(const ColiV41IndexerSnapshot *snapshot,
                                   FILE *stream);
int coli_v41_indexer_snapshot_read(FILE *stream, ColiV41IndexerSnapshot **output);
/* ==== end deepseek_v41_indexer_snapshot.h ==== */

/* ==== begin deepseek_v41_expert.h ==== */

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v41_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit);

/* Batch-major routed-expert forward.  The row-major FP4 path streams each
 * matrix once for all items; unsupported packed layouts fall back to the
 * scalar entry point without changing its numerical contract. */
int coli_v41_expert_forward_batch_ref(float *outputs,
                                     const ColiExpertView *expert,
                                     const float *inputs,
                                     const float *route_weights,
                                     int batch, float swiglu_limit);

int coli_v41_shared_expert_forward_ref(float *output,
                                      const ColiTensorView *gate,
                                      const ColiTensorView *down,
                                      const ColiTensorView *up,
                                      const float *input,
                                      float swiglu_limit);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_expert.h ==== */

/* ==== begin deepseek_v41_expert_store.h ==== */

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV41ExpertStoreOptions {
    const char *model_dir;
    int layers;
    int experts_per_layer;
    uint64_t cache_bytes;
    /* Optional hot-pin policy (-1 / 0 => implementation default). */
    int pin_slots_per_layer;
    uint64_t repin_interval;
    /* Internal range executors must not benchmark arbitrary bytes from a
     * complete checkpoint while opening one layer slice. */
    int skip_mirror_setup;
} ColiDeepSeekV41ExpertStoreOptions;

int coli_deepseek_v41_expert_store_open(
    const ColiDeepSeekV41ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size);

/* Plain SSD implementation underneath the optional hot-row/autopin wrapper.
 * Segment adapters use it so opening a layer range never warms experts from
 * layers outside that range.  It remains an internal symbol, not public ABI. */
int coli_deepseek_v41_expert_store_open_base(
    const ColiDeepSeekV41ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size);

/* Batched CPU prefill only: `layer` borrows the complete expert-cache pool;
 * a negative value restores ordinary per-layer miss allocation for decode.
 * Alternative registered ExpertStore backends safely ignore the request. */
void coli_v41_expert_store_prefill_pool(ColiExpertStore *store, int layer);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_expert_store.h ==== */

/* ==== begin deepseek_v41_block.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v41_config.h */
/* amalgamated: deepseek_v41_attention.h */
/* amalgamated: deepseek_v41_layer.h */
#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v41_block_token_ref(float *output_hc,
                            const ColiDeepSeekV41LayerWeights *weights,
                            const ColiDeepSeekV41Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size);
int coli_v41_block_window_token_ref(
    float *output_hc, ColiDeepSeekV41WindowAttentionState *attention,
    const ColiDeepSeekV41LayerWeights *weights,
    const ColiDeepSeekV41Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v41_block.h ==== */

/* ==== begin deepseek_v41_block_batch.h ==== */

/* amalgamated: deepseek_v41_attention.h */
/* amalgamated: deepseek_v41_config.h */
/* amalgamated: deepseek_v41_layer.h */
#include "expert_store.h"

int coli_v41_block_window_batch_ref(
    float *outputs_hc, ColiDeepSeekV41WindowAttentionState *attention,
    const ColiDeepSeekV41LayerWeights *weights,
    const ColiDeepSeekV41Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size);
/* ==== end deepseek_v41_block_batch.h ==== */

/* ==== begin deepseek_v41_resource_plan.h ==== */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t available_bytes;
    uint64_t user_limit_bytes;
    uint64_t maximum_layer_bytes;
    uint64_t runtime_other_bytes;
    uint64_t expert_record_bytes;
    int sparse_layers;
    int routed_topk;
    int experts_per_layer;
} ColiDeepSeekV41ResourceInputs;

typedef struct {
    uint64_t os_available_bytes;
    uint64_t planner_available_bytes;
    uint64_t system_reserve_bytes;
    uint64_t runtime_reserve_bytes;
    uint64_t minimum_expert_bytes;
    uint64_t expert_cache_bytes;
    uint64_t projected_bytes;
    int slots_per_layer;
} ColiDeepSeekV41ResourcePlan;

typedef struct {
    uint64_t available_bytes;
    uint64_t fixed_bytes;
    uint64_t dense_bytes;
    uint64_t minimum_expert_bytes;
} ColiDeepSeekV41ResidentTierInputs;

typedef struct {
    uint64_t dense_bytes;
    int dense_resident;
} ColiDeepSeekV41ResidentTierPlan;

uint64_t coli_v41_os_available_memory(void);
int coli_v41_resource_plan_compute(
    ColiDeepSeekV41ResourcePlan *plan,
    const ColiDeepSeekV41ResourceInputs *inputs,
    char *error, size_t error_size);
int coli_v41_resident_tier_plan(
    ColiDeepSeekV41ResidentTierPlan *plan,
    const ColiDeepSeekV41ResidentTierInputs *inputs,
    char *error, size_t error_size);
/* ==== end deepseek_v41_resource_plan.h ==== */

/* ==== begin deepseek_v41_head_cache.h ==== */

#include <stddef.h>
#include <stdint.h>

int coli_v41_head_cache_probe(const ColiSafetensorsIndex *index, uint64_t *bytes,
                             char *error, size_t error_size);
int coli_v41_head_cache_load(ColiV41Engine *engine,
                            const ColiSafetensorsIndex *index,
                            char *error, size_t error_size);
uint64_t coli_v41_head_cache_bytes(const ColiV41Engine *engine);
const void *coli_v41_head_cache_data(const ColiV41Engine *engine,
                                    int shard, uint64_t offset, size_t length);
/* ==== end deepseek_v41_head_cache.h ==== */


/* Runtime options live on ColiV41Engine. */
typedef struct {
    const char *target_model_dir;
    uint64_t memory_limit_bytes;
    int context_tokens;
    int dense_resident;
    uint64_t target_expert_cache_bytes;
    int pin_slots_per_layer;
    uint64_t repin_interval;
    uint64_t dspark_reserve_bytes;
} ColiDeepSeekV41RuntimeOptions;

enum { COLI_V41_RESIDENT_MAX_LAYERS = 128 };

/* engine open: seconds spent building target_index (printed by the auto
 * store planner as the v41_open line). Defined in the engine unit. */
extern double g_v41_open_index_seconds;

#ifdef COLI_V4_GPU_TIER
/* Provided by the COLI_V41_UNIT_GPU translation unit. Compiled in only on the
 * Windows CUDA build; every call site elsewhere is guarded by COLI_V4_GPU_TIER
 * so non-GPU objects never reference these symbols. */
int coli_v41_gpu_engine_open(ColiV41Engine *engine);
void coli_v41_gpu_engine_close(ColiV41Engine *engine);
int coli_v41_gpu_layer_upload(ColiV41Engine *engine, int layer,
                             ColiDeepSeekV41LayerWeights *weights);
int coli_v4_gpu_fp8_matvec(const ColiTensorView *w, float *output,
                           const float *input);
int coli_v4_gpu_fp8_matmul_batch(const ColiTensorView *w, float *outputs,
                                 const float *inputs, int batch);
int coli_v41_gpu_kv_ring_append(const ColiDeepSeekV41LayerWeights *weights,
                               const float *rows, int start_pos, int count,
                               int window, int dim);
int coli_v41_gpu_kv_comp_append(const ColiDeepSeekV41LayerWeights *weights,
                               const float *rows, int start_idx, int count,
                               int dim);
int coli_v41_gpu_kv_cache_sync(const ColiDeepSeekV41LayerWeights *weights,
                              const float *cpu_ring, int window, int head_dim,
                              int start_position, const float *compressed,
                              int comp_total);
void coli_v41_gpu_kv_cache_advance(const ColiDeepSeekV41LayerWeights *weights,
                                  const float *rows, int start_position,
                                  int count, int window, int head_dim,
                                  int comp_total);
void coli_v41_gpu_kv_cache_poison(const ColiDeepSeekV41LayerWeights *weights);
void coli_v41_gpu_kv_cache_invalidate_all(void);
int coli_v41_gpu_sparse_attention_batch_cached(
    const ColiDeepSeekV41LayerWeights *weights, float *attended,
    const float *q, const float *chunk, int chunk_start, const float *sinks,
    const int *meta, int abs_base, int comp_limit, int heads, int head_dim,
    int batch);
int coli_v41_gpu_fp8_ref_matmul(const ColiDeepSeekV41LayerWeights *weights,
                               const ColiTensorView *w, const float *x_qdq,
                               int tokens, float *y);
int coli_v41_gpu_indexer_score_batch(
    const ColiDeepSeekV41LayerWeights *weights, float *scores,
    const float *queries, const float *keys, const float *head_w,
    const int *counts, int tokens, int heads, int dim, int count);
int coli_v41_gpu_sparse_attention_batch_cached_idx(
    const ColiDeepSeekV41LayerWeights *weights, float *attended,
    const float *q, const float *chunk, int chunk_start, const float *sinks,
    const int *meta, const int *sel, int selstride, int abs_base,
    int comp_limit, int heads, int head_dim, int batch);
int coli_v41_gpu_moe_batch_wanted(void);
void coli_v41_gpu_moe_batch_release(void);
void coli_v41_gpu_moe_batch_hint(int total_fresh_tokens);
int coli_v41_gpu_moe_batch_union(float *outputs,
                                const ColiDeepSeekV41LayerWeights *weights,
                                const ColiDeepSeekV41Config *config,
                                ColiExpertStore *store,
                                const float *inputs, const int *tokens,
                                int batch);
int coli_v4_gpu_matvec_grouped(const ColiTensorView *w, float *output,
                               const float *input, int groups);
/* Batched GPU attention offloads for prefill (COLI_CUDA_ATTN_BATCH=1).
 * Every entry returns non-zero on any refusal so the caller can fall back to
 * the CPU reference for the whole chunk. */
int coli_v41_gpu_attn_batch_wanted(void);
/* Runs both bf16 projection matrices (wkv_key/wgate_key mirrors) over the
 * whole chunk: kv_proj/gate_proj receive [batch][rows-of-mirror]. */
int coli_v41_gpu_compressor_project_batch(
    const ColiDeepSeekV41LayerWeights *weights, const char *wkv_key,
    const char *wgate_key, int expected_rows, float *kv_proj,
    float *gate_proj, const float *inputs, int batch);
/* Sparse window attention over a linear KV slab; contract described at
 * dsv4_cuda_sparse_attn_batch. */
int coli_v41_gpu_sparse_attention_batch(
    const ColiDeepSeekV41LayerWeights *weights, float *attended,
    const float *q, const float *values, const float *sinks, const int *meta,
    int value_rows, int comp_base, int heads, int head_dim, int batch);
/* Grouped wo_a + wo_b over the whole chunk through the fp8-bf16 wo_a mirror.
 * q_width = heads*head_dim (context row), hidden = output row. */
int coli_v41_gpu_attention_wo_batch(
    const ColiDeepSeekV41LayerWeights *weights, float *outputs,
    const float *attended, int groups, int q_width, int hidden, int batch);
/* Batched mHC: whole-chunk normalized_hc_pre / coli_v41_hc_post through the
 * hc_<branch>_fn/scale/base and norm f32 mirrors (hc must be 4, hidden 4096).
 * posts is [batch][hc], combs [batch][hc*hc]; layouts match the CPU arrays. */
int coli_v41_gpu_mhc_pre_norm_batch(
    const ColiDeepSeekV41LayerWeights *weights, const char *branch,
    const char *norm_key, float *posts, float *combs, float *normalized,
    const float *inputs_hc, int hc, int hidden, int batch);
int coli_v41_gpu_mhc_post_batch(
    const ColiDeepSeekV41LayerWeights *weights, float *outputs_hc,
    const float *branch, const float *residual_hc, const float *posts,
    const float *combs, int hc, int hidden, int batch);
/* MoE router offload: mirrors the bf16 route contract of coli_v41_route_bf16.
 * Uses the resident layer's uploaded f32 gate/bias mirrors through
 * dsv4_cuda_route (which hardcodes 256 experts / top-k 6); any shape or
 * mirror mismatch returns non-zero so the caller falls back to the CPU route. */
int coli_v41_gpu_route(float *route_weights, int *indices, const float *input,
                      const ColiDeepSeekV41LayerWeights *weights,
                      const float *bias, const int *forced_indices,
                      int experts, int dimension, int topk, float route_scale);
/* Best-effort fp4 mirror attach for a routed expert. Returns 0 when all three
 * of view->gate/up/down now carry Dsv4CudaTensor* handles (cached on first
 * use), non-zero when the tier is inactive or the expert must stay on the CPU
 * fp4 path. Only block_rows==1 views are mirrored; rows16-packed slots are
 * skipped. */
int coli_v41_gpu_expert_attach(ColiExpertStore *store, ColiExpertView *view);
/* lookup-only twin: reports residency, never uploads (hybrid q* split) */
int coli_v41_gpu_expert_peek(ColiExpertStore *store, ColiExpertView *view);
/* DSV4_HYBRID=1 gate plus its cross-unit counters/EMAs: defined in the block
 * unit, read by the serve unit's per-turn stderr line. */
int coli_v41_hybrid_enabled(void);
/* async attach (enqueue only; drain closes the pipeline before compute) */
int coli_v41_gpu_expert_attach_async(ColiExpertStore *store,
                                    ColiExpertView *view);
int coli_v41_gpu_expert_drain(ColiExpertStore *store);
extern double g_v41_hyb_fill_bw, g_v41_hyb_host_bw;
extern unsigned long long g_v41_hyb_gpu_n, g_v41_hyb_cpu_n;
extern unsigned long long g_v41_hyb_upload_n, g_v41_hyb_skip_n;
/* Dspark (MTP) resident-expert mirrors: a separate bounded LRU so drafting can
 * never evict the target model's learned expert mirrors. ensure() lazily
 * allocates the cache on the first V41_MTP_GPU=1 draft; attach mirrors one
 * block_rows==1 fp4 expert view into it (returns non-zero to stay on CPU). */
int coli_v41_gpu_dspark_mirrors_ensure(ColiV41Engine *engine);
int coli_v41_gpu_dspark_expert_attach(void *cache, ColiExpertView *view);
#endif

struct ColiV41Engine {
    ColiDeepSeekV41Config config;
    ColiDeepSeekV41RuntimeOptions runtime;
    ColiSafetensorsIndex *target_index;
    ColiExpertStore *experts;
    ColiV41EngineMemorySummary summary;
    struct {
        unsigned char *data;
        uint64_t bytes;
        uint64_t offset;
        int shard;
    } head_cache;
    struct {
        ColiDeepSeekV41LayerWeights layers[COLI_V41_RESIDENT_MAX_LAYERS];
        unsigned char ready[COLI_V41_RESIDENT_MAX_LAYERS];
        const ColiSafetensorsIndex *index;
        uint64_t total_bytes;
    } dense_resident;
    /* Optional CUDA tier (compiled in only when the engine build defines
     * COLI_V4_GPU_TIER on Windows). enabled is 1 only after the loader resolved
     * coli_cuda_dsv4.dll and dsv4_cuda_init succeeded; matvec dispatch then
     * short-circuits the dense fp8 projections through backend_cuda_dsv4.cu. */
    struct {
        int enabled;
        int device;
        unsigned char layer_ready[COLI_V41_RESIDENT_MAX_LAYERS];
        long long uploaded_bytes;
        /* Optional opaque V41GpuExpertMirrorCache* for the dspark/MTP draft
         * experts (separate bounded LRU; see dspark_mirrors_ensure). NULL
         * unless V41_MTP_GPU=1 and the tier opened successfully. */
        void *dspark_mirrors;
    } gpu;
    struct {
        uint16_t *markov_w1;
        uint16_t *markov_w2;
        uint64_t bytes;
        int rank;
        int block_size;
        int stage;
        int enabled;
    } dspark;
    char *owned_target_model_dir;
    int owns_experts;
    int owns_index;
    int active_sessions; /* sessions created against this engine */
};

/* Session ownership helpers shared by production session code and tests. */
void coli_v41_engine_attach_session(ColiV41Engine *engine);
void coli_v41_engine_detach_session(ColiV41Engine *engine);

#include "tok.h"
#include "kv_prefix.h"

struct ColiV41Session {
    ColiV41Engine *engine;
    ColiDeepSeekV41Config config;
    ColiDeepSeekV41WindowAttentionState **attention;
    float *state;
    float *next;
    float *hidden;
    int *prompt_ids;
    int *generated;
    int max_prompt_tokens;
    int max_new_tokens_cap;
    int prompt_count;
    int generated_count;
    Tok tokenizer;
    int tokenizer_ready;
    char *text;
    int text_length;
    /* Token ids this session's attention state already holds, prompt and
     * generated alike, in the shared format colibri.c/inkling.c/kimi_k3.c use.
     * A follow-up request whose prompt starts with exactly these ids continues
     * from that position instead of re-prefilling it. */
    kv_prefix fed;
    int prefix_reused;   /* reuse length of the request in flight, for stats */
    uint64_t spec_attempts;
    uint64_t spec_drafted;
    uint64_t spec_accepted;
    int spec_disabled;
};

/* RAM-tiered expert open used by coli_v41_engine_open (replaces ld --wrap).
 * `config` is the model geometry (== engine->config on the engine_open path);
 * it is forwarded by the backend registry so engine-less callers (the standalone
 * CLI) can use routed backends that only need geometry, not the engine. */
int coli_v41_expert_store_open_planned(
    ColiV41Engine *engine,
    const ColiDeepSeekV41Config *config,
    const ColiDeepSeekV41ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size);

/* Internal accessors — not part of the experimental public API. */
ColiSafetensorsIndex *coli_v41_engine_target_index(ColiV41Engine *engine);
ColiExpertStore *coli_v41_engine_expert_store(ColiV41Engine *engine);

/* Head-cache aware safetensors read (engine NULL => plain coli_st_read_at). */
int coli_st_read_at_engine(ColiV41Engine *engine,
                           const ColiSafetensorsIndex *index, int shard,
                           uint64_t offset, size_t length, void *destination);

#ifdef COLI_V41_TEST_HOOKS
/*
 * Fault-injection / counters for ownership tests only.
 * Compile ownership objects with -DCOLI_V41_TEST_HOOKS; production objects omit this.
 */
extern int coli_v41_test_fail_expert_store_open;
extern int coli_v41_test_skip_expert_store_open;
extern int coli_v41_test_closed_owned_index;
extern void (*coli_v41_test_expert_read_hook)(ColiExpertKey key);
extern void (*coli_v41_test_expert_wait_hook)(ColiExpertKey key);
extern uint64_t coli_v41_test_fp4_batch_calls;
extern uint64_t coli_v41_test_expert_victim_probes;
int coli_v41_test_expert_slot_index(ColiExpertStore *store, ColiExpertKey key);

ColiV41Session *coli_v41_test_session_bare_create(ColiV41Engine *engine);
void coli_v41_test_session_bare_destroy(ColiV41Session *session);
#endif /* COLI_V41_TEST_HOOKS */

#endif /* COLIBRI_DEEPSEEK_V41_INTERNAL_H */
