#ifndef COLIBRI_DEEPSEEK_V41_H
#define COLIBRI_DEEPSEEK_V41_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- config ---- */

#define COLI_V41_MAX_LAYERS 128
/* V4.1 source tables: kv and index sources come from the config, as do the
 * engram and DSpark layer lists. */
#define COLI_V41_MAX_SOURCE_LAYERS 32
#define COLI_V41_MAX_ENGRAM_LAYERS 8

typedef struct {
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int head_dim;
    int q_lora_rank;
    int qk_rope_head_dim;
    int o_groups;
    int o_lora_rank;
    int sliding_window;
    int index_n_heads;
    int index_head_dim;
    int index_topk;
    int n_routed_experts;
    int num_experts_per_tok;
    int n_shared_experts;
    int moe_intermediate_size;
    int num_hash_layers;   /* V4.1 dropped token-id hash routing: must stay 0 */
    int num_key_value_heads;
    int num_nextn_predict_layers;
    int dspark_block_size;
    int dspark_noise_token_id;
    int dspark_markov_rank;
    int hc_mult;
    int hc_sinkhorn_iters;
    int vocab_size;
    int max_position_embeddings;
    int original_max_position_embeddings;
    int compress_ratio_count;
    int compress_ratios[COLI_V41_MAX_LAYERS];
    /* V41 DELTA: shared KV / index (docs/deepseek-v41-delta.md). Only the
     * kv_source layers own a compressor and the index keys; the index_source
     * layers run an indexer and publish the top-k that the layers between them
     * reuse. Compress ratio > 0 does not make a layer an owner. */
    int kv_source_layer_ids[COLI_V41_MAX_SOURCE_LAYERS];
    int kv_source_layer_count;
    int index_source_layer_ids[COLI_V41_MAX_SOURCE_LAYERS];
    int index_source_layer_count;
    int candidate_source_layer_id;      /* -1 = none */
    int candidate_topk_blocks;
    int candidate_block_size;
    /* V41 DELTA: engram n-gram tables. ~3.8e8 rows of 256 fp8 per layer:
     * memory-mapped, never resident (see the delta doc). */
    int engram_layer_ids[COLI_V41_MAX_ENGRAM_LAYERS];
    int engram_layer_count;
    int engram_num_embeddings[COLI_V41_MAX_ENGRAM_LAYERS];
    int engram_max_ngram_size;
    int engram_vocab_size;
    int engram_n_heads;
    int engram_head_dim;
    int engram_compressed_vocab_size;
    int engram_pad_token_id;
    /* V41 DELTA: DSpark predict layers. The stages are heterogeneous: stage 0
     * carries the target-state projection, the last one the markov and
     * confidence heads. */
    int dspark_target_layer_ids[COLI_V41_MAX_ENGRAM_LAYERS];
    int dspark_target_layer_count;
    int dspark_n_routed_experts;
    int dspark_num_experts_per_tok;
    int vision_enabled;
    float rms_norm_eps;
    float hc_eps;
    float routed_scaling_factor;
    float swiglu_limit;
    float rope_theta;
    float rope_factor;
    float compress_rope_theta;
    int rope_beta_fast;
    int rope_beta_slow;
} ColiDeepSeekV41Config;

int coli_v41_config_parse(ColiDeepSeekV41Config *config, const char *json,
                         char *error, size_t error_size);
int coli_v41_config_load(ColiDeepSeekV41Config *config, const char *model_dir,
                        char *error, size_t error_size);

/* ---- prompt ---- */

typedef enum {
    COLI_V41_PROMPT_CHAT,
    COLI_V41_PROMPT_THINKING,
    COLI_V41_PROMPT_RAW,
} ColiDeepSeekV41PromptMode;

int coli_v41_prompt_build(char **output, size_t *output_length,
                         const char *user_message, const char *system_message,
                         ColiDeepSeekV41PromptMode mode);

/* ---- experimental public engine API ---- */
/*
 * The API is scoped to the DeepSeek V4 engine and may change while
 * the implementation remains experimental.
 */

typedef struct ColiV41Engine ColiV41Engine;

typedef struct {
    /* Copied by coli_v41_engine_open; caller strings need not outlive the engine. */
    const char *target_model_dir;   /* required */
    uint64_t memory_limit_bytes;    /* 0 => use OS available memory */
    int context_tokens;             /* 0 => 4096 */
    int pin_slots_per_layer;        /* -1 => auto */
    uint64_t repin_interval;        /* 0 => auto */
    int no_dspark;                  /* disable speculative draft/verification */
} ColiV41EngineOpenOptions;

typedef struct {
    uint64_t projected_bytes;
    uint64_t expert_cache_bytes;
    int slots_per_layer;
    int dense_resident;
    int head_resident;
} ColiV41EngineMemorySummary;

int coli_v41_engine_open(ColiV41Engine **engine,
                        const ColiV41EngineOpenOptions *options,
                        char *error, size_t error_size);
/* Undefined if any ColiV41Session created from this engine is still alive.
 * Destroy every session before destroying the engine. */
void coli_v41_engine_destroy(ColiV41Engine *engine);

const ColiDeepSeekV41Config *coli_v41_engine_config(const ColiV41Engine *engine);
void coli_v41_engine_memory_summary(const ColiV41Engine *engine,
                                   ColiV41EngineMemorySummary *summary);

const char *coli_v41_engine_target_model_dir(const ColiV41Engine *engine);

/* ---- experimental public session API ----
 * Session borrows the engine; destroy all sessions before coli_v41_engine_destroy.
 */

typedef struct ColiV41Session ColiV41Session;

typedef struct {
    int max_prompt_tokens;   /* 0 => 512 */
    int max_new_tokens_cap;  /* 0 => 512 */
} ColiV41SessionCreateOptions;

/* Return non-zero to abort generation. Polled between prefill chunks, where
 * the per-token callback cannot fire; a NULL callback keeps prefill
 * uninterruptible as before. */
typedef int (*ColiV41SessionAbortFn)(void *user_data);

typedef struct {
    int max_new_tokens;      /* required; clamped by session cap */
    int stop_at_sentence;
    int no_dspark;           /* disable speculative draft/verification */
    ColiV41SessionAbortFn should_abort;  /* optional prefill abort poll */
    void *abort_user_data;
    /* Optional: byte length of the prompt's stable leading prefix (the
     * rendered system turn). The session snapshots the attention state at
     * that token boundary during this prefill so later conversations that
     * share it start there. 0 = unknown. */
    size_t prefix_bytes;
} ColiV41SessionGenerateOptions;

typedef struct {
    int prompt_tokens;
    int generated_tokens;
    int eos_stopped;
    double time_to_first_token_sec;
    double decode_sec;
    uint64_t speculative_drafted;
    uint64_t speculative_accepted;
} ColiV41SessionGenerateStats;

/* Return non-zero to stop generation. */
typedef int (*ColiV41SessionTokenFn)(void *user_data, int token, float logit,
                                    int position, int ordinal);

int coli_v41_session_create(ColiV41Session **session, ColiV41Engine *engine,
                           const ColiV41SessionCreateOptions *options,
                           char *error, size_t error_size);
void coli_v41_session_destroy(ColiV41Session *session);

int coli_v41_session_generate(ColiV41Session *session,
                             const char *prompt, size_t prompt_length,
                             const ColiV41SessionGenerateOptions *options,
                             ColiV41SessionTokenFn on_token, void *user_data,
                             ColiV41SessionGenerateStats *stats,
                             char *error, size_t error_size);

int coli_v41_session_generated_text(const ColiV41Session *session,
                                   char *buffer, size_t buffer_size,
                                   size_t *out_length);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_DEEPSEEK_V41_H */
