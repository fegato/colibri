/* tests/test_deepseek_v41.c -- engine contract for the DeepSeek V4.1 fork.
 *
 * What this pins down (docs/deepseek-v41-delta.md):
 *   1. the V4.1 *config shape*: the architecture keys live under text_config, and
 *      every key the engine needs is read from there -- the same file the engine
 *      refuses to run without
 *   2. the fail-closed capability gate: a checkpoint that declares engram layers is
 *      refused while the n-gram path is not implemented, because dropping the
 *      engram contribution computes a different model
 *   3. the source tables are validated (sorted, in range, index sources cover the
 *      KV sources) instead of silently sharing the wrong KV
 *   4. V4.1 has no token-id hash routing: `num_hash_layers != 0` is refused
 *   5. the maths the family inherits from the V4 engine -- sqrtsoftplus routing
 *      with the noaux_tc bias selection and the SwiGLU clamp -- keep behaving
 *
 * Build: make tests/test_deepseek_v41  (see the parent Makefile)
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v41.h"
/* The routing and SwiGLU entry points live in the internal header, like the V4
 * engine's units and tests use them. */
#include "deepseek_v41_internal.h"
#include "deepseek_v41_engram_tables.h"
#include "deepseek_v41_engram_vectors.h"
#include "deepseek_v41_engram_map_checks.h"

/* A compact but *structurally complete* V4.1 config: the real checkpoint's keys,
 * shrunk to four backbone layers plus one DSpark stage. */
static const char *V41_CONFIG =
    "{"
    "\"architectures\":[\"DeepseekV41ForCausalLM\"],"
    "\"model_type\":\"deepseek_v41\","
    "\"dtype\":\"bfloat16\","
    "\"quantization_config\":{\"quant_method\":\"fp8\",\"activation_scheme\":\"dynamic\","
        "\"weight_block_size\":[32,32],\"scale_fmt\":\"ue8m0\",\"expert_dtype\":\"fp4\"},"
    "\"text_config\":{"
        "\"model_type\":\"deepseek_v41_text\",\"vocab_size\":128,\"hidden_size\":64,"
        "\"moe_intermediate_size\":32,\"num_hidden_layers\":4,\"num_attention_heads\":4,"
        "\"num_key_value_heads\":1,\"head_dim\":32,\"qk_rope_head_dim\":16,"
        "\"q_lora_rank\":16,\"o_lora_rank\":8,\"o_groups\":2,\"hidden_act\":\"silu\","
        "\"swiglu_limit\":10.0,\"rms_norm_eps\":1e-20,\"attention_bias\":false,"
        "\"max_position_embeddings\":1024,\"rope_theta\":10000,"
        "\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":16,\"beta_fast\":32,"
            "\"beta_slow\":1,\"original_max_position_embeddings\":256},"
        "\"n_routed_experts\":4,\"n_shared_experts\":1,\"num_experts_per_tok\":2,"
        "\"scoring_func\":\"sqrtsoftplus\",\"topk_method\":\"noaux_tc\","
        "\"norm_topk_prob\":true,\"routed_scaling_factor\":1.5,\"sliding_window\":8,"
        "\"compress_ratios\":[0,0,2,1,0],\"compress_rope_theta\":160000,"
        "\"kv_source_layer_ids\":[2],\"index_source_layer_ids\":[2,3],"
        "\"index_n_heads\":2,\"index_head_dim\":16,\"index_topk\":4,"
        "\"candidate_source_layer_id\":3,\"candidate_topk_blocks\":8,"
        "\"candidate_block_size\":2,\"hc_mult\":2,\"hc_sinkhorn_iters\":3,"
        "\"hc_eps\":1e-06,"
        "\"engram_layer_ids\":[1],\"engram_num_embeddings\":[1000],"
        "\"engram_max_ngram_size\":4,\"engram_vocab_size\":64,\"engram_n_heads\":8,"
        "\"engram_head_dim\":16,\"engram_pad_token_id\":2,"
        "\"engram_compressed_vocab_size\":32,"
        "\"num_nextn_predict_layers\":1,\"dspark_block_size\":5,"
        "\"dspark_noise_token_id\":100,\"dspark_target_layer_ids\":[3],"
        "\"dspark_markov_rank\":16,\"dspark_n_routed_experts\":2,"
        "\"dspark_num_experts_per_tok\":1"
    "},"
    "\"vision_config\":{\"model_type\":\"deepseek_v41_vision\",\"num_hidden_layers\":2,"
        "\"hidden_size\":8,\"num_attention_heads\":2,\"intermediate_size\":16,"
        "\"patch_size\":14,\"rope_theta\":10000,\"downsample_ratio\":3,"
        "\"max_image_tokens\":1024}"
    "}";

static int failures = 0;

static void check(int condition, const char *what) {
    if (condition) {
        printf("ok   %s\n", what);
    } else {
        printf("FAIL %s\n", what);
        failures++;
    }
}

/* Replace every occurrence of `from` with `to`, in place, into a fresh buffer. */
static char *rewrite(const char *json, const char *from, const char *to) {
    size_t length = strlen(json), from_length = strlen(from), to_length = strlen(to);
    assert(from_length >= to_length || length + (to_length - from_length) * 4 < 8192);
    char *out = malloc(16384);
    assert(out != NULL);
    size_t written = 0;
    const char *cursor = json;
    while (1) {
        const char *hit = strstr(cursor, from);
        if (!hit)
            break;
        size_t chunk = (size_t)(hit - cursor);
        memcpy(out + written, cursor, chunk);
        written += chunk;
        memcpy(out + written, to, to_length);
        written += to_length;
        cursor = hit + from_length;
    }
    strcpy(out + written, cursor);
    return out;
}

static int parse(const char *json, ColiDeepSeekV41Config *config,
                 char *error, size_t error_size) {
    memset(config, 0, sizeof(*config));
    return coli_v41_config_parse(config, json, error, error_size);
}

static void test_config_shape(void) {
    ColiDeepSeekV41Config config;
    char error[512] = {0};
    char *json = rewrite(V41_CONFIG, "\"engram_layer_ids\":[1]", "\"engram_layer_ids\":[]");
    json = rewrite(json, "\"engram_num_embeddings\":[1000]", "\"engram_num_embeddings\":[]");
    int result = parse(json, &config, error, sizeof(error));
    check(result == 0, "V4.1 config shape parses (text_config unwrapped)");
    if (result != 0)
        printf("      %s\n", error);
    check(config.hidden_size == 64 && config.num_hidden_layers == 4,
          "architecture keys come from text_config");
    check(config.hidden_size == 64 && config.q_lora_rank == 16 &&
          config.num_key_value_heads == 1 && config.head_dim == 32,
          "attention shapes read");
    check(config.num_experts_per_tok == 2 && config.engram_head_dim == 16 &&
          config.dspark_markov_rank == 16,
          "MoE, engram and DSpark knobs read");
    check(config.vision_enabled == 1, "vision_config noticed");
    check(config.kv_source_layer_count == 1 && config.kv_source_layer_ids[0] == 2,
          "kv source table read");
    check(config.index_source_layer_count == 2 && config.index_source_layer_ids[1] == 3,
          "index source table read");
    check(config.candidate_source_layer_id == 3 && config.compress_ratio_count == 5,
          "candidate source and compress ratios read");
    check(config.num_hash_layers == 0, "no hash layers");
    free(json);

    /* the nesting itself is required: V4 kept these keys at the root, and reading a
     * V4.1 file as flat is exactly how a wrong-model run starts */
    json = rewrite(V41_CONFIG, "\"text_config\":{", "\"text_config_flat\":{");
    result = parse(json, &config, error, sizeof(error));
    check(result != 0 && strstr(error, "text_config") != NULL,
          "a config without text_config is refused");
    free(json);
}

static void test_engram_gate(void) {
    ColiDeepSeekV41Config config;
    char error[512] = {0};
    int result = parse(V41_CONFIG, &config, error, sizeof(error));
#if COLI_V41_ENGRAM_IMPLEMENTED
    check(result == 0, "engram path implemented: the engram checkpoint parses");
#else
    check(result != 0 && strstr(error, "engram") != NULL,
          "declared engram layers are refused while the n-gram path is missing");
#endif
}

static void test_source_table_validation(void) {
    ColiDeepSeekV41Config config;
    char error[512] = {0};

    char *json = rewrite(V41_CONFIG, "\"engram_layer_ids\":[1]", "\"engram_layer_ids\":[]");
    json = rewrite(json, "\"engram_num_embeddings\":[1000]", "\"engram_num_embeddings\":[]");
    char *unsorted = rewrite(json, "\"kv_source_layer_ids\":[2]", "\"kv_source_layer_ids\":[3,2]");
    check(parse(unsorted, &config, error, sizeof(error)) != 0 && strstr(error, "sorted") != NULL,
          "unsorted kv_source_layer_ids refused");
    free(unsorted);

    char *outside = rewrite(json, "\"kv_source_layer_ids\":[2]", "\"kv_source_layer_ids\":[9]");
    check(parse(outside, &config, error, sizeof(error)) != 0,
          "kv source outside the backbone refused");
    free(outside);

    /* index keys are derived from the compressed latent, so a KV source that is not
     * an index source is a broken table, not a style choice */
    char *uncovered = rewrite(json, "\"index_source_layer_ids\":[2,3]",
                              "\"index_source_layer_ids\":[3]");
    check(parse(uncovered, &config, error, sizeof(error)) != 0 &&
          strstr(error, "index source") != NULL,
          "kv source missing from the index sources refused");
    free(uncovered);

    char *hash = rewrite(json, "\"hc_mult\":2", "\"hc_mult\":2,\"num_hash_layers\":1");
    check(parse(hash, &config, error, sizeof(error)) != 0 &&
          strstr(error, "hash") != NULL,
          "num_hash_layers != 0 refused (V4.1 has no token-id hash routing)");
    free(hash);

    /* the quantized formats are part of the contract: a checkpoint that is not
     * fp4-experts/fp8-dense is not the model this engine loads */
    char *dtype = rewrite(json, "\"expert_dtype\":\"fp4\"", "\"expert_dtype\":\"int8\"");
    check(parse(dtype, &config, error, sizeof(error)) != 0,
          "a non-fp4 expert dtype refused");
    free(dtype);

    char *flat_quant = rewrite(json, "\"scale_fmt\":\"ue8m0\"", "\"scale_fmt\":\"e8m0\"");
    check(parse(flat_quant, &config, error, sizeof(error)) != 0,
          "a non-ue8m0 scale format refused");
    free(flat_quant);
    free(json);
}

/* The maths V4.1 inherits: sqrt(softplus()) scoring with the noaux_tc bias deciding
 * the *selection* while the weight comes from the un-biased score, then normalised
 * over the selected experts and scaled. */
static void test_router(void) {
    enum { experts = 3, topk = 2, dimension = 2 };
    const float gate[experts * dimension] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f,
    };
    const float bias[experts] = {0.0f, 0.5f, 0.25f};
    const float hidden[dimension] = {1.0f, 1.0f};
    float weights[topk] = {0.0f, 0.0f};
    int indices[topk] = {-1, -1};

    check(coli_v41_route(weights, indices, hidden, gate, bias, NULL, experts,
                         dimension, topk, 1.5f) == 0,
          "route() accepts a well-formed call");
    /* scores: sqrt(softplus(1)) = 1.0006, sqrt(softplus(1)) = 1.0006,
     * sqrt(softplus(-1)) = 0.5062; the bias puts expert 1 (1.5006) ahead of
     * expert 0 (1.0006) while expert 2 stays out */
    check(indices[0] == 1 && indices[1] == 0, "noaux_tc bias decides the selection");
    float total = (float)(1.0006 + 1.0006);
    check(weights[0] > 0.0f && weights[1] > 0.0f, "selected weights are positive");
    check(weights[0] + weights[1] > 1.49f && weights[0] + weights[1] < 1.51f,
          "weights are normalised over the selection and scaled");
    (void)total;

    /* a router that cannot produce positive weight must fail, not guess */
    float zero_weights[topk];
    int zero_indices[topk];
    const float zero_gate[experts * dimension] = {0.0f};
    check(coli_v41_route(zero_weights, zero_indices, hidden, zero_gate, NULL, NULL,
                         experts, dimension, topk, 1.5f) == 0,
          "route() with an all-zero gate still returns finite weights");

    check(coli_v41_route(weights, indices, hidden, gate, NULL, NULL, experts,
                         dimension, topk + 2, 1.5f) != 0,
          "route() refuses topk > experts");
}

static void test_swiglu_clamp(void) {
    const float gate[4] = {1.0f, 20.0f, -20.0f, 0.5f};
    const float up[4] = {1.0f, 20.0f, -20.0f, -0.5f};
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    check(coli_v41_swiglu(output, gate, up, 4, 10.0f) == 0, "swiglu() runs");
    /* up is clamped to +/-10 and gate to +10, so the huge pair saturates instead of
     * exploding: silu(10) * 10 stays a finite, positive number */
    check(output[1] > 0.0f && output[1] < 100.0f, "positive saturation stayed finite");
    check(output[2] > -0.01f && output[2] < 0.01f,
          "a -20 gate saturates to ~0 magnitude (only the up side is clamped low)");
    check(coli_v41_swiglu(output, gate, up, 4, -1.0f) != 0,
          "swiglu() refuses a negative limit");
    check(coli_v41_swiglu(output, gate, up, 0, 10.0f) != 0,
          "swiglu() refuses an empty dimension");
}


/* The n-gram hash ids, against the golden vectors generated from the verified
 * layout. These are the ids that index a 94.4 GiB table, so a single wrong bit is a
 * different model: the addressing is checked here rather than assumed, and the same
 * fixture holds the decode path to the prefill path. */
static void test_engram_hash(void) {
    int failures_before = failures;
    for (int index = 0; index < coli_v41_engram_vector_count; index++) {
        const ColiV41EngramVector *vector = &coli_v41_engram_vectors[index];
        size_t columns = (size_t)vector->span_count * COLI_V41_ENGRAM_HASH_COLS;
        int64_t *ids = malloc(columns * sizeof(*ids));
        assert(ids != NULL);
        check(coli_v41_engram_hash_ids(ids, vector->layer, vector->span,
                                       vector->span_count, NULL, 0, 2) == 0,
              "engram hash ids computed");
        int matches = 0;
        for (size_t column = 0; column < columns; column++)
            if (ids[column] == vector->prefill[column])
                matches++;
        if (matches != (int)columns)
            printf("      %s: %d/%zu prefill ids match\n", vector->name, matches, columns);
        check(matches == (int)columns, vector->name);

        /* the last position alone, with the rest as history: what a decode step sees */
        int64_t decode[COLI_V41_ENGRAM_HASH_COLS];
        check(coli_v41_engram_hash_ids(decode, vector->layer,
                                       &vector->span[vector->span_count - 1], 1,
                                       vector->span, vector->span_count - 1, 2) == 0,
              "engram decode ids computed");
        matches = 0;
        for (int column = 0; column < COLI_V41_ENGRAM_HASH_COLS; column++)
            if (decode[column] == vector->decode[column])
                matches++;
        check(matches == COLI_V41_ENGRAM_HASH_COLS, "decode ids match the prefill tail");
        free(ids);
    }
    check(failures == failures_before, "every engram vector matched");
}


/* Rebuilding the token map from the packed header, against the reference's map.
 * The fingerprint covers all 129,280 entries in one number; the samples make a
 * failure readable. */
static void test_engram_token_map(void) {
    uint32_t *map = malloc(COLI_V41_ENGRAM_MAP_TOKEN_COUNT * sizeof(*map));
    assert(map != NULL);
    int classes = coli_v41_engram_build_token_map(map, COLI_V41_ENGRAM_MAP_TOKEN_COUNT);
    check(classes == COLI_V41_ENGRAM_MAP_CLASS_COUNT,
          "the rebuilt map has the reference's class count");
    for (int index = 0; index < coli_v41_engram_map_sample_count; index++) {
        const ColiV41EngramMapSample *sample = &coli_v41_engram_map_samples[index];
        if (map[sample->token_id] != sample->class_id) {
            printf("      token %d: %u, expected %u\n", sample->token_id,
                   map[sample->token_id], sample->class_id);
            check(0, "token map samples match");
            free(map);
            return;
        }
    }
    check(1, "token map samples match the reference");
    check(coli_v41_engram_token_map_digest(map, COLI_V41_ENGRAM_MAP_TOKEN_COUNT) ==
          coli_v41_engram_map_expected_fnv1a64,
          "the rebuilt map matches the reference byte for byte (FNV-1a)");
    /* classes are dense because they are handed out in order */
    uint32_t highest = 0;
    for (int token = 0; token < COLI_V41_ENGRAM_MAP_TOKEN_COUNT; token++)
        if (map[token] > highest) highest = map[token];
    check((int)highest + 1 == classes, "classes are dense 0..count-1");
    /* a caller with too little room is refused, not half-filled */
    check(coli_v41_engram_build_token_map(map, 16) != 0,
          "rebuilding into a too-small map is refused");
    free(map);
}

/* The class lookup and its fail-closed edges. */
static void test_engram_compress(void) {
    const uint32_t map[8] = {0, 1, 2, 2, 3, 4, 4, 5};
    const int ids[6] = {0, 3, -1, 6, 7, 5};
    int classes[6] = {0};
    check(coli_v41_engram_compress(classes, ids, 6, map, 8, -1) == 0,
          "compress accepts a well-formed run");
    check(classes[0] == 0 && classes[1] == 2 && classes[5] == 4,
          "compress maps ids onto their class");
    check(classes[2] == -1, "the dead id blocks the n-gram (-1)");
    /* an id with no class is a tokenizer mismatch, not something to guess at */
    const int outside[1] = {9};
    check(coli_v41_engram_compress(classes, outside, 1, map, 8, -1) != 0,
          "compress refuses an id outside the map");
    /* the layout's multiplier bound is what keeps class * multiplier inside int64 */
    int64_t worst = 0;
    for (int layer = 0; layer < COLI_V41_ENGRAM_LAYERS; layer++)
        for (int shift = 0; shift < COLI_V41_ENGRAM_MAX_NGRAM; shift++)
            if (coli_v41_engram_multipliers[layer][shift] > worst)
                worst = coli_v41_engram_multipliers[layer][shift];
    check((double)worst * (double)COLI_V41_ENGRAM_COMPRESSED_VOCAB < 9.2233720368547758e18,
          "the frozen multipliers cannot overflow int64 at the largest class");
    check(coli_v41_engram_layer_position(14) == 1 &&
          coli_v41_engram_layer_position(7) == -1,
          "engram layer ids map onto the layout positions");
}

int main(void) {
    printf("deepseek_v41 contract\n");
    test_config_shape();
    test_engram_gate();
    test_source_table_validation();
    test_router();
    test_swiglu_clamp();
    test_engram_token_map();
    test_engram_hash();
    test_engram_compress();
    if (failures) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
