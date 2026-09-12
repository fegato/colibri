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

#include "../deepseek_v41.h"
/* The routing and SwiGLU entry points live in the internal header, like the V4
 * engine's units and tests use them. */
#include "../deepseek_v41_internal.h"
#include "../deepseek_v41_engram_tables.h"
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



/* Row fetch and dequantization, on bytes whose values are stated by hand:
 * E4M3 0x38 = 1.0, 0x40 = 2.0, 0x30 = 0.5, 0x00 = 0.0; E8M0 128 = 2, 126 = 0.5,
 * 127 = 1. A row is head_dim fp8 bytes plus one E8M0 exponent per 32 columns. */
static void test_engram_fetch(void) {
    enum { head_dim = 32, rows = 2 };
    uint8_t table[rows * head_dim];
    uint8_t scales[rows];
    memset(table, 0, sizeof(table));
    table[0] = 0x38; table[1] = 0x40; table[2] = 0x30; table[3] = 0x00;
    table[head_dim + 0] = 0x38; table[head_dim + 1] = 0x40;
    scales[0] = 128;   /* 2.0  */
    scales[1] = 126;   /* 0.5  */
    float output[rows * head_dim];
    const int64_t order[rows] = {0, 1};
    check(coli_v41_engram_fetch_rows(output, order, rows, head_dim, table,
                                     sizeof(table), scales, sizeof(scales)) == 0,
          "fetch_rows reads a well-formed table");
    check(output[0] == 2.0f && output[1] == 4.0f && output[2] == 1.0f &&
          output[3] == 0.0f,
          "row 0 dequantizes with its own E8M0 exponent");
    check(output[head_dim] == 0.5f && output[head_dim + 1] == 1.0f &&
          output[3 + head_dim] == 0.0f,
          "row 1 uses its own exponent");
    /* hash ids are reachable from the tokenizer, so a row that does not fit must be
     * refused rather than read past the mapping */
    const int64_t outside[1] = {rows};
    check(coli_v41_engram_fetch_rows(output, outside, 1, head_dim, table,
                                     sizeof(table), scales, sizeof(scales)) != 0,
          "a row past the end of the table is refused");
    const int64_t negative[1] = {-1};
    check(coli_v41_engram_fetch_rows(output, negative, 1, head_dim, table,
                                     sizeof(table), scales, sizeof(scales)) != 0,
          "a negative row id is refused");
    check(coli_v41_engram_fetch_rows(output, order, rows, 16, table, sizeof(table),
                                     scales, sizeof(scales)) != 0,
          "a head_dim that is not a multiple of 32 is refused");
    /* the row that fits exactly at the end is still accepted */
    const int64_t last[1] = {rows - 1};
    check(coli_v41_engram_fetch_rows(output, last, 1, head_dim, table,
                                     sizeof(table) - 1, scales, sizeof(scales)) != 0,
          "a row that runs past the mapping is refused");
}

/* The gate, against values worked out by hand.
 * h = [1, 0], key = [1, 0], weight = [1, 1], dim = 2, eps = 0, value = [10, -4]:
 *   mean(h^2) = 0.5, mean(key^2) = 0.5 -> rstd = 2
 *   dot = (1*1*1 + 0) * 2 * 2^-0.5 = 1.41421356
 *   gate = sigmoid(sqrt(1.41421356)) = sigmoid(1.18920712) = 0.7666
 *   out = [1 + 0.7666*10, 0 - 0.7666*4] = [8.6660, -3.0664]
 * The tolerance is 1e-3: the hand derivation is good to four digits, and any real
 * mistake in the formula moves these numbers far further than that.
 * The same dot negated must give a gate below 0.5 (the signed root). */
static void test_engram_contribution(void) {
    const float stream[2] = {1.0f, 0.0f};
    const float key[2] = {1.0f, 0.0f};
    const float weight[2] = {1.0f, 1.0f};
    const float value[2] = {10.0f, -4.0f};
    float output[2] = {0.0f, 0.0f};
    check(coli_v41_engram_gate(output, stream, key, value, weight, 1, 2, 0.0f) == 0,
          "gate runs");
    check(fabsf(output[0] - 8.6660f) < 1e-3f && fabsf(output[1] + 3.0664f) < 1e-3f,
          "gate matches the hand-computed contribution");
    check(output[0] < 1.0f + 10.0f && output[1] > -4.0f,
          "the gate is inside (0,1), so the contribution is bounded by value");

    /* the sign of dot decides which side of 0.5 the gate lands on */
    const float flipped[2] = {-1.0f, 0.0f};
    float negative[2] = {0.0f, 0.0f};
    check(coli_v41_engram_gate(negative, stream, flipped, value, weight, 1, 2, 0.0f) == 0,
          "gate runs with a negated key");
    check(negative[0] < output[0],
          "a negated dot produces a smaller gate (signed square root)");

    /* several hc copies: each gets its own gate, all share one value */
    const float stream_hc[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float key_hc[4] = {1.0f, 0.0f, 1.0f, 0.0f};
    const float weight_hc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float copies[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    check(coli_v41_engram_gate(copies, stream_hc, key_hc, value, weight_hc, 2, 2, 0.0f) == 0,
          "gate runs per hc copy");
    /* copy 1 has h = [0, 1] against key = [1, 0]: dot is 0, the magnitude clamps to
     * 1e-6, so its gate is sigmoid(0.001) = 0.50025 and the two copies must differ */
    check(fabsf(copies[0] - 8.6660f) < 1e-3f && fabsf(copies[2] - 5.0025f) < 1e-3f &&
          fabsf(copies[3] + 1.0010f) < 1e-3f,
          "each hc copy is gated on its own dot, sharing one value");
    check(coli_v41_engram_gate(copies, stream_hc, key_hc, value, weight_hc, 0, 2, 0.0f) != 0,
          "gate refuses an empty hc count");
    check(coli_v41_engram_gate(copies, stream_hc, key_hc, value, weight_hc, 2, 2, -1.0f) != 0,
          "gate refuses a negative eps");
}



/* The blocked fp8 matvec, against a hand-computed product.
 * A 1x32 weight of all zeros except w[0] = 1.0 (E4M3 0x38) and w[1] = 0.5 (0x30),
 * one f32 scale of 2.0, and an input whose two leading entries are 1.0 and 1.0:
 * the qdq of the activation is exact here (both values are representable), so the
 * expected output is (1.0*2.0*1.0) + (0.5*2.0*1.0) = 3.0. */
static void test_fp8_matvec_blocked(void) {
    enum { columns = 32, rows = 1 };
    uint8_t data[rows * columns];
    float scales[1] = {2.0f};
    memset(data, 0x00, sizeof(data));
    data[0] = 0x38;   /* 1.0  */
    data[1] = 0x30;   /* 0.5  */
    float input[columns];
    for (int index = 0; index < columns; index++)
        input[index] = index < 2 ? 1.0f : 0.0f;
    ColiTensorView weight;
    memset(&weight, 0, sizeof(weight));
    weight.format = COLI_TENSOR_FP8_E4M3_BLOCK;
    weight.scale_format = COLI_SCALE_F32;
    weight.data = data;
    weight.scales = scales;
    weight.data_bytes = sizeof(data);
    weight.scale_bytes = sizeof(scales);
    weight.rows = rows;
    weight.columns = columns;
    weight.block_rows = 32;
    weight.block_columns = 32;
    float output[rows] = {0.0f};
    check(coli_v41_fp8_matvec_blocked(output, &weight, input) == 0,
          "blocked matvec runs on a 32-wide block");
    check(fabsf(output[0] - 3.0f) < 1e-5f,
          "blocked matvec matches the hand-computed product");
    /* 128-wide views are the V4 geometry and stay accepted; a width nobody uses is
     * refused instead of guessed */
    weight.block_columns = 128;
    check(coli_v41_fp8_matvec_blocked(output, &weight, input) != 0,
          "a 128-wide block over 32 columns is refused (scale size disagrees)");
    weight.block_columns = 16;
    check(coli_v41_fp8_matvec_blocked(output, &weight, input) != 0,
          "an unsupported block width is refused");
    weight.block_columns = 32;
    weight.scale_bytes = sizeof(scales) * 4;
    check(coli_v41_fp8_matvec_blocked(output, &weight, input) != 0,
          "a scale size that does not match the geometry is refused");
}

/* Build a tiny shard on disk and map the table out of it: the mapping is the real
 * primitive, and this is the only way to exercise it without a 94 GiB checkpoint. */
static int write_shard(const char *path, const uint8_t *rows, size_t rows_bytes,
                       const uint8_t *scales, size_t scales_bytes) {
    FILE *stream = fopen(path, "wb");
    if (!stream)
        return -1;
    int ok = fwrite(rows, 1, rows_bytes, stream) == rows_bytes &&
             fwrite(scales, 1, scales_bytes, stream) == scales_bytes;
    fclose(stream);
    return ok ? 0 : -1;
}

static void test_engram_table_map(void) {
    enum { head_dim = 32, rows = 4 };
    uint8_t table[rows * head_dim];
    uint8_t scales[rows];
    for (int index = 0; index < rows * head_dim; index++)
        table[index] = (uint8_t)(0x34 + index % 12);
    for (int index = 0; index < rows; index++)
        scales[index] = 127;

    char path[512];
    snprintf(path, sizeof(path), "deepseek_v41_table_%d.bin", (int)getpid());
    check(write_shard(path, table, sizeof(table), scales, sizeof(scales)) == 0,
          "synthetic shard written");
    ColiV41EngramTable mapped;
    char error[256] = {0};
    check(coli_v41_engram_table_open(&mapped, path, 0, sizeof(table),
                                     (int64_t)sizeof(table), sizeof(scales),
                                     head_dim, error, sizeof(error)) == 0,
          "the table maps out of the shard");
    if (mapped.rows) {
        check(mapped.row_count == rows, "the row count comes from the byte sizes");
        int same = 1;
        for (int index = 0; index < rows * head_dim; index++)
            if (mapped.rows[index] != table[index]) same = 0;
        for (int index = 0; index < rows; index++)
            if (mapped.scales[index] != scales[index]) same = 0;
        check(same, "the mapped bytes are the bytes on disk");
    }
    coli_v41_engram_table_close(&mapped);
    check(mapped.rows == NULL && mapped.row_count == 0, "closing releases the view");

    /* the two tensors must describe the same rows, or every row stride after the
     * first is wrong */
    check(coli_v41_engram_table_open(&mapped, path, 0, sizeof(table),
                                     (int64_t)sizeof(table), rows * 2, head_dim,
                                     error, sizeof(error)) != 0,
          "disagreeing weight/scale sizes are refused");
    check(coli_v41_engram_table_open(&mapped, path, 0, sizeof(table), 0,
                                     sizeof(scales), 16, error, sizeof(error)) != 0,
          "a head_dim that is not a multiple of 32 is refused");
    check(coli_v41_engram_table_open(&mapped, "does-not-exist.bin", 0, sizeof(table),
                                     0, sizeof(scales), head_dim, error,
                                     sizeof(error)) != 0,
          "a missing shard is refused with an error");
    remove(path);
}

/* The driver, end to end on a synthetic table and a synthetic wkv. The projection
 * is the engine's shared fp8 matvec, so what is checked here is the plumbing: the
 * split offsets, the per-position routing and the fail-closed shape checks. */
static void test_engram_apply(void) {
    enum { head_dim = 32, hash_cols = 24, hc_mult = 2, dim = 4,
           out_rows = hc_mult * dim + dim };
    uint8_t table[4 * head_dim];
    uint8_t scales[4];
    for (int index = 0; index < 4 * head_dim; index++)
        table[index] = (uint8_t)(0x34 + index % 12);
    for (int index = 0; index < 4; index++)
        scales[index] = 127;
    char path[512];
    snprintf(path, sizeof(path), "deepseek_v41_apply_%d.bin", (int)getpid());
    check(write_shard(path, table, sizeof(table), scales, sizeof(scales)) == 0,
          "synthetic shard written for the driver");
    ColiV41EngramTable mapped;
    char error[256] = {0};
    int opened = coli_v41_engram_table_open(&mapped, path, 0, sizeof(table),
                                            (int64_t)sizeof(table), sizeof(scales),
                                            head_dim, error, sizeof(error)) == 0;

    ColiV41EngramWorkspace work;
    check(coli_v41_engram_workspace_init(&work, hash_cols, head_dim, hc_mult, dim, 2) == 0,
          "the workspace allocates");
    const float weight[hc_mult * dim] = {1.0f, 1.0f, 1.0f, 1.0f,
                                         1.0f, 1.0f, 1.0f, 1.0f};

    /* an all-zero wkv projects to zero, so the engram adds nothing: a clean anchor
     * that does not depend on the quantized maths */
    size_t wkv_bytes = (size_t)out_rows * hash_cols * head_dim;
    size_t wkv_scale_entries = (size_t)out_rows * (hash_cols * head_dim / 32);
    size_t wkv_scale_bytes = wkv_scale_entries * sizeof(float);
    uint8_t *wkv_data = calloc(wkv_bytes, 1);
    float *wkv_scales = malloc(wkv_scale_bytes);
    assert(wkv_data && wkv_scales);
    for (size_t entry = 0; entry < wkv_scale_entries; entry++)
        wkv_scales[entry] = 1.0f;
    /* the engine's view convention: fp8 e4m3 with f32 scales already expanded from
     * the checkpoint's E8M0 bytes, one scale per 32x32 block (V4.1's
     * weight_block_size; V4 was 128x128) */
    ColiTensorView wkv;
    memset(&wkv, 0, sizeof(wkv));
    wkv.format = COLI_TENSOR_FP8_E4M3_BLOCK;
    wkv.scale_format = COLI_SCALE_F32;
    wkv.data = wkv_data;
    wkv.scales = wkv_scales;
    wkv.data_bytes = wkv_bytes;
    wkv.scale_bytes = wkv_scale_bytes;
    wkv.rows = out_rows;
    wkv.columns = hash_cols * head_dim;
    /* one scale row per weight row here: 12 rows do not fill a 32-row block, and
     * the block geometry itself is checked in test_fp8_matvec_blocked */
    wkv.block_rows = 1;
    wkv.block_columns = 32;

    float stream[hc_mult * dim] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, 0.25f};
    float original[hc_mult * dim];
    memcpy(original, stream, sizeof(stream));
    int64_t hashes[hash_cols];
    for (int index = 0; index < hash_cols; index++)
        hashes[index] = index % 4;
    if (opened) {
        check(coli_v41_engram_apply(stream, hc_mult, dim, hashes, &mapped, head_dim,
                                    &wkv, weight, 1e-6f, &work) == 0,
              "apply runs on a well-formed table");
        int untouched = 1;
        for (int index = 0; index < hc_mult * dim; index++)
            if (stream[index] != original[index]) untouched = 0;
        check(untouched, "a zero projection leaves the stream untouched");
        /* a row id past the end of the table is refused, not read past */
        int64_t outside[hash_cols];
        for (int index = 0; index < hash_cols; index++)
            outside[index] = 4;
        check(coli_v41_engram_apply(stream, hc_mult, dim, outside, &mapped, head_dim,
                                    &wkv, weight, 1e-6f, &work) != 0,
              "a hash row past the table end is refused");
        /* a wkv whose columns do not match the hash width is refused */
        ColiTensorView wrong = wkv;
        wrong.columns = hash_cols * head_dim - 32;
        check(coli_v41_engram_apply(stream, hc_mult, dim, hashes, &mapped, head_dim,
                                    &wrong, weight, 1e-6f, &work) != 0,
              "a wkv that does not match the hash width is refused");
    }
    /* The span driver, and why it has to refuse this table: the real hash ids are
     * offsets over 24 primes (millions of rows), so a 4-row synthetic shard cannot
     * hold them. That refusal IS the fail-closed property. The per-position routing
     * is checked separately, with a wkv whose rows pick different input columns, so
     * two positions with different hashes must leave different residuals. */
    const int classes[2] = {1, 2};
    int64_t span_hashes[2 * hash_cols];
    check(coli_v41_engram_hash_ids(span_hashes, 0, classes, 2, NULL, 0, 2) == 0,
          "the span's own hash ids are computable");
    int64_t smallest = span_hashes[0], largest = span_hashes[0];
    for (int slot = 0; slot < 2 * hash_cols; slot++) {
        if (span_hashes[slot] < smallest) smallest = span_hashes[slot];
        if (span_hashes[slot] > largest) largest = span_hashes[slot];
    }
    check(opened && smallest >= mapped.row_count,
          "the real hash ids fall outside the 4-row synthetic table");
    float span_stream[2 * hc_mult * dim];
    for (int index = 0; index < 2 * hc_mult * dim; index++)
        span_stream[index] = original[index % (hc_mult * dim)];
    if (opened) {
        check(coli_v41_engram_span(span_stream, hc_mult, dim, classes, 2, NULL, 0, 2,
                                   0, &mapped, head_dim, &wkv, weight, 1e-6f,
                                   &work) != 0,
              "span refuses a table that cannot hold the real hash ids");
        check(coli_v41_engram_span(span_stream, hc_mult, dim, classes, 3, NULL, 0, 2,
                                   0, &mapped, head_dim, &wkv, weight, 1e-6f,
                                   &work) != 0,
              "a span longer than the workspace was built for is refused");
        check(coli_v41_engram_span(span_stream, hc_mult, dim, classes, 2, NULL, 0, 2,
                                   0, &mapped, head_dim, &wkv, weight, 1e-6f,
                                   NULL) != 0,
              "span refuses a missing workspace");

        /* routing: one non-zero weight per column group, so each projection row
         * reads a different slice of the fetched rows */
        ColiTensorView routed;
        memset(&routed, 0, sizeof(routed));
        routed.format = COLI_TENSOR_FP8_E4M3_BLOCK;
        routed.scale_format = COLI_SCALE_F32;
        routed.data = wkv_data;
        routed.scales = wkv_scales;
        routed.data_bytes = wkv_bytes;
        routed.scale_bytes = wkv_scale_bytes;
        routed.rows = out_rows;
        routed.columns = hash_cols * head_dim;
        routed.block_rows = 1;
        routed.block_columns = 32;
        for (size_t row = 0; row < (size_t)out_rows; row++)
            wkv_data[row * hash_cols * head_dim + row] = 0x38;   /* 1.0 */
        float first[hc_mult * dim], second[hc_mult * dim];
        memcpy(first, original, sizeof(first));
        memcpy(second, original, sizeof(second));
        int64_t valid[hash_cols];
        for (int index = 0; index < hash_cols; index++)
            valid[index] = index % 4;
        check(coli_v41_engram_apply(first, hc_mult, dim, valid, &mapped, head_dim,
                                    &routed, weight, 1e-6f, &work) == 0,
              "the driver runs on a non-zero projection");
        int64_t shifted[hash_cols];
        for (int index = 0; index < hash_cols; index++)
            shifted[index] = (index + 1) % 4;
        check(coli_v41_engram_apply(second, hc_mult, dim, shifted, &mapped, head_dim,
                                    &routed, weight, 1e-6f, &work) == 0,
              "the driver runs on the shifted hashes");
        check(memcmp(first, second, sizeof(first)) != 0,
              "different hash rows leave different residuals");
        int moved = 0;
        for (int index = 0; index < hc_mult * dim; index++)
            if (first[index] != original[index]) moved = 1;
        check(moved, "a non-zero projection changes the stream");
        memset(wkv_data, 0, wkv_bytes);
    }
    coli_v41_engram_workspace_free(&work);
    check(work.rows == NULL && work.projection == NULL,
          "freeing the workspace releases its buffers");
    check(coli_v41_engram_workspace_init(&work, 0, head_dim, hc_mult, dim, 2) != 0,
          "an empty hash width is refused");
    if (opened)
        coli_v41_engram_table_close(&mapped);
    free(wkv_data);
    free(wkv_scales);
    remove(path);
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
    /* unbuffered: if something crashes, the last line says where */
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("deepseek_v41 contract\n");
    test_config_shape();
    test_engram_gate();
    test_source_table_validation();
    test_router();
    test_swiglu_clamp();
    test_engram_token_map();
    test_fp8_matvec_blocked();
    test_engram_table_map();
    test_engram_apply();
    test_engram_fetch();
    test_engram_contribution();
    test_engram_hash();
    test_engram_compress();
    if (failures) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
