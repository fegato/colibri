/* Throwaway-values probe: print what the engine's own primitives compute, so the torch
 * reference can be held to them (they are already validated by the V4 oracle).
 *
 * One JSON object per line: {"op":..., "input":[...], "output":[...], ...}
 * Build (from c/):  gcc -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -O2 -I. \
 *                       tests/v41_ops_probe.c COLI_V41_UNIT_NATIVE_QUANT.o ... -o v41_ops_probe
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "deepseek_v41_internal.h"
#include "hyper_connections.h"

static void print_floats(const char *name, const float *values, int count) {
    printf("\"%s\":[", name);
    for (int index = 0; index < count; index++)
        printf("%s%.9g", index ? "," : "", (double)values[index]);
    printf("]");
}

static void print_bytes(const char *name, const uint8_t *values, int count) {
    printf("\"%s\":[", name);
    for (int index = 0; index < count; index++)
        printf("%s%u", index ? "," : "", (unsigned)values[index]);
    printf("]");
}

/* A pattern with several magnitudes, a zero block, and a subnormal block: the scale rule
 * (2**ceil(log2(amax/448))) must be exercised, not just the happy case. */
static void fill_input(float *values, int count) {
    for (int index = 0; index < count; index++) {
        float magnitude = (float)(1.0 + (index % 7) * 0.75);
        values[index] = ((index % 5) == 0) ? 0.0f : (index & 1 ? -magnitude : magnitude);
    }
    values[3] = 311.0f;      /* a large one, so the exponent is not the floor */
    values[40] = 0.0007f;    /* small, inside the second block */
    values[96] = 0.0f;       /* a block that is entirely zero */
}

int main(void) {
    /* 128 wide: both block geometries the engine supports divide it (and the vendor's own
     * wrapper asserts divisibility, so a partial block is not a case to invent) */
    enum { length = 128 };
    float input[length];
    fill_input(input, length);

    for (int block = 32; block <= 128; block *= 4) {
        float quantized[length];
        uint8_t scales[length] = {0};
        memset(quantized, 0, sizeof(quantized));
        if (coli_fp8_activation_qdq_ref(quantized, scales, input, (size_t)length,
                                        (size_t)block) != 0) {
            printf("{\"op\":\"fp8_act_qdq\",\"block\":%d,\"error\":\"refused\"}\n", block);
            continue;
        }
        printf("{\"op\":\"fp8_act_qdq\",\"block\":%d,", block);
        print_floats("input", input, length);
        printf(",");
        print_floats("output", quantized, length);
        printf(",");
        print_bytes("scales", scales, length / block);
        printf("}\n");
    }

    /* an fp8 weight with the V4 geometry (128-wide blocks): the shared matvec's own case */
    {
        enum { rows = 4, columns = 128, block = 128 };
        uint8_t data[rows * columns];
        /* ceil(4 / 128) = 1 scale row, 128 / 128 = 1 scale column: the exact sizes the
         * validator demands, and the tile a weight scale covers is block x block */
        static float packed_scales[1][1];
        for (int index = 0; index < rows * columns; index++)
            data[index] = (uint8_t)(0x30 + (index % 5));      /* e4m3 values near 1..3 */
        packed_scales[0][0] = 2.0f;
        ColiTensorView weight;
        memset(&weight, 0, sizeof(weight));
        weight.format = COLI_TENSOR_FP8_E4M3_BLOCK;
        weight.scale_format = COLI_SCALE_F32;
        weight.data = data;
        weight.scales = packed_scales;
        weight.data_bytes = sizeof(data);
        weight.scale_bytes = sizeof(packed_scales);
        weight.rows = rows;
        weight.columns = columns;
        weight.block_rows = 128;
        weight.block_columns = 128;
        float output[rows];
        memset(output, 0, sizeof(output));
        if (coli_fp8_matvec_ref(output, &weight, input) != 0) {
            printf("{\"op\":\"fp8_matvec_128\",\"error\":\"refused\"}\n");
        } else {
            printf("{\"op\":\"fp8_matvec_128\",\"rows\":%d,\"columns\":%d,", rows, columns);
            print_floats("input", input, columns);
            printf(",");
            print_floats("output", output, rows);
            printf(",");
            print_bytes("weight", data, rows * columns);
            printf(",");
            print_floats("scales", &packed_scales[0][0], 1);
            printf("}\n");
        }
    }

    /* the hyper-connection split, which every block of the architecture goes through */
    {
        enum { hc = 4, mix_hc = (2 + 4) * 4 };
        float mixes[3 * mix_hc], scale[3] = {1.25f, 0.75f, 0.5f}, base[mix_hc];
        float pre[3 * hc], post[3 * hc], comb[3 * hc * hc];
        for (int index = 0; index < mix_hc; index++)
            base[index] = 0.05f * (float)(index % 5) - 0.1f;
        for (int index = 0; index < 3 * mix_hc; index++)
            mixes[index] = 0.4f * sinf((float)index * 0.7f) + 0.1f * (float)(index % 3);
        memset(pre, 0, sizeof(pre));
        memset(post, 0, sizeof(post));
        memset(comb, 0, sizeof(comb));
        /* one token per call: the primitive takes a single (2 + hc) * hc vector */
        int failed = 0;
        for (int token = 0; token < 3 && !failed; token++)
            failed = coli_v41_hc_split_sinkhorn(pre + token * hc, post + token * hc,
                                                comb + token * hc * hc,
                                                mixes + token * mix_hc, scale, base,
                                                hc, 3, 1e-6f) != 0;   /* iters, then eps */
        if (!failed) {
            printf("{\"op\":\"hc_split_sinkhorn\",\"hc\":%d,\"count\":3,", hc);
            print_floats("mixes", mixes, 3 * mix_hc);
            printf(",");
            print_floats("scale", scale, 3);
            printf(",");
            print_floats("base", base, mix_hc);
            printf(",");
            print_floats("pre", pre, 3 * hc);
            printf(",");
            print_floats("post", post, 3 * hc);
            printf(",");
            print_floats("comb", comb, 3 * hc * hc);
            printf("}\n");
        } else {
            printf("{\"op\":\"hc_split_sinkhorn\",\"error\":\"refused\"}\n");
        }
    }
    return 0;
}
