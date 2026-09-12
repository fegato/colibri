/* tests/engram_math_probe.c -- prints the engram fetch/gate inputs and outputs for
 * fixed data, so tools/check_deepseek_v41_engram_math.py can hold them to the
 * reference implementation (which is Python and cannot be linked into a C test).
 *
 * The probe prints its *inputs* as well as its outputs: the validator rebuilds the
 * reference from what it reads, so the two sides cannot drift apart.
 *
 * Build (what the tool runs):
 *   gcc -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -O2 -I<c> tests/engram_math_probe.c \
 *       COLI_V41_UNIT_ENGRAM.o COLI_V41_UNIT_NATIVE_QUANT.o -o engram_math_probe -lm
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "deepseek_v41_internal.h"

#define TABLE_ROWS 4
#define HEAD_DIM 32
#define HC_MULT 2
#define DIM 4

static void print_floats(const char *label, const float *values, int count) {
    printf("%s", label);
    for (int index = 0; index < count; index++)
        printf(" %.9g", values[index]);
    printf("\n");
}

static void print_bytes(const char *label, const uint8_t *bytes, int count) {
    printf("%s", label);
    for (int index = 0; index < count; index++)
        printf(" %02x", bytes[index]);
    printf("\n");
}

int main(void) {
    /* fixed, printable data: E4M3 magnitudes in [0.5, 3.0], E8M0 exponents 124..128 */
    uint8_t table[TABLE_ROWS * HEAD_DIM];
    uint8_t scales[TABLE_ROWS];
    for (int index = 0; index < TABLE_ROWS * HEAD_DIM; index++)
        table[index] = (uint8_t)(0x34 + (index * 7) % 12);
    for (int row = 0; row < TABLE_ROWS; row++)
        scales[row] = (uint8_t)(124 + row);

    printf("table_rows %d\n", TABLE_ROWS);
    printf("head_dim %d\n", HEAD_DIM);
    print_bytes("table", table, sizeof(table));
    print_bytes("scales", scales, sizeof(scales));

    const int64_t order[TABLE_ROWS] = {2, 0, 3, 1};
    printf("order %d %d %d %d\n", (int)order[0], (int)order[1], (int)order[2],
           (int)order[3]);
    float rows[TABLE_ROWS * HEAD_DIM];
    if (coli_v41_engram_fetch_rows(rows, order, TABLE_ROWS, HEAD_DIM, table,
                                   sizeof(table), scales, sizeof(scales)) != 0) {
        printf("fetch failed\n");
        return 1;
    }
    print_floats("fetch", rows, TABLE_ROWS * HEAD_DIM);

    const float stream[HC_MULT * DIM] = {1.0f, 0.0f, 0.0f, 1.0f,
                                         -2.0f, 0.5f, 0.25f, 3.0f};
    const float key[HC_MULT * DIM] = {1.0f, 0.0f, 0.0f, 1.0f,
                                      0.5f, -1.0f, 2.0f, 0.0f};
    const float weight[HC_MULT * DIM] = {1.0f, 0.5f, 0.25f, 1.0f,
                                         2.0f, 1.0f, 0.5f, 0.5f};
    const float value[DIM] = {10.0f, -4.0f, 0.5f, 2.0f};
    printf("hc_mult %d\n", HC_MULT);
    printf("dim %d\n", DIM);
    print_floats("stream", stream, HC_MULT * DIM);
    print_floats("key", key, HC_MULT * DIM);
    print_floats("weight", weight, HC_MULT * DIM);
    print_floats("value", value, DIM);

    float copies[HC_MULT * DIM];
    if (coli_v41_engram_gate(copies, stream, key, value, weight, HC_MULT, DIM,
                             1e-6f) != 0) {
        printf("gate failed\n");
        return 1;
    }
    print_floats("gate", copies, HC_MULT * DIM);
    return 0;
}
