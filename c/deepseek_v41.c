/* deepseek_v41.c — DeepSeek V4.1-Flash engine placeholder.
 *
 * V4.1 (model_type "deepseek_v41") is a NEW architecture, not a V4 re-quant:
 * 40 layers / hidden 5120 / 384 routed experts top-6, sqrtsoftplus router,
 * KV/index shared across designated source layers, engram n-gram tables,
 * DSpark predict layers, swiglu clamp, plus a vision tower. Running it
 * through deepseek_v4.c would silently compute the WRONG model, so this
 * translation unit exists only to keep the build/install/release/site
 * contracts green while the real port lands -- and it fails CLOSED at
 * runtime: any invocation explains itself on stderr and exits nonzero.
 *
 * Port tracker: docs/deepseek-v41-delta.md
 */
#include <stdio.h>

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    fprintf(stderr,
        "deepseek_v41: engine not implemented yet.\n"
        "DeepSeek V4.1-Flash needs its own port (router, shared KV/index,\n"
        "engram, DSpark); running it as deepseek_v4 would silently compute\n"
        "the wrong model. See docs/deepseek-v41-delta.md\n");
    return 2;
}
