/* deepseek_v41_portable.c — fail-closed artifact for hosts the V4.1 engine
 * does not build on.
 *
 * The real engine (deepseek_v41.c, forked from the V4 engine by
 * tools/fork_deepseek_v41.py) is gated on the same host matrix as the V4
 * engine: x86-64/aarch64 Linux and Windows, arm64 macOS.  Everywhere else this
 * translation unit keeps the build/install/release/site contracts green -- and
 * it fails CLOSED: deepseek_v41 must never be built as, or fall back to, the V4
 * engine, because V4.1 is a different model and running it on a V4 binary would
 * silently compute the wrong thing.
 *
 * Port tracker: docs/deepseek-v41-delta.md
 */
#include <stdio.h>

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    fprintf(stderr,
        "deepseek_v41: the V4.1 engine is not built for this host.\n"
        "This binary is the fail-closed placeholder: it refuses to run rather\n"
        "than fall back to the deepseek_v4 engine, which computes a different\n"
        "model. See docs/deepseek-v41-delta.md\n");
    return 2;
}
