#!/usr/bin/env python3
"""Hold the engram addressing to the *reference implementation*, not to a transcript.

The C unit is checked against golden vectors produced by this repository's own
transcription of the reference hash rule. That is only half a verification: if the
transcription misreads the reference, both sides agree and both are wrong. This
script closes the loop by running the official engram code (`v41_ref_engram.py`,
the checkpoint's own implementation) on the same inputs and comparing:

    1. the compressed token map -- the reference asserts its class count itself
    2. the primes, offsets and multipliers it derives internally
    3. the hash ids for the same sequences, prefill and decode paths

Nothing here reaches the network or the weights: it is the addressing only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REFERENCE = Path('C:/Users/david/v41_ref_engram.py')
LAYOUT_TOOL = Path('C:/Users/david/colibri-dev/c/tools/make_deepseek_v41_engram.py')
FIXTURE = Path('C:/Users/david/colibri-dev/c/tools/make_deepseek_v41_engram_fixture.py')
TOKENIZER = Path('C:/Users/david/v41_tokenizer.json')
CONFIG = Path('C:/Users/david/v41_config.json')


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import torch
    from tokenizers import Tokenizer

    engram = load(REFERENCE, 'reference_engram')
    layout_tool = load(LAYOUT_TOOL, 'layout_tool')   # stage one: reconstruction
    fixture = load(FIXTURE, 'fixture_tool')          # stage two: hash rule + vectors

    raw = json.loads(CONFIG.read_text(encoding='utf-8'))
    text = raw['text_config']

    class Shim:
        """The reference expects an HF fast tokenizer; the raw Rust object is the
        same tokenizer, so it only needs the two attributes the reference touches."""

        def __init__(self, tokenizer):
            self.backend_tokenizer = tokenizer

        def __len__(self):
            return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    args = SimpleNamespace(
        engram_layer_ids=list(text['engram_layer_ids']),
        engram_max_ngram_size=text['engram_max_ngram_size'],
        engram_n_heads=text['engram_n_heads'],
        engram_head_dim=text['engram_head_dim'],
        engram_vocab_size=text['engram_vocab_size'],
        engram_num_embeddings=list(text['engram_num_embeddings']),
        engram_compressed_vocab_size=text['engram_compressed_vocab_size'],
        engram_pad_id=text['engram_pad_token_id'],
        max_batch_size=1,
        max_seq_len=64,
    )

    layout = engram.EngramLayout.from_args(args)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    # NgramHashState.__init__ builds the compressed map, asserts its class count
    # against args.engram_compressed_vocab_size, and derives the multipliers itself
    state = engram.NgramHashState(args, layout, Shim(tokenizer))
    print("reference built its own layout: class count assertion passed")

    my_layout = layout_tool.hash_layout(text)
    reference_primes = [[list(per) for per in layer] for layer in layout.primes]
    reference_offsets = []
    for layer in layout.primes:
        flat = [prime for per in layer for prime in per]
        running, acc = [0], 0
        for prime in flat[:-1]:
            acc += prime
            running.append(acc)
        reference_offsets.append(running)
    same_primes = reference_primes == my_layout['primes']
    same_offsets = reference_offsets == my_layout['offsets']
    reference_multipliers = state.multipliers.tolist()
    same_multipliers = reference_multipliers == my_layout['multipliers']
    print(f"primes identical      : {same_primes}")
    print(f"offsets identical     : {same_offsets}")
    print(f"multipliers identical : {same_multipliers}")
    print(f"  reference multipliers: {reference_multipliers}")
    if not (same_primes and same_offsets and same_multipliers):
        print("MISMATCH: the frozen layout is not what the reference derives")
        return 1

    reference_map = state.token_map.tolist()

    sequences = {
        'ascii': list(range(0, 16)),
        'mixed': [128000, 5, 5, 99091, 7, 0, 100, 100, 100, 129279],
        'long': [17, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
    }
    problems = 0
    for name, ids in sequences.items():
        tensor = torch.tensor([ids], dtype=torch.int64)
        # prefill from zero
        hashes = state.forward(tensor, 0)
        for layer_position, layer_id in enumerate(layout.layer_ids):
            reference_ids = hashes[0, :, layer_position, :].flatten().tolist()
            mine = fixture.hash_ids(
                [reference_map[token] for token in ids], [],
                layout.max_ngram_size, layout.n_heads, reference_primes[layer_position],
                reference_multipliers[layer_position], reference_offsets[layer_position],
                reference_map[args.engram_pad_id])
            if reference_ids != mine:
                problems += 1
                print(f"  MISMATCH prefill {name} layer {layer_id}: "
                      f"{reference_ids[:6]} vs {mine[:6]}")

        # decode: fill the cache with the prefix, then feed the last token alone
        decode_state = engram.NgramHashState(args, layout, Shim(tokenizer))
        decode_state.forward(torch.tensor([ids[:-1]], dtype=torch.int64), 0)
        step = decode_state.forward(torch.tensor([[ids[-1]]], dtype=torch.int64),
                                    len(ids) - 1)
        for layer_position, layer_id in enumerate(layout.layer_ids):
            reference_ids = step[0, 0, layer_position, :].flatten().tolist()
            history = [reference_map[token] for token in ids[:-1]]
            mine = fixture.hash_ids(
                [reference_map[ids[-1]]], history,
                layout.max_ngram_size, layout.n_heads, reference_primes[layer_position],
                reference_multipliers[layer_position], reference_offsets[layer_position],
                reference_map[args.engram_pad_id])
            if reference_ids != mine:
                problems += 1
                print(f"  MISMATCH decode {name} layer {layer_id}: "
                      f"{reference_ids[:6]} vs {mine[:6]}")
    if problems:
        print(f"\n{problems} mismatch(es): the addressing does not match the reference")
        return 1
    print("\nreference parity OK: prefill and decode ids match the official "
          "engram implementation on every sequence")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
