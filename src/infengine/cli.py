import sys
import torch

from .backend import build_model_from_hf, build_tokenizer, resolve_device, resolve_dtype
from .engine import LLMEngine


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse
    parser = argparse.ArgumentParser("infengine-cli")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--bits", type=int, default=0, help="quantize weights to INT* (0 = fp)")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--draft-model", type=str, default=None, help="smaller model for speculative decoding")
    parser.add_argument("--spec-len", type=int, default=4, help="draft tokens per speculative step")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device != "auto" else resolve_device()
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else resolve_dtype(device)

    print(f"Loading model {args.model}...", file=sys.stderr)
    model, cfg = build_model_from_hf(args.model, device=device, dtype=dtype, bits=args.bits)
    tokenizer = build_tokenizer(args.model)
    engine = LLMEngine(
        model, tokenizer, device, dtype,
        block_size=args.block_size, num_blocks=args.num_blocks,
        max_batch=args.max_batch, chunk_size=args.chunk_size,
    )

    print(f"Generating (temp={args.temperature}, top_p={args.top_p})...", file=sys.stderr)
    if args.draft_model:
        print(f"Loading draft model {args.draft_model}...", file=sys.stderr)
        draft_model, _ = build_model_from_hf(args.draft_model, device=device, dtype=dtype, bits=args.bits)
        tokens = engine.generate_speculative(
            args.prompt, draft_model, None,
            spec_len=args.spec_len, max_new_tokens=args.max_new_tokens,
        )
    else:
        tokens = engine.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    out = tokenizer.decode(list(tokens))
    print(out)


if __name__ == "__main__":
    main()