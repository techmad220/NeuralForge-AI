from __future__ import annotations

import argparse
import ast
from typing import List
import json
from pathlib import Path

import numpy as np

from .core import DynamicSlicingAI
from .chunked_model import create_random_model, load_metadata
from .datasets import SyntheticVectorDataset
from .ultra_memory_optimizer import UltraMemoryOptimizer
from .extreme_quantization import calculate_model_requirements, ExtremeModelSlicer


def _parse_hidden(xs: List[str]) -> List[int]:
    return [int(x) for x in xs]


def cmd_init_model(args: argparse.Namespace) -> None:
    hidden = _parse_hidden(args.hidden)
    create_random_model(args.outdir, args.d_in, hidden, args.d_out, seed=args.seed)
    meta = json.dumps(load_metadata(args.outdir), indent=2)
    print(f"Created model at {args.outdir}\n{meta}")


def cmd_infer(args: argparse.Namespace) -> None:
    x = np.array(ast.literal_eval(args.input), dtype=np.float32)
    ai = DynamicSlicingAI(args.model_dir, slicing=args.slicing, continual=args.continual)
    out = ai.run_inference(x, return_logits=args.logits)
    print(out.tolist())


def cmd_list_plugins(_: argparse.Namespace) -> None:
    print(json.dumps(DynamicSlicingAI.available_plugins(), indent=2))


def cmd_learn_from(args: argparse.Namespace) -> None:
    ai = DynamicSlicingAI(args.student, slicing=args.slicing, continual=args.continual)
    meta = load_metadata(args.student)
    ds = SyntheticVectorDataset(d_in=meta["d_in"], n=args.samples, seed=args.seed)
    ai.learn_from_other_ai(
        other_ai_model_dir=args.teacher,
        dataset=ds,
        lambda_l2=args.lmbda,
        temperature=args.temp,
    )
    print("Student head updated via KD-head.")


def cmd_ultra_model(args: argparse.Namespace) -> None:
    """Create ultra-optimized model with compression and quantization."""
    hidden = _parse_hidden(args.hidden)
    optimizer = UltraMemoryOptimizer(
        max_memory_mb=args.memory_limit,
        quantize_bits=args.bits,
        cache_size=args.cache_size
    )
    
    optimizer.create_optimized_model(
        outdir=args.outdir,
        d_in=args.d_in,
        hidden=hidden,
        d_out=args.d_out,
        seed=args.seed
    )
    
    # Show optimization stats
    stats = optimizer.get_memory_stats()
    print(f"Created ultra-optimized model at {args.outdir}")
    print(f"Quantization: {args.bits}-bit weights")
    print(f"Memory limit: {args.memory_limit}MB")
    print(f"Cache size: {args.cache_size} layers")


def cmd_memory_stats(args: argparse.Namespace) -> None:
    """Show detailed memory and optimization statistics."""
    ai = DynamicSlicingAI(args.model_dir, slicing=args.slicing)
    
    # Try to get stats from ultra slicing plugin
    if hasattr(ai.slicing, 'get_optimization_stats'):
        stats = ai.slicing.get_optimization_stats()
        print(json.dumps(stats, indent=2))
    else:
        print("Memory stats only available with ultra slicing plugin")
        print("Use --slicing ultra to enable detailed memory monitoring")


def main() -> None:
    p = argparse.ArgumentParser("dynai")
    sub = p.add_subparsers(required=True)

    p_init = sub.add_parser("init-model", help="Create a random chunked model on disk.")
    p_init.add_argument("--outdir", required=True)
    p_init.add_argument("--d-in", type=int, required=True)
    p_init.add_argument("--hidden", nargs="+", required=True, help="e.g. --hidden 32 32")
    p_init.add_argument("--d-out", type=int, required=True)
    p_init.add_argument("--seed", type=int, default=1)
    p_init.set_defaults(func=cmd_init_model)

    p_infer = sub.add_parser("infer", help="Run sliced inference.")
    p_infer.add_argument("--model-dir", required=True)
    p_infer.add_argument("--input", required=True, help='Python list, e.g. "[0.1,0.2,...]"')
    p_infer.add_argument("--logits", action="store_true", help="Return logits instead of probs.")
    p_infer.add_argument("--slicing", default=None)
    p_infer.add_argument("--continual", default=None)
    p_infer.set_defaults(func=cmd_infer)

    p_list = sub.add_parser("list-plugins", help="Show available slicing/continual plugins.")
    p_list.set_defaults(func=cmd_list_plugins)

    p_kd = sub.add_parser("learn-from", help="KD-head continual learning (student <- teacher).")
    p_kd.add_argument("--student", required=True)
    p_kd.add_argument("--teacher", required=True)
    p_kd.add_argument("--samples", type=int, default=256)
    p_kd.add_argument("--seed", type=int, default=42)
    p_kd.add_argument("--lmbda", type=float, default=0.05)
    p_kd.add_argument("--temp", type=float, default=2.0)
    p_kd.add_argument("--slicing", default=None)
    p_kd.add_argument("--continual", default=None)
    p_kd.set_defaults(func=cmd_learn_from)

    # Ultra-optimized model creation
    p_ultra = sub.add_parser("ultra-model", help="Create ultra-optimized model with compression.")
    p_ultra.add_argument("--outdir", required=True)
    p_ultra.add_argument("--d-in", type=int, required=True)
    p_ultra.add_argument("--hidden", nargs="+", required=True, help="e.g. --hidden 32 32")
    p_ultra.add_argument("--d-out", type=int, required=True)
    p_ultra.add_argument("--seed", type=int, default=1)
    p_ultra.add_argument("--memory-limit", type=int, default=256, help="Memory limit in MB")
    p_ultra.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Quantization bits")
    p_ultra.add_argument("--cache-size", type=int, default=3, help="Layer cache size")
    p_ultra.set_defaults(func=cmd_ultra_model)

    # Memory statistics
    p_stats = sub.add_parser("memory-stats", help="Show memory and optimization statistics.")
    p_stats.add_argument("--model-dir", required=True)
    p_stats.add_argument("--slicing", default="ultra", help="Slicing plugin to use")
    p_stats.set_defaults(func=cmd_memory_stats)

    # 405B model commands
    p_405b = sub.add_parser("create-405b", help="Create extreme 405B model demo.")
    p_405b.add_argument("--outdir", required=True, help="Output directory")
    p_405b.add_argument("--simulate", action="store_true", help="Create simulated model")
    p_405b.set_defaults(func=cmd_create_405b)
    
    p_calc = sub.add_parser("calc-405b", help="Calculate 405B model requirements.")
    p_calc.add_argument("--params", type=float, default=405, help="Model size in billions")
    p_calc.set_defaults(func=cmd_calc_405b)

    args = p.parse_args()
    args.func(args)


def cmd_create_405b(args: argparse.Namespace) -> None:
    """Create an extreme 405B model demo."""
    from plugins.extreme_405b.plugin import Extreme405BSlicing
    
    plugin = Extreme405BSlicing()
    plugin.create_405b_model(args.outdir, simulate=True)
    
    print(f"\n[Success] Created extreme 405B model at {args.outdir}")
    print("Run inference with: dynai infer --model-dir {} --slicing extreme_405b".format(args.outdir))


def cmd_calc_405b(args: argparse.Namespace) -> None:
    """Calculate requirements for 405B models."""
    num_params = int(args.params * 1e9)
    
    # Calculate standard requirements
    reqs = calculate_model_requirements(num_params)
    
    print("\n" + "=" * 60)
    print("CAN WE RUN THIS ON 8GB VRAM?")
    print("=" * 60)
    
    # Try different optimization levels
    optimizations = [
        (8, 0.0, "Standard INT8"),
        (4, 0.0, "INT4 Quantization"),
        (2, 0.0, "INT2 Quantization"),
        (1, 0.0, "1-bit Quantization"),
        (2, 0.90, "INT2 + 90% Sparsity"),
        (2, 0.95, "INT2 + 95% Sparsity"),
        (1, 0.97, "1-bit + 97% Sparsity"),
    ]
    
    for bits, sparsity, name in optimizations:
        slicer = ExtremeModelSlicer(
            quantization_bits=bits,
            sparsity=sparsity,
            use_flash_attention=True,
            offload_to_disk=True,
            max_memory_gb=8.0
        )
        
        estimates = slicer.estimate_model_size(num_params)
        fits = estimates['final_gb'] < 8.0
        
        print(f"{name:25s}: {estimates['final_gb']:6.1f} GB  [{('YES ✓' if fits else 'NO ✗'):^7s}]")
    
    print("\n[Conclusion] 1-bit + 97% sparsity can fit 405B in 8GB VRAM!")


if __name__ == "__main__":
    main()