"""
Entry point for training one category.

Usage:
    python scripts/train_category.py --category gambling
    python scripts/train_category.py --category gambling --steps 60000000 --wandb
    python scripts/train_category.py --category trick_taking --resume checkpoints/trick_taking_latest.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core.types import CategoryID
from models.full_agent import build_agent
from training.curriculum import CURRICULA
from training.multi_task_trainer import MultiTaskTrainer, TrainingConfig
from training.ppo import PPOConfig

CATEGORY_MAP = {
    "gambling":         CategoryID.GAMBLING,
    "matching":         CategoryID.MATCHING_SHEDDING,
    "matching_shedding": CategoryID.MATCHING_SHEDDING,
    "trick_taking":     CategoryID.TRICK_TAKING,
    "trick":            CategoryID.TRICK_TAKING,
    "melding":          CategoryID.MELDING,
    "climbing":         CategoryID.CLIMBING,
    "solitaire":        CategoryID.SOLITAIRE,
}

DEFAULT_STEPS = {
    CategoryID.GAMBLING:          60_000_000,
    CategoryID.MATCHING_SHEDDING: 80_000_000,
    CategoryID.TRICK_TAKING:     100_000_000,
    CategoryID.MELDING:           80_000_000,
    CategoryID.CLIMBING:         100_000_000,
    CategoryID.SOLITAIRE:         50_000_000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AI card game agent for one category")
    p.add_argument("--category", required=True,
                   choices=list(CATEGORY_MAP.keys()),
                   help="Which game category to train")
    p.add_argument("--steps", type=int, default=None,
                   help="Total training steps (default: per-category preset)")
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    category = CATEGORY_MAP[args.category]
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")

    print(f"Device: {device}")
    print(f"Category: {category.name}")

    # Build agent
    agent = build_agent(device=device)
    print(f"Agent parameters: {agent.total_parameters():,}")

    # Load checkpoint if resuming
    if args.resume:
        _load_checkpoint(agent, args.resume)
        print(f"Resumed from {args.resume}")

    # Configure training
    total_steps = args.steps or DEFAULT_STEPS[category]
    config = TrainingConfig(
        category=category,
        total_steps=total_steps,
        use_wandb=args.wandb,
        checkpoint_dir=args.checkpoint_dir,
        ppo=PPOConfig(lr=args.lr, minibatch_size=args.minibatch),
    )

    print(f"Training for {total_steps:,} steps")
    print(f"Curriculum phases: {[p.name for p in CURRICULA[category]]}")
    print("-" * 60)

    trainer = MultiTaskTrainer(agent, config)
    trainer.train()

    # Save final checkpoint
    save_path = os.path.join(args.checkpoint_dir, f"{category.name.lower()}_final.pt")
    _save_checkpoint(agent, save_path, total_steps)
    print(f"Saved final checkpoint: {save_path}")


def _save_checkpoint(agent, path: str, step: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": step,
        "backbone": agent.backbone.state_dict(),
        "policy_head": agent.policy_head.state_dict(),
        "value_head": agent.value_head.state_dict(),
    }, path)


def _load_checkpoint(agent, path: str) -> None:
    state = torch.load(path, map_location=agent.device)
    agent.backbone.load_state_dict(state["backbone"])
    agent.policy_head.load_state_dict(state["policy_head"])
    agent.value_head.load_state_dict(state["value_head"])


if __name__ == "__main__":
    main()
