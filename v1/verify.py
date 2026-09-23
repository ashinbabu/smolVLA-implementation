import argparse

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

parser = argparse.ArgumentParser(description="Load a SmolVLA checkpoint.")
parser.add_argument(
	"--checkpoint",
	default="outputs/train/smolvla_aloha_insertion/checkpoints/020000/pretrained_model",
	help="Path or Hub ID of the SmolVLA checkpoint.",
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device)
params = sum(p.numel() for p in policy.parameters()) / 1e6
print(f"Successfully loaded {args.checkpoint} with {params:.1f}M parameters.")