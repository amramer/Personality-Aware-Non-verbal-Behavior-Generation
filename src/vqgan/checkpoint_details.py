"""Print basic details of a saved VQ-VAE checkpoint.

Usage:
    python checkpoint_details.py path/to/model_best.pth
"""
import argparse

import torch


def load_checkpoint(checkpoint_path):
    """Load the checkpoint from the given path onto the CPU."""
    return torch.load(checkpoint_path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint_path", help="Path to a .pth checkpoint file")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint_path)
    print("Last epoch:", checkpoint["epoch"])


if __name__ == "__main__":
    main()
