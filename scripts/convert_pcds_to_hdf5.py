#!/usr/bin/env python3
"""Convert point cloud pickle file to HDF5 format for efficient loading.

Usage:
    python scripts/convert_pcds_to_hdf5.py \
        --input data/DexGraspNet/object_pcds_nors.pkl \
        --output data/DexGraspNet/object_pcds.h5 \
        --max_points 4096

This creates an HDF5 file with:
- One dataset per object: obj_code -> (max_points, 6) float32
- Attributes: num_objects, max_points, source_file
"""
import argparse
import logging
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_pcds_to_hdf5(
    input_path: str,
    output_path: str,
    max_points: int = 4096,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """Convert pickle point clouds to HDF5 format."""
    logger.info(f"Loading point clouds from {input_path}...")
    with open(input_path, "rb") as f:
        pcds = pickle.load(f)

    logger.info(f"Loaded {len(pcds)} objects")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing to {output_path}...")
    with h5py.File(output_path, "w") as f:
        # Store metadata
        f.attrs["num_objects"] = len(pcds)
        f.attrs["max_points"] = max_points
        f.attrs["source_file"] = os.path.basename(input_path)

        # Create group for point clouds
        pc_group = f.create_group("point_clouds")

        # Store object list for quick lookup
        obj_codes = sorted(pcds.keys())
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("obj_codes", data=obj_codes, dtype=dt)

        for obj_code in tqdm(obj_codes, desc="Converting"):
            pc = pcds[obj_code]
            if not isinstance(pc, np.ndarray):
                pc = np.array(pc)

            # Ensure float32
            pc = pc.astype(np.float32)

            # Subsample if needed
            if pc.shape[0] > max_points:
                indices = np.random.permutation(pc.shape[0])[:max_points]
                pc = pc[indices]

            # Pad if needed (to ensure consistent shape)
            if pc.shape[0] < max_points:
                pad = np.zeros((max_points - pc.shape[0], pc.shape[1]), dtype=np.float32)
                pc = np.concatenate([pc, pad], axis=0)

            # Store with compression
            pc_group.create_dataset(
                obj_code,
                data=pc,
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_opts,
            )

    final_size = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"Done! Output size: {final_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert point cloud pickle to HDF5")
    parser.add_argument(
        "--input",
        type=str,
        default="data/DexGraspNet/object_pcds_nors.pkl",
        help="Input pickle file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/DexGraspNet/object_pcds.h5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=4096,
        help="Maximum points per object (default: 4096)",
    )
    args = parser.parse_args()

    convert_pcds_to_hdf5(args.input, args.output, args.max_points)


if __name__ == "__main__":
    main()
