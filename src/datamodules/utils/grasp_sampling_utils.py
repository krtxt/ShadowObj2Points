"""
Grasp sampling utility functions shared across SceneLeap datasets.

This module centralizes sampling strategies (random, FPS, NPS, chunked) so that
datasets can reuse identical logic without duplicating helper methods.
"""

from __future__ import annotations

from typing import List, Sequence

import torch


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Perform Farthest Point Sampling (FPS) on 3D points.

    Args:
        points: Point coordinates (N, 3)
        num_samples: Number of samples to select

    Returns:
        Tensor of sampled indices (num_samples,)
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")

    N = points.shape[0]
    device = points.device

    if N <= num_samples:
        return torch.arange(N, device=device)

    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    sampled_indices[0] = torch.randint(0, N, (1,), device=device)
    distances = torch.full((N,), float("inf"), device=device)

    for i in range(1, num_samples):
        last_idx = sampled_indices[i - 1]
        last_point = points[last_idx]
        current_distances = torch.norm(points - last_point.unsqueeze(0), dim=1)
        distances = torch.minimum(distances, current_distances)
        sampled_indices[i] = torch.argmax(distances)
        distances[sampled_indices[i]] = 0

    return sampled_indices


def nearest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Perform Nearest Point Sampling (NPS) by seeding a random center.

    Args:
        points: Point coordinates (N, 3)
        num_samples: Number of samples to select

    Returns:
        Tensor of sampled indices (num_samples,)
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")

    N = points.shape[0]
    device = points.device

    if N <= num_samples:
        return torch.arange(N, device=device)

    center_idx = torch.randint(0, N, (1,), device=device).item()
    center_point = points[center_idx]
    distances = torch.norm(points - center_point.unsqueeze(0), dim=1)
    _, nearest_indices = torch.topk(distances, num_samples, largest=False)
    return nearest_indices


def sample_indices_from_available(
    indices: Sequence[int],
    all_poses_for_obj: torch.Tensor,
    num_samples: int,
    strategy: str,
) -> List[int]:
    """
    Sample indices according to the configured strategy.

    Args:
        indices: Existing indices to sample from
        all_poses_for_obj: Pose tensor used for spatial strategies
        num_samples: Number of samples desired
        strategy: Sampling strategy name

    Returns:
        List of sampled indices (with replacement if necessary)
    """
    if num_samples <= 0:
        return []

    indices = list(indices)
    n = len(indices)

    if n == 0:
        return []

    if n <= num_samples:
        if strategy == "repeat":
            return [indices[i % n] for i in range(num_samples)]
        extra = [
            indices[torch.randint(0, n, (1,)).item()]
            for _ in range(max(0, num_samples - n))
        ]
        return indices + extra

    if strategy == "random":
        perm = torch.randperm(n)[:num_samples]
        return [indices[i] for i in perm]
    if strategy == "first_n":
        return indices[:num_samples]
    if strategy == "repeat":
        return [indices[i % n] for i in range(num_samples)]
    if strategy == "farthest_point":
        points = all_poses_for_obj[indices, :3]
        fps_idx = farthest_point_sampling(points, num_samples)
        return [indices[i.item()] for i in fps_idx]
    if strategy == "nearest_point":
        points = all_poses_for_obj[indices, :3]
        nps_idx = nearest_point_sampling(points, num_samples)
        return [indices[i.item()] for i in nps_idx]

    # Default fallback: random
    perm = torch.randperm(n)[:num_samples]
    return [indices[i] for i in perm]


def sample_grasps_from_available(
    available_grasps: torch.Tensor,
    num_grasps: int,
    strategy: str,
) -> torch.Tensor:
    """
    Sample a fixed number of grasps from available poses.

    Args:
        available_grasps: Pose tensor (N, 23)
        num_grasps: Number of grasps to sample
        strategy: Sampling strategy

    Returns:
        Sampled grasp tensor (num_grasps, 23)
    """
    num_available = available_grasps.shape[0]

    if num_available == 0:
        return torch.zeros(
            (num_grasps, available_grasps.shape[1]),
            dtype=available_grasps.dtype,
            device=available_grasps.device,
        )

    if num_available >= num_grasps:
        if strategy == "random":
            indices = torch.randperm(num_available, device=available_grasps.device)[
                :num_grasps
            ]
        elif strategy == "first_n":
            indices = torch.arange(num_grasps, device=available_grasps.device)
        elif strategy == "farthest_point":
            indices = farthest_point_sampling(available_grasps[:, :3], num_grasps)
        elif strategy == "nearest_point":
            indices = nearest_point_sampling(available_grasps[:, :3], num_grasps)
        else:  # repeat / fallback
            indices = (
                torch.arange(num_grasps, device=available_grasps.device) % num_available
            )
        return available_grasps.index_select(0, indices)

    # num_available < num_grasps
    if strategy == "repeat":
        indices = (
            torch.arange(num_grasps, device=available_grasps.device) % num_available
        )
    elif strategy in {"farthest_point", "nearest_point"}:
        base_indices = torch.arange(num_available, device=available_grasps.device)
        remaining = num_grasps - num_available
        additional = torch.randint(
            0, num_available, (remaining,), device=available_grasps.device
        )
        indices = torch.cat([base_indices, additional], dim=0)
    else:
        indices = torch.randint(
            0, num_available, (num_grasps,), device=available_grasps.device
        )

    return available_grasps.index_select(0, indices)


def generate_exhaustive_chunks(
    all_poses: torch.Tensor,
    num_grasps: int,
    strategy: str,
) -> List[List[int]]:
    """
    Generate non-overlapping grasp chunks for exhaustive sampling.

    Args:
        all_poses: Pose tensor for the object (N, 23)
        num_grasps: Chunk size
        strategy: Strategy name ("sequential", "random", "interleaved",
                  "chunk_farthest_point", "chunk_nearest_point")

    Returns:
        List of index lists, one per chunk
    """
    total_grasps = all_poses.shape[0]
    num_chunks = total_grasps // num_grasps
    if num_chunks == 0:
        return []

    if strategy == "sequential":
        return [
            list(range(i * num_grasps, (i + 1) * num_grasps)) for i in range(num_chunks)
        ]
    if strategy == "random":
        all_indices = torch.randperm(total_grasps)
        return [
            all_indices[i * num_grasps : (i + 1) * num_grasps].tolist()
            for i in range(num_chunks)
        ]
    if strategy == "interleaved":
        step = total_grasps // num_grasps
        return [
            [int((start_offset + j * step) % total_grasps) for j in range(num_grasps)]
            for start_offset in range(num_chunks)
        ]
    if strategy in {"chunk_farthest_point", "chunk_nearest_point"}:
        spatial_strategy = (
            "farthest_point" if "farthest" in strategy else "nearest_point"
        )
        return _generate_spatial_chunks(
            all_poses, num_chunks, num_grasps, spatial_strategy
        )

    raise ValueError(f"Unknown exhaustive sampling strategy: {strategy}")


def _generate_spatial_chunks(
    all_poses: torch.Tensor,
    num_chunks: int,
    num_grasps: int,
    spatial_strategy: str,
) -> List[List[int]]:
    translation_points = all_poses[:, :3]
    total_points = translation_points.shape[0]
    used_indices = set()
    chunk_indices: List[List[int]] = []

    for _ in range(num_chunks):
        available_indices = [i for i in range(total_points) if i not in used_indices]
        if len(available_indices) < num_grasps:
            break

        available_points = translation_points[available_indices]

        if spatial_strategy == "farthest_point":
            local_indices = farthest_point_sampling(available_points, num_grasps)
            selected = [available_indices[i] for i in local_indices]
        else:
            local_indices = nearest_point_sampling(available_points, num_grasps)
            selected = [available_indices[i] for i in local_indices]

        chunk_indices.append(selected)
        used_indices.update(selected)

    return chunk_indices
