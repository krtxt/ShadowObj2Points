from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

import numpy as np

try:
    from pytorch3d.ops import knn_points

    _PYTORCH3D_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    knn_points = None
    _PYTORCH3D_AVAILABLE = False

from torchsdf import compute_sdf

from .hand_constants import SELF_PENETRATION_POINT_RADIUS
from .hand_helper import decompose_hand_pose as _decompose_hand_pose_helper
from .hand_types import HandModelType
from .shadown_hand_model import get_handmodel


class ShadowHandModelAdapter:
    """Expose ShadowHandModel with the same public API as utils.hand_model.HandModel."""

    _EXCLUDED_VISUAL_LINKS: Sequence[str] = ("forearm",)
    _FINGERTIP_LINKS: Sequence[str] = ("fftip", "mftip", "rftip", "lftip", "thtip")
    _DEFAULT_SCALE: float = 1.0

    def __init__(
        self,
        hand_model_type: HandModelType,
        n_surface_points: int,
        rot_type: str,
        device: torch.device,
        anchor: str = "base",
    ) -> None:
        if hand_model_type != HandModelType.SHADOW:
            raise ValueError(
                f"ShadowHandModelAdapter only supports HandModelType.SHADOW, got {hand_model_type}."
            )

        self.hand_model_type = hand_model_type
        self.n_surface_points = int(max(0, n_surface_points))
        self.rot_type = rot_type
        self._device = torch.device(device)
        self.hand_scale = self._DEFAULT_SCALE
        self.anchor = str(anchor).lower()
        if self.anchor not in {"base", "palm_center"}:
            raise ValueError(f"Unsupported anchor '{anchor}'. Expected 'base' or 'palm_center'.")

        self._rot_dim = self._rotation_dim(rot_type)
        self._raw_model = None
        self._link_names: Sequence[str] = ()
        self._n_dofs: Optional[int] = None
        self._pose_dim: Optional[int] = None

        # Build a minimal model instance to obtain static metadata.
        self._ensure_hand_model(batch_size=1)
        assert self._n_dofs is not None
        assert self._pose_dim is not None

    def __call__(
        self,
        hand_pose: Tensor,
        scene_pc: Optional[Tensor] = None,
        with_meshes: bool = False,
        with_surface_points: bool = False,
        with_penetration: bool = False,
        with_penetration_keypoints: bool = False,
        with_self_penetration: bool = False,
        with_contact_candidates: bool = False,
        with_distance: bool = False,
        with_fingertip_keypoints: bool = False,
        **extra_kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compute Shadow Hand geometry for batched multi-grasp poses.
        """

        hand_pose_flat, batch_size, num_grasps = self._flatten_hand_pose(hand_pose)
        batch_size_flat = hand_pose_flat.shape[0]
        self._ensure_hand_model(batch_size_flat)
        self._raw_model.update_kinematics(hand_pose_flat)

        outputs: Dict[str, Tensor] = {}
        scene_pc_flat = (
            self._prepare_scene_pc(scene_pc, batch_size, num_grasps)
            if scene_pc is not None
            else None
        )

        if extra_kwargs:
            logging.getLogger(__name__).warning(
                "ShadowHandModelAdapter: 忽略未支持的 hand_model 参数: %s",
                list(extra_kwargs.keys()),
            )

        if with_surface_points:
            outputs["surface_points"] = self._get_surface_points()

        if with_meshes:
            vertices, faces = self._get_mesh_vertices_and_faces()
            outputs["vertices"] = vertices
            outputs["faces"] = faces

        if with_penetration_keypoints:
            outputs["penetration_keypoints"] = self._get_penetration_keypoints()

        if with_contact_candidates:
            if scene_pc_flat is None:
                raise ValueError(
                    "scene_pc is required when with_contact_candidates=True."
                )
            outputs["contact_candidates_dis"] = self._compute_contact_distances(
                scene_pc_flat
            )

        if with_penetration:
            if scene_pc_flat is None:
                raise ValueError("scene_pc is required when with_penetration=True.")
            outputs["penetration"] = self._compute_penetration(scene_pc_flat)

        if with_distance:
            if scene_pc_flat is None:
                logging.getLogger(__name__).warning(
                    "ShadowHandModelAdapter: with_distance=True 但 scene_pc 缺失，返回空张量。"
                )
                outputs["distances"] = torch.zeros(
                    batch_size_flat, 0, device=self._device
                )
            else:
                outputs["distances"] = self._compute_penetration(scene_pc_flat)

        if with_fingertip_keypoints:
            outputs["fingertip_keypoints"] = self._get_fingertip_keypoints()

        if with_self_penetration:
            outputs["self_penetration"] = self._compute_self_penetration()

        return outputs

    def decompose_hand_pose(
        self, hand_pose: Tensor, rot_type: Optional[str] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Match HandModel API: split pose into translation / rotation matrix / joint angles.
        """
        pose = hand_pose
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
        elif pose.dim() == 3:
            pose = pose.reshape(-1, pose.shape[-1])
        elif pose.dim() != 2:
            raise ValueError(
                f"hand_pose must be 1D, 2D or 3D tensor, got {pose.dim()}D tensor with shape {pose.shape}."
            )

        pose = pose.to(self._device, dtype=torch.float32)
        current_rot_type = rot_type if rot_type is not None else self.rot_type
        return _decompose_hand_pose_helper(pose, current_rot_type)

    @property
    def n_dofs(self) -> int:
        if self._n_dofs is None:
            raise RuntimeError("Shadow hand model is not initialized.")
        return self._n_dofs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def chain(self):
        """Expose the kinematics chain for compatibility with evaluate_utils.py"""
        if self._raw_model is None:
            raise RuntimeError("Shadow hand model is not initialized. Call _ensure_hand_model first.")
        return self._raw_model.robot

    @property
    def mesh(self):
        """Expose mesh data in format compatible with evaluate_utils.py"""
        if self._raw_model is None:
            raise RuntimeError("Shadow hand model is not initialized. Call _ensure_hand_model first.")

        # Build mesh dict on-demand (cache it)
        if not hasattr(self, '_mesh_cache'):
            self._mesh_cache = {}
            for link_name in self._link_names:
                mesh_data = {}

                # Add surface_points if available
                if hasattr(self._raw_model, 'surface_points') and link_name in self._raw_model.surface_points:
                    pts = self._raw_model.surface_points[link_name]
                    if isinstance(pts, torch.Tensor):
                        if pts.dim() == 3:
                            pts_proc = pts[:1, ..., :3]
                        elif pts.dim() == 2:
                            pts_proc = pts.unsqueeze(0)[..., :3]
                        else:
                            pts_proc = torch.zeros(1, 0, 3, device=self._device)
                        mesh_data['surface_points'] = pts_proc.to(self._device).contiguous()
                    else:
                        mesh_data['surface_points'] = torch.zeros(1, 0, 3, device=self._device)
                else:
                    mesh_data['surface_points'] = torch.zeros(1, 0, 3, device=self._device)

                # Add contact_candidates if available
                if hasattr(self._raw_model, 'dis_key_point') and link_name in self._raw_model.dis_key_point:
                    points = self._raw_model.dis_key_point[link_name]
                    if points:
                        mesh_data['contact_candidates'] = torch.tensor(
                            points, device=self._device, dtype=torch.float32
                        ).reshape(-1, 3)
                    else:
                        mesh_data['contact_candidates'] = torch.empty(0, 3, device=self._device)
                else:
                    mesh_data['contact_candidates'] = torch.empty(0, 3, device=self._device)

                # Add penetration_keypoints (use dis_key_point as approximation)
                if hasattr(self._raw_model, 'dis_key_point') and link_name in self._raw_model.dis_key_point:
                    points = self._raw_model.dis_key_point[link_name]
                    if points:
                        mesh_data['penetration_keypoints'] = torch.tensor(
                            points, device=self._device, dtype=torch.float32
                        ).reshape(-1, 3)
                    else:
                        mesh_data['penetration_keypoints'] = torch.empty(0, 3, device=self._device)
                else:
                    mesh_data['penetration_keypoints'] = torch.empty(0, 3, device=self._device)

                # Add face_verts for penetration calculation
                if hasattr(self._raw_model, 'link_face_verts') and link_name in self._raw_model.link_face_verts:
                    mesh_data['face_verts'] = self._raw_model.link_face_verts[link_name]

                # Add radius if it's a primitive (not common for shadowhand)
                # Could add support later if needed

                self._mesh_cache[link_name] = mesh_data

        return self._mesh_cache

    def _ensure_hand_model(self, batch_size: int) -> None:
        if self._raw_model is not None and self._raw_model.batch_size == batch_size:
            return

        self._raw_model = get_handmodel(
            batch_size=batch_size,
            device=self._device,
            hand_scale=self.hand_scale,
            rot_type=self.rot_type,
            robot="shadowhand",
            anchor=self.anchor,
        )
        if hasattr(self, "_mesh_cache"):
            del self._mesh_cache
        self._link_names = [
            name
            for name in self._raw_model.mesh_verts.keys()
            if name not in self._EXCLUDED_VISUAL_LINKS
        ]

        if self._n_dofs is None:
            self._n_dofs = len(self._raw_model.robot.get_joint_parameter_names())
            self._pose_dim = 3 + self._n_dofs + self._rot_dim

    def _flatten_hand_pose(self, hand_pose: Tensor) -> Tuple[Tensor, int, int]:
        if hand_pose.dim() == 2:
            hand_pose = hand_pose.unsqueeze(1)
        if hand_pose.dim() != 3:
            raise ValueError(
                f"hand_pose must have shape [B, G, pose_dim], got {hand_pose.shape}."
            )

        batch_size, num_grasps, pose_dim = hand_pose.shape
        if self._pose_dim is None:
            raise RuntimeError("Pose dimension metadata is not initialized.")
        expected_full = self._pose_dim
        expected_compact = 3 + max(self._n_dofs - 2, 0) + self._rot_dim
        if pose_dim != expected_full and pose_dim != expected_compact:
            raise ValueError(
                f"hand_pose last dimension {pose_dim} does not match expected {expected_full} or {expected_compact}."
            )

        hand_pose = hand_pose.to(self._device, dtype=torch.float32)
        return hand_pose.reshape(-1, pose_dim), batch_size, num_grasps

    def _prepare_scene_pc(
        self, scene_pc: Optional[Tensor], batch_size: int, num_grasps: int
    ) -> Optional[Tensor]:
        if scene_pc is None:
            return None

        if scene_pc.ndim == 2:
            if batch_size * num_grasps != 1:
                raise ValueError(
                    "2D scene_pc can only be used when batch_size * num_grasps == 1."
                )
            scene_pc = scene_pc.unsqueeze(0)
        elif scene_pc.ndim == 3:
            if scene_pc.shape[0] == batch_size * num_grasps:
                pass
            elif scene_pc.shape[0] == batch_size:
                scene_pc = (
                    scene_pc.unsqueeze(1)
                    .expand(batch_size, num_grasps, *scene_pc.shape[1:])
                    .reshape(batch_size * num_grasps, *scene_pc.shape[1:])
                )
            else:
                raise ValueError(
                    f"scene_pc batch dimension {scene_pc.shape[0]} does not match "
                    f"batch_size={batch_size} or flattened batch size {batch_size * num_grasps}."
                )
        elif scene_pc.ndim == 4:
            if scene_pc.shape[0] != batch_size or scene_pc.shape[1] != num_grasps:
                raise ValueError(
                    f"scene_pc shape {scene_pc.shape[:2]} does not match (B, G)=({batch_size}, {num_grasps})."
                )
            scene_pc = scene_pc.reshape(
                batch_size * num_grasps, scene_pc.shape[2], scene_pc.shape[3]
            )
        else:
            raise ValueError(
                f"scene_pc must have ndim 2, 3 or 4, got {scene_pc.ndim}."
            )

        if scene_pc.shape[-1] < 3:
            raise ValueError(
                f"scene_pc last dimension must contain xyz coordinates, got {scene_pc.shape[-1]}."
            )

        return scene_pc.to(self._device, dtype=torch.float32)

    def _get_surface_points(self) -> Tensor:
        surface_points = self._raw_model.get_surface_points()
        if self.n_surface_points > 0 and surface_points.shape[1] > self.n_surface_points:
            idx = torch.linspace(
                0,
                surface_points.shape[1] - 1,
                steps=self.n_surface_points,
                dtype=torch.long,
                device=surface_points.device,
            )
            surface_points = surface_points.index_select(1, idx)
        return surface_points

    def _get_mesh_vertices_and_faces(self) -> Tuple[Tensor, Tensor]:
        vertices_per_link = []
        faces = []
        vertex_offset = 0

        for link_name in self._link_names:
            verts_np = self._raw_model.mesh_verts.get(link_name)
            faces_np = self._raw_model.mesh_faces.get(link_name)

            if verts_np is None or verts_np.size == 0:
                continue

            verts = torch.from_numpy(verts_np).to(self._device, dtype=torch.float32)
            verts_h = torch.cat(
                [verts, torch.ones_like(verts[:, :1])], dim=1
            ).unsqueeze(0)
            verts_h = verts_h.repeat(self._raw_model.batch_size, 1, 1)

            trans_matrix = self._raw_model.current_status[link_name].get_matrix()
            transformed = (
                torch.matmul(trans_matrix, verts_h.transpose(1, 2))
                .transpose(1, 2)[..., :3]
            )
            transformed = (
                torch.matmul(
                    self._raw_model.global_rotation, transformed.transpose(1, 2)
                ).transpose(1, 2)
                + self._raw_model.global_translation.unsqueeze(1)
            )
            transformed = transformed * self._raw_model.scale

            vertices_per_link.append(transformed)

            if faces_np is not None and faces_np.size > 0:
                faces_tensor = (
                    torch.from_numpy(faces_np)
                    .to(self._device, dtype=torch.long)
                    .clone()
                    + vertex_offset
                )
                faces.append(faces_tensor)

            vertex_offset += verts.shape[0]

        if vertices_per_link:
            vertices = torch.cat(vertices_per_link, dim=1)
        else:
            vertices = torch.empty(
                self._raw_model.batch_size, 0, 3, device=self._device
            )

        if faces:
            faces_out = torch.cat(faces, dim=0)
        else:
            faces_out = torch.empty(0, 3, dtype=torch.long, device=self._device)

        return vertices, faces_out

    def _compute_contact_distances(self, scene_pc: Tensor) -> Tensor:
        if not _PYTORCH3D_AVAILABLE:
            raise ImportError(
                "PyTorch3D (knn_points) is required for contact candidate distances."
            )

        scene_xyz = scene_pc[..., :3]
        contact_candidates = self._raw_model.get_dis_keypoints()
        if contact_candidates.shape[1] == 0:
            return torch.empty(
                scene_xyz.shape[0], scene_xyz.shape[1], device=self._device
            )

        dists_sq, _, _ = knn_points(scene_xyz, contact_candidates, K=1)
        return dists_sq.squeeze(-1)

    def _compute_penetration(self, scene_pc: Tensor) -> Tensor:
        """
        返回相对于手部表面的 signed distance：
        - 正值：点位于手内部（需要惩罚）
        - 负值：点位于手外部（用于 contact-map 等需要距离幅值的指标）
        """
        obj_pcd = scene_pc[..., :3].float()
        global_translation = self._raw_model.global_translation.float()
        global_rotation = self._raw_model.global_rotation.float()
        obj_in_hand = (obj_pcd - global_translation.unsqueeze(1)) @ global_rotation

        penetration_values = []
        for link_name, face_verts in self._raw_model.link_face_verts.items():
            trans_matrix = self._raw_model.current_status[link_name].get_matrix()
            obj_local = (
                obj_in_hand - trans_matrix[:, :3, 3].unsqueeze(1)
            ) @ trans_matrix[:, :3, :3]
            obj_local = obj_local.reshape(-1, 3)

            # dis_local, _, dis_signs, _, _ = compute_sdf(obj_local, face_verts.detach())
            dis_local, dis_signs, _, _ = compute_sdf(obj_local, face_verts.detach())
            dis_local = torch.sqrt(dis_local + 1e-8)
            pen_map = dis_local * (-dis_signs)
            penetration_values.append(
                pen_map.reshape(obj_in_hand.shape[0], obj_in_hand.shape[1])
            )

        if not penetration_values:
            return torch.zeros(
                obj_in_hand.shape[0], obj_in_hand.shape[1], device=self._device
            )

        signed_penetration = torch.stack(penetration_values, dim=0).max(dim=0).values
        return signed_penetration

    def _get_fingertip_keypoints(self) -> Tensor:
        fingertip_points = []
        for link_name in self._FINGERTIP_LINKS:
            if link_name not in self._raw_model.current_status:
                continue
            matrix = self._raw_model.current_status[link_name].get_matrix()
            origin = matrix[:, :3, 3]
            fingertip_points.append(origin.unsqueeze(1))

        if not fingertip_points:
            return torch.empty(self._raw_model.batch_size, 0, 3, device=self._device)

        origins = torch.cat(fingertip_points, dim=1)
        origins_world = (
            torch.matmul(
                self._raw_model.global_rotation, origins.transpose(1, 2)
            ).transpose(1, 2)
            + self._raw_model.global_translation.unsqueeze(1)
        )
        origins_world = origins_world * self._raw_model.scale
        return origins_world

    def _get_penetration_keypoints(self) -> Tensor:
        if hasattr(self._raw_model, "get_penetration_keypoints"):
            keypoints = self._raw_model.get_penetration_keypoints()
        elif hasattr(self._raw_model, "get_dis_keypoints"):
            keypoints = self._raw_model.get_dis_keypoints()
            # logging.getLogger(__name__).info(
            #     "ShadowHandModelAdapter: 使用接触关键点近似 penetration_keypoints。"
            # )
        else:
            logging.getLogger(__name__).warning(
                "ShadowHandModelAdapter: 无 penetration_keypoints 信息，返回空张量。"
            )
            keypoints = torch.empty(self._raw_model.batch_size, 0, 3, device=self._device)
        return keypoints

    def to(self, device):
        """Move the adapter and its underlying model to the specified device."""
        self._device = torch.device(device)

        # Move the raw model if it exists
        if self._raw_model is not None:
            # Move robot chain
            if hasattr(self._raw_model, 'robot') and self._raw_model.robot is not None:
                self._raw_model.robot = self._raw_model.robot.to(device=device)

            # Move mesh data
            if hasattr(self._raw_model, 'mesh_verts'):
                for link_name, mesh_data in self._raw_model.mesh_verts.items():
                    if isinstance(mesh_data, np.ndarray):
                        # numpy arrays don't need device movement
                        continue
                    elif isinstance(mesh_data, torch.Tensor):
                        self._raw_model.mesh_verts[link_name] = mesh_data.to(device)

            # Move other tensor attributes
            for attr_name in ['global_translation', 'global_rotation', 'joints_lower', 'joints_upper']:
                if hasattr(self._raw_model, attr_name):
                    attr_value = getattr(self._raw_model, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(self._raw_model, attr_name, attr_value.to(device))

        return self

    def _compute_self_penetration(self) -> Tensor:
        """Calculate self-penetration energy using keypoints."""
        # Get keypoints from the raw model
        keypoints = self._raw_model.get_keypoints()  # [B, N, 3]

        if keypoints.shape[1] == 0:
            return torch.zeros(keypoints.shape[0], device=self._device)

        # Calculate pairwise distances
        dis = torch.cdist(keypoints, keypoints, p=2)
        # Ignore self-distance by setting diagonal to large value
        batch_size, n_points = keypoints.shape[:2]
        dis.diagonal(dim1=1, dim2=2).fill_(1e6)

        # Calculate self-penetration energy
        spen = SELF_PENETRATION_POINT_RADIUS * 2 - dis
        E_spen = torch.where(spen > 0, spen, torch.zeros_like(spen))

        # Sum and divide by 2 (each pair counted twice)
        return E_spen.sum((1, 2)) / 2

    @staticmethod
    def _rotation_dim(rot_type: str) -> int:
        if rot_type == "quat":
            return 4
        if rot_type == "r6d":
            return 6
        raise ValueError(f"Unsupported rot_type '{rot_type}'. Expected 'quat' or 'r6d'.")
