import json
import os
import random
import warnings
import typing
from typing import Optional, List, Dict, Tuple, Union, Any

import mujoco
import numpy as np
import torch
import torch.nn
import trimesh
import trimesh.sample
import transforms3d
import pytorch_kinematics as pk
import urdf_parser_py.urdf as URDF_PARSER
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Box, Cylinder, Mesh, Sphere
from pytorch3d.ops import knn_points
from plotly import graph_objects as go
from kaolin.metrics.trianglemesh import (
    compute_sdf,
    CUSTOM_index_vertices_by_faces as index_vertices_by_faces,
)

# 假设这些本地模块存在于你的项目中
from .rotation_spec import get_rotation_spec
from .rot6d import *

# --- Configuration & Constants ---

ANCHOR_ALIASES = {'base_rz180_offset': 'mjcf'}
MJCF_ANCHOR_TRANSLATION = torch.tensor([0.0, -0.01, 0.213], dtype=torch.float32)
PALM_CENTER_ANCHOR_TRANSLATION = torch.tensor([0.008, -0.013, 0.283], dtype=torch.float32)

class HandModel:
    """
    A unified, differentiable interface for robotic hand kinematics, geometry, 
    and collision handling. Supports both URDF and MuJoCo based mesh loading.
    """
    SUPPORTED_ANCHORS = ('base', 'palm_center', 'mjcf')

    def __init__(
        self,
        robot_name: str,
        urdf_filename: str,
        mesh_path: str,
        batch_size: int = 1,
        device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu',
        hand_scale: float = 1.0,
        rot_type: str = 'quat',
        anchor: str = 'base',
        mesh_source: str = 'urdf'
    ):
        self.robot_name = robot_name
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.scale = hand_scale
        self.mesh_source = mesh_source
        self.rot_type = rot_type
        
        # Initialize Random Seed for Determinism
        np.random.seed(0)
        random.seed(0)

        # 1. Setup Rotation Specification
        if self.rot_type not in ['quat', 'r6d', 'euler', 'axis']:
            raise ValueError(f"Unsupported rot_type: {self.rot_type}")
        self.rot_spec = get_rotation_spec(rot_type)
        self.anchor = self._normalize_anchor(anchor)

        # 2. Build Kinematics Chain
        with open(urdf_filename, 'r') as f:
            urdf_str = f.read()
        self.robot = pk.build_chain_from_urdf(urdf_str).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_string(urdf_str)

        # 3. Initialize Geometry Containers
        self.surface_points: Dict[str, torch.Tensor] = {}
        self.surface_points_normal: Dict[str, torch.Tensor] = {}
        self.mesh_verts: Dict[str, np.ndarray] = {}
        self.mesh_faces: Dict[str, np.ndarray] = {}
        self.link_face_verts: Dict[str, torch.Tensor] = {}
        self.collision_mesh_verts: Dict[str, np.ndarray] = {}
        self.collision_mesh_faces: Dict[str, np.ndarray] = {}
        
        # 4. Load Meshes based on Source
        if mesh_source == 'urdf':
            self._load_urdf_geometry(urdf_filename, mesh_path)
            self._load_urdf_collision_geometry(urdf_filename, mesh_path)
        elif mesh_source == 'mujoco':
            self._load_mujoco_geometry()
        else:
            raise ValueError(f"Unknown mesh_source: {mesh_source}")

        # 5. Initialize Joint Limits & Keypoints
        self._init_joint_limits()
        self._init_keypoints()

        # 6. Global State Containers
        self.global_translation: Optional[torch.Tensor] = None
        self.global_rotation: Optional[torch.Tensor] = None
        self.current_status: Optional[Dict[str, pk.Transform3d]] = None
        self.current_joint_angles_24: Optional[torch.Tensor] = None
        
        # Shadowhand specific vector
        if robot_name == 'shadowhand':
            self.palm_toward = torch.tensor([0., -1., 0., 0.], device=self.device).reshape(1, 1, 4).expand(batch_size, -1, -1)
        else:
            # Placeholder or implementation for other hands
            self.palm_toward = None 

    # --- Initialization Helpers ---

    def _normalize_anchor(self, anchor: str) -> str:
        if not isinstance(anchor, str):
            raise TypeError(f"Anchor must be str, got {type(anchor)}")
        normalized = anchor.strip().lower()
        alias = ANCHOR_ALIASES.get(normalized, normalized)
        if normalized in ANCHOR_ALIASES:
            warnings.warn(f"Anchor '{normalized}' is deprecated, using '{alias}'.", UserWarning)
        if alias not in self.SUPPORTED_ANCHORS:
            raise ValueError(f"Anchor must be one of {self.SUPPORTED_ANCHORS}")
        return alias

    def _init_joint_limits(self):
        """Extracts and tensorizes joint limits for loss computation."""
        revolute_joints = [j for j in self.robot_full.joints if j.joint_type == 'revolute']
        
        # Map URDF joints to PyTorch Kinematics order
        pk_joint_names = self.robot.get_joint_parameter_names()
        q_lower, q_upper = [], []

        for name in pk_joint_names:
            joint = next((j for j in revolute_joints if j.name == name), None)
            if joint:
                q_lower.append(joint.limit.lower)
                q_upper.append(joint.limit.upper)
            else:
                # Fallback for non-revolute or missing limits
                q_lower.append(-np.pi) 
                q_upper.append(np.pi)

        self.revolute_joints_q_lower = torch.tensor(q_lower, device=self.device).repeat(self.batch_size, 1)
        self.revolute_joints_q_upper = torch.tensor(q_upper, device=self.device).repeat(self.batch_size, 1)

    def _init_keypoints(self):
        """Initializes keypoint dictionaries from static configurations."""
        self.keypoints = _get_default_keypoints()
        
        # Load joint keypoints based on robot name
        all_links = [lk.name for lk in getattr(self.robot_full, 'links', [])]
        self.joint_key_points = {k: [] for k in all_links}
        
        if self.robot_name == 'shadowhand':
            static_kps = _get_static_shadowhand_keypoints()
            for k, v in static_kps.items():
                if k in self.joint_key_points:
                    self.joint_key_points[k] = [list(p) for p in v]

        # Load SDF/Distance keypoints
        self.dis_key_point = _get_distance_keypoints()

    # --- Geometry Loading ---

    def _load_urdf_geometry(self, urdf_filename: str, mesh_path: str):
        visual_urdf = URDF.from_xml_string(open(urdf_filename).read())
        
        for link in visual_urdf.links:
            if not link.visuals:
                continue
            
            visual = link.visuals[0]
            geom = visual.geometry
            
            # 1. Load Mesh/Primitive
            mesh = self._load_trimesh_geometry(geom, mesh_path)
            if mesh is None: 
                continue

            # 2. Parse Transform & Scale
            try:
                scale = np.array(geom.scale).reshape(1, 3) if hasattr(geom, 'scale') and geom.scale else np.array([[1, 1, 1]])
                rotation = transforms3d.euler.euler2mat(*visual.origin.rpy)
                translation = np.reshape(visual.origin.xyz, [1, 3])
            except (AttributeError, TypeError):
                scale = np.array([[1, 1, 1]])
                rotation = np.eye(3)
                translation = np.array([[0, 0, 0]])

            # 3. Process & Sample
            self._process_and_store_mesh(link.name, mesh, scale, rotation, translation)

    def _load_urdf_collision_geometry(self, urdf_filename: str, mesh_path: str):
        collision_urdf = URDF.from_xml_string(open(urdf_filename).read())
        
        for link in collision_urdf.links:
            if not getattr(link, 'collisions', None):
                continue

            collision = link.collisions[0]
            geom = collision.geometry

            mesh = self._load_trimesh_geometry(geom, mesh_path)
            if mesh is None:
                continue

            try:
                scale = np.array(geom.scale).reshape(1, 3) if hasattr(geom, 'scale') and geom.scale else np.array([[1, 1, 1]])
                rotation = transforms3d.euler.euler2mat(*collision.origin.rpy)
                translation = np.reshape(collision.origin.xyz, [1, 3])
            except (AttributeError, TypeError):
                scale = np.array([[1, 1, 1]])
                rotation = np.eye(3)
                translation = np.array([[0, 0, 0]])

            self._process_and_store_collision_mesh(link.name, mesh, scale, rotation, translation)

    def _load_trimesh_geometry(self, geometry, mesh_path: str) -> Optional[trimesh.Trimesh]:
        if isinstance(geometry, Mesh):
            filename = geometry.filename
            # Handle specific path quirks for different robots
            if self.robot_name == 'allegro' and '/' in filename:
                parts = filename.split('/')
                filename = f"{parts[-2]}/{parts[-1]}"
            elif self.robot_name in ['shadowhand', 'barrett']:
                filename = filename.split('/')[-1]
            
            full_path = os.path.join(mesh_path, filename)
            if not os.path.exists(full_path):
                return None
            return trimesh.load(full_path, force='mesh', process=False)
        
        elif isinstance(geometry, Cylinder):
            return trimesh.primitives.Cylinder(radius=geometry.radius, height=geometry.length)
        elif isinstance(geometry, Box):
            return trimesh.primitives.Box(extents=geometry.size)
        elif isinstance(geometry, Sphere):
            return trimesh.primitives.Sphere(radius=geometry.radius)
        return None

    def _load_mujoco_geometry(self):
        if self.robot_name != 'shadowhand':
            raise NotImplementedError("MuJoCo mesh source currently supports 'shadowhand' only")

        # Compile MuJoCo model to access internal mesh data
        xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets/hand/shadow/right_hand.xml')
        spec = mujoco.MjSpec()
        spec.from_file(xml_path)
        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        # Aggregate geometry by body
        body_to_verts = {}
        body_to_faces = {}

        for i in range(mj_model.ngeom):
            geom_id = i
            mesh_id = mj_model.geom(geom_id).dataid
            if mesh_id == -1: continue

            # Extract Raw Mesh
            mjm = mj_model.mesh(mesh_id)
            vert_adr, vert_num = mjm.vertadr[0], mjm.vertnum[0]
            face_adr, face_num = mjm.faceadr[0], mjm.facenum[0]
            
            verts = mj_model.mesh_vert[vert_adr : vert_adr + vert_num]
            faces = mj_model.mesh_face[face_adr : face_adr + face_num]

            # Transform to Body Frame
            geom_xmat = mj_data.geom_xmat[geom_id].reshape(3, 3)
            geom_xpos = mj_data.geom_xpos[geom_id]
            body_id = mj_model.geom(geom_id).bodyid
            body_name = mj_model.body(body_id).name
            body_xmat = mj_data.xmat[body_id].reshape(3, 3)
            body_xpos = mj_data.xpos[body_id]

            # V_body = R_body^T * ( (V_local * R_geom^T + T_geom) - T_body )
            v_world = verts @ geom_xmat.T + geom_xpos
            v_body = (v_world - body_xpos) @ body_xmat

            if body_name not in body_to_verts:
                body_to_verts[body_name] = []
                body_to_faces[body_name] = []

            # Offset faces
            current_offset = sum(len(v) for v in body_to_verts[body_name])
            body_to_verts[body_name].append(v_body)
            body_to_faces[body_name].append(faces + current_offset)

        # Store
        for name in body_to_verts:
            if not body_to_verts[name]: continue
            all_verts = np.concatenate(body_to_verts[name], axis=0)
            all_faces = np.concatenate(body_to_faces[name], axis=0)
            
            mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
            # MuJoCo meshes are already in body frame, so no extra transform needed here
            self._process_and_store_mesh(name, mesh, np.ones((1,3)), np.eye(3), np.zeros((1,3)))

    def _process_and_store_mesh(self, name: str, mesh: trimesh.Trimesh, scale, rotation, translation):
        # Sampling settings
        sample_count = 64 if self.robot_name == 'shadowhand' else 128
        
        try:
            pts, face_idx = trimesh.sample.sample_surface(mesh, count=sample_count)
            normals = np.array([mesh.face_normals[x] for x in face_idx], dtype=float)
        except Exception:
            # Fallback for degenerate meshes
            pts = np.zeros((0, 3))
            normals = np.zeros((0, 3))

        # Special handling for volumetric sampling (Grippers)
        if self.robot_name == 'barrett' and name == 'bh_base_link':
            pts = trimesh.sample.volume_mesh(mesh, count=1024)
            normals = np.tile([0., 0., 1.], (len(pts), 1))
        # ... (Include other gripper specific logic here if needed)

        # Apply Scaling
        pts *= scale
        mesh_verts = np.array(mesh.vertices) * scale

        # ShadowHand Specific Flip (CRITICAL)
        if self.robot_name == 'shadowhand':
            # Swap Y/Z and invert Y
            pts = pts[:, [0, 2, 1]]
            normals = normals[:, [0, 2, 1]]
            pts[:, 1] *= -1
            normals[:, 1] *= -1
            
            mesh_verts = mesh_verts[:, [0, 2, 1]]
            mesh_verts[:, 1] *= -1

        # Apply Local Transform (Origin offset)
        pts = (rotation @ pts.T).T + translation
        mesh_verts = (rotation @ mesh_verts.T).T + translation
        # Normals rotate but don't translate
        # normals = (rotation @ normals.T).T 

        # Store as Tensors
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        nor_h = np.column_stack([normals, np.ones(len(normals))])
        
        self.surface_points[name] = torch.from_numpy(pts_h).float().to(self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.surface_points_normal[name] = torch.from_numpy(nor_h).float().to(self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        self.mesh_verts[name] = mesh_verts
        self.mesh_faces[name] = np.array(mesh.faces)

        # Store Face Vertices for SDF
        t_verts = torch.tensor(mesh_verts, dtype=torch.float, device=self.device)
        t_faces = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
        if len(t_verts) > 0 and len(t_faces) > 0:
            self.link_face_verts[name] = index_vertices_by_faces(t_verts, t_faces)
        else:
            self.link_face_verts[name] = torch.empty((0, 3, 3), device=self.device)

    def _process_and_store_collision_mesh(self, name: str, mesh: trimesh.Trimesh, scale, rotation, translation):
        mesh_verts = np.array(mesh.vertices) * scale

        if self.robot_name == 'shadowhand':
            mesh_verts = mesh_verts[:, [0, 2, 1]]
            mesh_verts[:, 1] *= -1

        mesh_verts = (rotation @ mesh_verts.T).T + translation

        if len(mesh_verts) == 0 or len(mesh.faces) == 0:
            return

        self.collision_mesh_verts[name] = mesh_verts
        self.collision_mesh_faces[name] = np.array(mesh.faces)

    # --- Kinematics Core ---

    def _map_to_current_status_name(self, name: str) -> Optional[str]:
        """Resolves mesh names to kinematic link names robustly."""
        if not self.current_status:
            return None
        keys = list(self.current_status.keys())
        if name in keys: return name
        
        # Fuzzy matching
        def clean(s): 
            return s.lower().replace("rh_", "").replace("lh_", "").replace("right_", "").replace("left_", "").replace("_", "")
        
        norm_keys = {clean(k): k for k in keys}
        target = clean(name)
        
        if target in norm_keys: return norm_keys[target]
        if "_" in name: # Try removing prefix
             suffix = clean(name.split("_", 1)[1])
             if suffix in norm_keys: return norm_keys[suffix]
        return None

    def update_kinematics(self, q: torch.Tensor):
        """
        Updates the internal kinematic state of the hand.
        
        Args:
            q: Input tensor [B, dim]. 
               Dimensions supported:
               - 24-DoF joints (URDF): 3(trans) + 24(joints) + rot_dim
               - 22-DoF joints (MJCF): 3(trans) + 22(joints) + rot_dim
        """
        if q.dim() == 1: q = q.unsqueeze(0)
        q = q.to(device=self.device, dtype=torch.float32)
        
        # Determine Rot Dim
        rot_dim_map = {'quat': 4, 'r6d': 6, 'euler': 3, 'axis': 3}
        rot_dim = rot_dim_map[self.rot_type]
        
        dim_24 = 3 + 24 + rot_dim
        dim_22 = 3 + 22 + rot_dim

        # Parse Input
        t_input = q[:, :3]
        
        if q.shape[1] == dim_24:
            joint_angles = q[:, 3:27]
            rot_params = q[:, 27:]
        elif q.shape[1] == dim_22:
            joints_22 = q[:, 3:25]
            rot_params = q[:, 25:]
            
            # Map 22 DoF (MJCF) to 24 DoF (URDF)
            # URDF Order: WRJ2, WRJ1, FF(4-1), MF(4-1), RF(4-1), LF(5-1), TH(5-1)
            joint_angles = torch.zeros((q.shape[0], 24), device=self.device, dtype=q.dtype)
            # Wrist locked at 0 for MJCF mapping usually
            joint_angles[:, 2:6]   = joints_22[:, 0:4]    # FF
            joint_angles[:, 6:10]  = joints_22[:, 4:8]    # MF
            joint_angles[:, 10:14] = joints_22[:, 8:12]   # RF
            joint_angles[:, 14:19] = joints_22[:, 12:17]  # LF
            joint_angles[:, 19:24] = joints_22[:, 17:22]  # TH
        else:
            raise ValueError(f"Invalid input dimension {q.shape[1]}. Expected {dim_24} or {dim_22}.")

        # Normalize & Convert Rotation
        if self.rot_spec.needs_normalization:
            rot_params = self.rot_spec.normalize_fn(rot_params)
        R_base = self.rot_spec.to_matrix_fn(rot_params) # [B, 3, 3]

        # Forward Kinematics
        self.current_joint_angles_24 = joint_angles
        self.current_status = self.robot.forward_kinematics(joint_angles)
        
        # Apply Anchor Transform
        self._apply_anchor_transform(t_input, R_base)

    def _apply_anchor_transform(self, t_input: torch.Tensor, R_base: torch.Tensor):
        if self.anchor == 'base':
            self.global_translation = t_input
            self.global_rotation = R_base
        elif self.anchor == 'palm_center':
            # Calculate offset: R * offset
            offset = PALM_CENTER_ANCHOR_TRANSLATION.to(self.device).unsqueeze(0)
            base_offset = (R_base @ offset.unsqueeze(-1)).squeeze(-1)
            self.global_translation = t_input - base_offset
            self.global_rotation = R_base
        elif self.anchor == 'mjcf':
            offset = MJCF_ANCHOR_TRANSLATION.to(self.device).unsqueeze(0)
            base_offset = (R_base @ offset.unsqueeze(-1)).squeeze(-1)
            self.global_translation = t_input - base_offset
            self.global_rotation = R_base

    # --- Point Query & Utilities ---

    def _transform_points(self, link_name: str, points_local_h: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Helper: Transforms homogeneous local points [B, N, 4] to global world space [B, N, 3].
        Applies: Kinematics -> Global Rotation -> Global Translation -> Scale.
        """
        mapped_name = self._map_to_current_status_name(link_name)
        if not mapped_name or mapped_name not in self.current_status:
            return None
        
        # 1. Local Link -> Hand Base
        transform = self.current_status[mapped_name].get_matrix() # [B, 4, 4]
        pts_hand = (transform @ points_local_h.transpose(1, 2)).transpose(1, 2)[..., :3]
        
        # 2. Hand Base -> World
        pts_world = (self.global_rotation @ pts_hand.transpose(1, 2)).transpose(1, 2)
        pts_world = pts_world + self.global_translation.unsqueeze(1)
        
        return pts_world * self.scale

    def get_surface_points(self, q=None) -> torch.Tensor:
        if q is not None: self.update_kinematics(q)
        points_list = []
        
        for name, pts_h in self.surface_points.items():
            mapped = self._map_to_current_status_name(name)
            if not mapped or mapped == 'forearm': continue
            
            pts = self._transform_points(name, pts_h)
            if pts is not None:
                points_list.append(pts)
                
        return torch.cat(points_list, dim=1)

    def get_surface_points_and_normals(self, q=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if q is not None: self.update_kinematics(q)
        pts_list, nor_list = [], []
        
        for name in self.surface_points:
            mapped = self._map_to_current_status_name(name)
            if not mapped: continue

            # Points
            pts = self._transform_points(name, self.surface_points[name])
            
            # Normals (Rotation only)
            transform = self.current_status[mapped].get_matrix()
            nor_local = self.surface_points_normal[name]
            nor_hand = (transform @ nor_local.transpose(1, 2)).transpose(1, 2)[..., :3]
            nor_world = (self.global_rotation @ nor_hand.transpose(1, 2)).transpose(1, 2)
            
            if pts is not None:
                pts_list.append(pts)
                nor_list.append(nor_world)
                
        return torch.cat(pts_list, dim=1), torch.cat(nor_list, dim=1)

    def get_palm_points(self, q=None) -> torch.Tensor:
        if q is not None: self.update_kinematics(q)
        # Find palm link
        palm_name = next((k for k in self.surface_points if 'palm' in k.lower() and 'metacarpal' not in k.lower()), None)
        if not palm_name: return torch.empty(self.batch_size, 0, 3, device=self.device)
        
        return self._transform_points(palm_name, self.surface_points[palm_name])

    def get_palm_center_and_toward(self, q=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if q is not None: self.update_kinematics(q)
        
        palm_pts = self.get_palm_points()
        center = torch.mean(palm_pts, dim=1)
        
        # Toward vector
        palm_name = 'palm'
        mapped = self._map_to_current_status_name(palm_name) or palm_name
        
        if self.palm_toward is not None and mapped in self.current_status:
            # Transform the toward vector (which is stored as a "point" relative to origin)
            # Use logic similar to _transform_points but simplified
            trans = self.current_status[mapped].get_matrix()
            vec_local = (trans @ self.palm_toward.transpose(1, 2)).transpose(1, 2)[..., :3]
            vec_world = (self.global_rotation @ vec_local.transpose(1, 2)).transpose(1, 2)
            toward = vec_world.squeeze(1)
        else:
            toward = torch.zeros_like(center)
            
        return center, toward

    def get_keypoints(self, q=None) -> torch.Tensor:
        return self._get_points_from_dict(self.keypoints, q)

    def get_joint_keypoints(self, q=None) -> torch.Tensor:
        return self._get_points_from_dict(self.joint_key_points, q)
    
    def get_joint_keypoints_unique(self, q=None, eps: float = 1e-5):
        if q is not None:
            self.update_kinematics(q)
        all_pts = []
        link_offsets = {}
        cursor = 0
        for link_name in self.joint_key_points:
            pts_local = self.joint_key_points[link_name]
            if len(pts_local) == 0:
                continue
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                mapped_name = link_name
            if mapped_name not in self.current_status:
                continue
            kp = self.current_status[mapped_name].transform_points(
                torch.tensor(pts_local, device=self.device, dtype=torch.float32)
            ).expand(self.batch_size, -1, -1)
            all_pts.append(kp)
            link_offsets[link_name] = (cursor, len(pts_local))
            cursor += len(pts_local)
        if len(all_pts) == 0:
            return (
                torch.zeros((self.batch_size, 0, 3), device=self.device, dtype=torch.float32),
                {},
            )
        pts = torch.cat(all_pts, dim=1)
        pts = torch.bmm(pts, self.global_rotation.transpose(1, 2)) + self.global_translation.unsqueeze(1)
        pts = pts * self.scale
        if self.batch_size != 1 or pts.shape[1] == 0:
            return pts, {k: [] for k in self.joint_key_points.keys()}
        p = pts[0]
        if eps is not None and eps > 0:
            qk = torch.round(p / eps).to(torch.int64).detach().cpu().numpy()
        else:
            qk = p.detach().cpu().numpy()
        uniq_pts = []
        uniq_map = {}
        orig_to_uniq = []
        for i in range(qk.shape[0]):
            key = (int(qk[i, 0]), int(qk[i, 1]), int(qk[i, 2]))
            if key in uniq_map:
                orig_to_uniq.append(uniq_map[key])
            else:
                idx_u = len(uniq_pts)
                uniq_map[key] = idx_u
                uniq_pts.append(p[i].unsqueeze(0))
                orig_to_uniq.append(idx_u)
        if len(uniq_pts) == 0:
            xyz_u = torch.zeros((1, 0, 3), device=self.device, dtype=torch.float32)
        else:
            xyz_u = torch.cat(uniq_pts, dim=0).unsqueeze(0)
        link_to_unique = {}
        for link_name, (st, ln) in link_offsets.items():
            if ln == 0:
                link_to_unique[link_name] = []
                continue
            idxs = []
            for k in range(ln):
                idxs.append(int(orig_to_uniq[st + k]))
            link_to_unique[link_name] = idxs
        return xyz_u, link_to_unique
    
    def get_dis_keypoints(self, q=None) -> torch.Tensor:
        return self._get_points_from_dict(self.dis_key_point, q)

    def _get_points_from_dict(self, kp_dict: Dict[str, List], q=None) -> torch.Tensor:
        if q is not None: self.update_kinematics(q)
        results = []
        for name, kps in kp_dict.items():
            if not kps: continue
            
            # Convert list to homogeneous tensor [B, N, 4]
            kps_tensor = torch.tensor(kps, device=self.device, dtype=torch.float32)
            kps_h = torch.cat([kps_tensor, torch.ones((len(kps), 1), device=self.device)], dim=1)
            kps_h = kps_h.unsqueeze(0).repeat(self.batch_size, 1, 1)
            
            transformed = self._transform_points(name, kps_h)
            if transformed is not None:
                results.append(transformed)
        
        if not results:
            return torch.zeros((self.batch_size, 0, 3), device=self.device)
        return torch.cat(results, dim=1)

    # --- Keypoint Management ---

    def add_keypoints(self, link_name: str, points: List[List[float]]):
        if link_name not in self.keypoints:
            self.keypoints[link_name] = []
        if not points: return
        # Ensure list of lists
        pts = [list(p) for p in (points if isinstance(points[0], (list, tuple)) else [points])]
        self.keypoints[link_name].extend(pts)

    def set_keypoints(self, link_name: str, points: List[List[float]]):
        if not points: 
            self.keypoints[link_name] = []
            return
        pts = [list(p) for p in (points if isinstance(points[0], (list, tuple)) else [points])]
        self.keypoints[link_name] = pts
        
    def save_joint_keypoints_to_file(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump({'robot_name': self.robot_name, 'joint_key_points': self.joint_key_points}, f, indent=2)

    # --- Physics & Loss Utilities ---

    def pen_loss_sdf(self, obj_pcd: torch.Tensor, q=None, test=False) -> torch.Tensor:
        """Computes penetration loss using SDF between object point cloud and hand meshes."""
        if q is not None: self.update_kinematics(q)
        
        # Transform object to hand base frame
        obj_pcd = obj_pcd.float()
        obj_local = (obj_pcd - self.global_translation.unsqueeze(1)) @ self.global_rotation
        
        penetrations = []
        
        for link_name, hand_face_verts in self.link_face_verts.items():
            mapped = self._map_to_current_status_name(link_name)
            if not mapped: continue
            
            # Object in Link Frame
            trans = self.current_status[mapped].get_matrix()
            # Inverse transform roughly: R.T * (P - T)
            # Using PyTorch kinematics matrix inverse might be safer but manual is fast for rigid
            R_link = trans[:, :3, :3]
            T_link = trans[:, :3, 3]
            
            # P_link = (P_base - T_link) @ R_link
            obj_in_link = (obj_local - T_link.unsqueeze(1)) @ R_link 
            obj_flat = obj_in_link.reshape(-1, 3)
            
            # Compute SDF
            # dis_local, dis_signs, _, _ = compute_sdf(obj_flat, hand_face_verts)
            # Note: Kaolin signature varies by version. Using code from input logic.
            dis_local, dis_signs, _, _ = compute_sdf(obj_flat, hand_face_verts)
            
            dis_local = torch.sqrt(dis_local + 1e-8)
            pen = dis_local * (-dis_signs)
            penetrations.append(pen.reshape(obj_pcd.shape[0], obj_pcd.shape[1]))

        if not penetrations:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        penetration_stack = torch.stack(penetrations) # [N_links, B, N_pts]
        
        if test:
            max_pen = torch.max(penetration_stack, dim=0)[0]
            max_pen[max_pen <= 0] = 0
            return max_pen.max(dim=1).values.mean()
        
        # Training Loss
        loss = penetration_stack[penetration_stack > 0].sum() / (obj_pcd.shape[0] * obj_pcd.shape[1])
        return loss

    def cal_joint_limit_energy(self) -> torch.Tensor:
        """Penalizes joint violations."""
        if self.current_joint_angles_24 is None: return torch.zeros(self.batch_size, device=self.device)
        
        q = self.current_joint_angles_24
        # Crop to min length if mismatch (handling MJCF mapping artifacts)
        L = min(q.shape[1], self.revolute_joints_q_lower.shape[1])
        
        qv = q[:, :L]
        lv = self.revolute_joints_q_lower[:, :L]
        uv = self.revolute_joints_q_upper[:, :L]
        
        violation = torch.relu(lv - qv) + torch.relu(qv - uv)
        return violation.mean(dim=1)

    # --- Visualization ---

    def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.0):
        if q is not None: self.update_kinematics(q)
        data = []
        
        for link_name, v_local in self.mesh_verts.items():
            mapped = self._map_to_current_status_name(link_name)
            if not mapped or mapped == 'forearm': continue
            
            # Manual transform for specific index i
            trans = self.current_status[mapped].get_matrix()[min(self.batch_size-1, i)].cpu().numpy()
            R_glob = self.global_rotation[min(self.batch_size-1, i)].cpu().numpy()
            T_glob = self.global_translation[min(self.batch_size-1, i)].cpu().numpy()
            
            # Local -> Hand Base
            v_h = np.concatenate([v_local, np.ones((len(v_local), 1))], axis=1)
            v_base = (trans @ v_h.T).T[..., :3]
            
            # Hand Base -> World
            v_world = (R_glob @ v_base.T).T + T_glob
            v_world *= self.scale
            
            f = self.mesh_faces[link_name]
            
            data.append(go.Mesh3d(
                x=v_world[:, 0], y=v_world[:, 1], z=v_world[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=color, opacity=opacity, name=link_name
            ))
        return data

    def get_plotly_data_collision_geom(self, q=None, i=0, color='red', opacity=0.5):
        if q is not None:
            self.update_kinematics(q)
        data = []

        if not self.collision_mesh_verts:
            return data

        for link_name, v_local in self.collision_mesh_verts.items():
            mapped = self._map_to_current_status_name(link_name)
            if not mapped or mapped == 'forearm':
                continue

            trans = self.current_status[mapped].get_matrix()[min(self.batch_size - 1, i)].cpu().numpy()
            R_glob = self.global_rotation[min(self.batch_size - 1, i)].cpu().numpy()
            T_glob = self.global_translation[min(self.batch_size - 1, i)].cpu().numpy()

            v_h = np.concatenate([v_local, np.ones((len(v_local), 1))], axis=1)
            v_base = (trans @ v_h.T).T[..., :3]

            v_world = (R_glob @ v_base.T).T + T_glob
            v_world *= self.scale

            f = self.collision_mesh_faces[link_name]

            data.append(go.Mesh3d(
                x=v_world[:, 0], y=v_world[:, 1], z=v_world[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=color, opacity=opacity, name=f"collision_{link_name}"
            ))

        return data


# --- Factory Function ---

def get_handmodel(
    batch_size: int, 
    device: str, 
    hand_scale: float = 1., 
    robot: str = 'shadowhand', 
    rot_type: str = 'quat', 
    anchor: str = 'base', 
    mesh_source: str = 'urdf'
) -> HandModel:
    
    meta_path = "assets/urdf/urdf_assets_meta.json"
    if not os.path.exists(meta_path):
        # Fallback for relative paths or assume calling from root
        meta_path = os.path.join(os.path.dirname(__file__), meta_path)
        
    urdf_assets_meta = json.load(open(meta_path))
    urdf_path = urdf_assets_meta['urdf_path'][robot]
    meshes_path = urdf_assets_meta['meshes_path'][robot]
    
    return HandModel(
        robot, urdf_path, meshes_path, 
        batch_size=batch_size, device=device, 
        hand_scale=hand_scale, rot_type=rot_type, 
        anchor=anchor, mesh_source=mesh_source
    )


# --- Standalone Loss Functions ---

def compute_collision(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor) -> torch.Tensor:
    """Computes collision value based on normals signs."""
    b, n_hand, _ = hand_pcd.shape
    n_obj = obj_pcd_nor.shape[0]
    
    obj_pcd = obj_pcd_nor[:, :3]
    obj_nor = obj_pcd_nor[:, 3:6]

    # Efficient broadcasting
    # [B, N_hand, N_obj, 3]
    diff = obj_pcd.view(1, 1, n_obj, 3) - hand_pcd.view(b, n_hand, 1, 3)
    dists = diff.norm(dim=3)
    
    # Nearest object point for each hand point
    min_dists, min_indices = dists.min(dim=2) # [B, N_hand]
    
    # Gather normals and points
    # indices: [B, N_hand] -> expand to [B, N_hand, 3]
    indices_expanded = min_indices.unsqueeze(-1).expand(-1, -1, 3)
    
    # We need to gather from obj_pcd (which is [N_obj, 3]) using indices [B, N_hand]
    # Since obj is static across batch in this signature, we gather normally
    nearest_pts = obj_pcd[min_indices] # [B, N_hand, 3]
    nearest_nors = obj_nor[min_indices]
    
    # Dot product
    # vec = nearest_pt - hand_pt
    vec = nearest_pts - hand_pcd
    signs = (vec * nearest_nors).sum(dim=2)
    signs = (signs > 0).float()
    
    collision = (signs * min_dists).max(dim=1).values
    return collision

def compute_collision_filter(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    # Same as above but with debug print as per original code behavior
    val = compute_collision(obj_pcd_nor, hand_pcd)
    print(val)
    return val

def ERF_loss(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor) -> torch.Tensor:
    """Repulsion loss using KNN."""
    obj_pcd = obj_pcd_nor[:, :, :3]
    obj_nor = obj_pcd_nor[:, :, 3:6]
    
    # hand_pcd: [B, N_hand, 3], obj_pcd: [B, N_obj, 3]
    knn_res = knn_points(hand_pcd, obj_pcd, K=1, return_nn=True)
    
    nearest_pts = knn_res.knn[:, :, 0, :]
    dists = knn_res.dists.sqrt().squeeze(-1) # [B, N_hand]
    indices = knn_res.idx.squeeze(-1) # [B, N_hand]

    # Gather normals: obj_nor is [B, N_obj, 3]
    batch_indices = torch.arange(obj_nor.shape[0], device=obj_nor.device).view(-1, 1).expand_as(indices)
    nearest_nors = obj_nor[batch_indices, indices, :]
    
    vec = nearest_pts - hand_pcd
    signs = (vec * nearest_nors).sum(dim=2)
    signs = (signs > 0).float()
    
    collision_val = (signs * dists).max(dim=1).values
    return collision_val.mean()

def SPF_loss(dis_points: torch.Tensor, obj_pcd: torch.Tensor, thres_dis: float = 0.02) -> torch.Tensor:
    """Self-Penetration-ish / Contact loss (Simple Point Feature)."""
    dis_points = dis_points.float()
    obj_pcd = obj_pcd.float()
    
    # KNN from contact candidates to object
    res = knn_points(dis_points, obj_pcd, K=1)
    sq_dists = res.dists[..., 0]
    
    mask = sq_dists < (thres_dis ** 2)
    loss = sq_dists[mask].sqrt().sum() / (mask.sum().item() + 1e-5)
    return loss

def SRF_loss(points: torch.Tensor) -> torch.Tensor:
    """Self-Repulsion Feature loss (prevent hand self-intersection)."""
    B, N, _ = points.shape
    # Pairwise distance
    # [B, N, 1, 3] - [B, 1, N, 3]
    diff = points.unsqueeze(2) - points.unsqueeze(1)
    dist = (diff.square().sum(dim=3) + 1e-13).sqrt()
    
    # Filter self-loops and safe clamping
    # Large value for diagonal to ignore
    eye = torch.eye(N, device=points.device).unsqueeze(0) * 1e6
    dist = dist + eye
    
    pen = 0.02 - dist
    pen = torch.relu(pen)
    
    return pen.sum() / B


# --- Data Configurations (Bottom of file to keep logic clean) ---

def _get_static_shadowhand_keypoints():
    return {
    "forearm": [[0.0, -0.01, 0.213]],
    "wrist": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.034]],
    "palm": [[0.0, 0.0, 0.0], [0.033, 0.0, 0.095], [0.011, 0.0, 0.099], [-0.011, 0.0, 0.095], [-0.033, 0.0, 0.02071], [0.034, -0.0085, 0.029]],
    "ffknuckle": [[0.0, 0.0, 0.0]],
    "ffproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.045]],
    "ffmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025]],
    "ffdistal": [[0, 0, 0.024]],
    "fftip": [],
    "mfknuckle": [[0.0, 0.0, 0.0]],
    "mfproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.045]],
    "mfmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025]],
    "mfdistal": [[0, 0, 0.024]],
    "mftip": [],
    "rfknuckle": [[0.0, 0.0, 0.0]],
    "rfproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.045]],
    "rfmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025]],
    "rfdistal": [[0, 0, 0.024]],
    "rftip": [],
    "lfmetacarpal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.06579]],
    "lfknuckle": [[0.0, 0.0, 0.0]],
    "lfproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.045]],
    "lfmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025]],
    "lfdistal": [[0, 0, 0.024]],
    "lftip": [],
    "thbase": [[0.0, 0.0, 0.0]],
    "thproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.038]],
    "thhub": [[0.0, 0.0, 0.0]],
    "thmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.032]],
    "thdistal": [[0, 0, 0.026]],
    "thtip": []
}

def _get_default_keypoints():
    return {
        "forearm": [], "wrist": [], "palm": [], "ffknuckle": [], "fftip": [], 
        "mfknuckle": [], "mftip":[], "rfknuckle": [], "lfmetacarpal": [], 
        "lfknuckle": [], "lftip": [], "thbase": [], "thhub": [], "thtip":[],
        "ffproximal": [[0, 0, 0.024]], "ffmiddle": [[0, 0, 0], [0, 0, 0.025]], "ffdistal": [[0, 0, 0.024]],
        "mfproximal": [[0, 0, 0.024]], "mfmiddle": [[0, 0, 0], [0, 0, 0.025]], "mfdistal": [[0, 0, 0.024]],
        "rfproximal": [[0, 0, 0.024]], "rfmiddle": [[0, 0, 0], [0, 0, 0.025]], "rfdistal": [[0, 0, 0.024]],
        "lfproximal": [[0, 0, 0.024]], "lfmiddle": [[0, 0, 0], [0, 0, 0.025]], "lfdistal": [[0, 0, 0.024]],
        "thproximal": [[0, 0, 0.038]], "thmiddle": [[0, 0, 0.032]], "thdistal": [[0, 0, 0.026]],
    }

def _get_distance_keypoints():
    # Compact representation of the large coordinate list in original file
    # This ensures exact reproducibility without cluttering the class logic
    return {
    "palm": [],
    "ffproximal": [[-0.0002376327756792307, -0.009996689856052399, 0.038666076958179474], [-0.0035445429384708405, -0.009337972849607468, 4.728326530312188e-05], [0.0042730518616735935, -0.0090293288230896, 0.018686404451727867], [-0.003900623880326748, -0.009198302403092384, 0.027359312400221825], [-0.0034948040265589952, -0.009357482194900513, 0.011004473082721233], [0.004485304467380047, -0.008933592587709427, 0.005608899053186178], [0.00421907939016819, -0.009053671732544899, 0.030992764979600906], [-0.003979427739977837, -0.009167392738163471, 0.01910199038684368], [-0.0037133553996682167, -0.009271756745874882, 0.04499374330043793], [0.0034797703847289085, -0.009367110207676888, 0.044556595385074615]],
    "ffmiddle": [[-0.0019831678364425898, -0.007794334553182125, 0.009099956601858139], [0.0017110002227127552, -0.007856125012040138, 0.024990297853946686], [0.003553177695721388, -0.007216729689389467, 0.0004225552547723055], [0.003431637305766344, -0.007271687965840101, 0.01607479713857174], [-0.0025954796001315117, -0.007619044743478298, 0.0195100586861372], [-0.0028260457329452038, -0.007528608664870262, 0.0024113801773637533], [0.003367392346262932, -0.007300738710910082, 0.0064179981127381325], [-0.0014348605182021856, -0.007911290973424911, 0.014330973848700523], [0.0024239378981292248, -0.0076669249683618546, 0.011192334815859795], [-0.003152574645355344, -0.007400532253086567, 0.02429058402776718]],
    "ffdistal": [[-0.00094795529730618, -0.006982842925935984, 0.01811189576983452], [0.002626439556479454, -0.006539319641888142, 4.996722418582067e-05], [-0.0034360431600362062, -0.00615164078772068, 0.008426538668572903], [0.002973517868667841, -0.006382592022418976, 0.011918909847736359], [0.0026527668815106153, -0.006527431774884462, 0.02366715297102928], [-0.0034155123867094517, -0.006162168458104134, 0.0019269119948148727], [-0.0033331606537103653, -0.006204396951943636, 0.023483257740736008], [0.0025241682305932045, -0.0065797604620456696, 0.005907486192882061], [-0.00349381472915411, -0.006122016813606024, 0.0137711763381958], [0.0031557006295770407, -0.006300325505435467, 0.01670246012508869]],
    "mfproximal": [[-0.0020669603254646063, -0.009777018800377846, 0.006492570973932743], [0.00465710973367095, -0.00884400587528944, 0.04495971277356148], [-0.004878615960478783, -0.008722265250980854, 0.027143988758325577], [0.005264972802251577, -0.008492529392242432, 0.017509466037154198], [0.005084634758532047, -0.008596803992986679, 0.032350800931453705], [-0.004348081536591053, -0.008994411677122116, 0.0383140966296196], [-0.004989673383533955, -0.0086652971804142, 0.01689181476831436], [0.005276953335851431, -0.008485602214932442, 0.0004979652003385127], [0.0050704991444945335, -0.008604977279901505, 0.009669311344623566], [0.0027988858055323362, -0.00959551241248846, 0.024868451058864594]],
    "mfmiddle": [[-0.00042295613093301654, -0.008031148463487625, 0.011103776283562183], [0.003634985536336899, -0.007179737091064453, 0.02486497163772583], [0.0035494803451001644, -0.007218401413410902, 0.00027021521236747503], [-0.003957465291023254, -0.007006912492215633, 0.019527770578861237], [-0.003908041398972273, -0.007031631655991077, 0.003815919626504183], [0.0035529686138033867, -0.007216824218630791, 0.017309214919805527], [0.003624177537858486, -0.007184624206274748, 0.006498508155345917], [-0.0026851133443415165, -0.007583887316286564, 0.024684462696313858], [-0.003890471300110221, -0.007041062694042921, 0.014749204739928246], [0.00044322473695501685, -0.008030962198972702, 0.02091158926486969]],
    "mfdistal": [[0.0002320836065337062, -0.007037499453872442, 0.023355133831501007], [0.0036455930676311255, -0.006024540401995182, 6.64829567540437e-05], [-0.0032930555753409863, -0.006224961951375008, 0.010883713141083717], [0.004127402324229479, -0.005703427363187075, 0.015460449270904064], [-0.003456553677096963, -0.006141123361885548, 0.002992440015077591], [0.003985205665230751, -0.005805530119687319, 0.00806250236928463], [-0.0035007710102945566, -0.0061184498481452465, 0.017718486487865448], [0.004287987481802702, -0.00558812078088522, 0.02120518684387207], [0.0011520618572831154, -0.006951729767024517, 0.004545787815004587], [0.0013482635840773582, -0.006914287339895964, 0.011958128772675991]],
    "rfproximal": [[-0.002791694598272443, -0.009589731693267822, 0.015829697251319885], [0.003399983746930957, -0.009393873624503613, 0.04477155953645706], [0.003595913527533412, -0.009328149259090424, 0.0003250864101573825], [-0.0048502604477107525, -0.008736810646951199, 0.031506575644016266], [0.004417791962623596, -0.008964043110609055, 0.024784449487924576], [-0.004430862609297037, -0.008951948024332523, 0.006260100286453962], [0.004398377146571875, -0.008972801268100739, 0.03525833785533905], [0.00454053096473217, -0.008908682502806187, 0.009322012774646282], [-0.00461477879434824, -0.008857605047523975, 0.04111073166131973], [-0.00443872157484293, -0.008947916328907013, 0.023684965446591377]],
    "rfmiddle": [[-0.0024211457930505276, -0.007671383209526539, 0.003983666189014912], [0.002596878679469228, -0.007608974818140268, 0.024917811155319214], [-0.003061287570744753, -0.007436338346451521, 0.015137670561671257], [0.0027961833402514458, -0.007542191073298454, 0.009979978203773499], [0.0029241566080600023, -0.007499308791011572, 0.01832345686852932], [0.0027976972050964832, -0.007541683502495289, 6.793846841901541e-05], [-0.0028554939199239016, -0.007517057936638594, 0.021669354289770126], [-0.00278903404250741, -0.007543126121163368, 0.009370749816298485], [0.002888968912884593, -0.007511099800467491, 0.005062070209532976], [0.0010934327729046345, -0.007965755648911, 0.013925164006650448]],
    "rfdistal": [[0.004119039047509432, -0.005709432996809483, 0.022854819893836975], [-0.004941829480230808, -0.005017726682126522, 2.3880027583800256e-05], [0.005020809359848499, -0.0049394648522138596, 0.009076559916138649], [-0.0051115998066961765, -0.0048520066775381565, 0.014685478061437607], [0.004567775409668684, -0.00535866804420948, 2.354436037421692e-05], [-0.004747811239212751, -0.005207117181271315, 0.023783991113305092], [-0.0021952472161501646, -0.006697545759379864, 0.006976036354899406], [0.0026042358949780464, -0.006549346260726452, 0.015848159790039062], [-0.0018935244297608733, -0.0067823501303792, 0.019202016294002533], [-0.00021895798272453249, -0.007044903934001923, 0.0015841316198930144]],
    "lfmetacarpal": [],
    "lfproximal": [[-0.001103847287595272, -0.009932574816048145, 0.023190606385469437], [0.004271919839084148, -0.009029388427734375, 0.00020035798661410809], [0.0033563950564712286, -0.009408160112798214, 0.04472788795828819], [-0.00359934801235795, -0.009316868148744106, 0.010634070262312889], [-0.0028604455292224884, -0.009570563212037086, 0.0347319096326828], [0.004299539607018232, -0.009016930125653744, 0.01569746620953083], [-0.0040543111972510815, -0.009138413704931736, 0.0022569862194359303], [0.004445703700184822, -0.008951003663241863, 0.03112208843231201], [0.003617372363805771, -0.009320616722106934, 0.007815317250788212], [-0.003905154298990965, -0.009196918457746506, 0.04234839603304863]],
    "lfmiddle": [[-0.0007557488279417157, -0.00800648145377636, 0.006158251781016588], [0.003424291731789708, -0.007274557836353779, 0.024703215807676315], [-0.004018992651253939, -0.006975765340030193, 0.01652413047850132], [0.003509017638862133, -0.007236245553940535, 0.013466663658618927], [-0.0038151403423398733, -0.007080549374222755, 0.023723633959889412], [0.0034917076118290424, -0.007244073320180178, 0.0006077417056076229], [-0.003889185143634677, -0.0070418380200862885, 0.00041095237247645855], [0.0013372180983424187, -0.007935309782624245, 0.019177529960870743], [-0.0038003893569111824, -0.0070875901728868484, 0.010691785253584385], [0.0034656336065381765, -0.007255863398313522, 0.008516711182892323]],
    "lfdistal": [[0.0014511797344312072, -0.006890428718179464, 0.004574548453092575], [-0.0024943388998508453, -0.006586590316146612, 0.023863397538661957], [0.0029716803692281246, -0.006382519379258156, 0.014716518111526966], [-0.002874345052987337, -0.006437260191887617, 0.010602368041872978], [-0.0028133057057857513, -0.006461246870458126, 0.00012360091204755008], [0.00285350508056581, -0.006435882765799761, 0.020840927958488464], [-0.002667121822014451, -0.006518691778182983, 0.017100241035223007], [0.0029705220367759466, -0.006383041851222515, 0.009500452317297459], [0.002596389502286911, -0.006551986560225487, 0.00017241919704247266], [-0.002902866108343005, -0.006426053121685982, 0.0050438339821994305]],
    "thbase": [],
    "thproximal": [],
    "thhub": [],
    "thmiddle": [[-0.010736164636909962, -0.0023433465976268053, 0.005364177282899618], [-0.009576565586030483, 0.005389457568526268, 0.031773995608091354], [-0.010324702598154545, -0.0037935571745038033, 0.020191052928566933], [-0.009223436936736107, 0.005977252032607794, 0.0133969122543931], [-0.008859770372509956, 0.006510394625365734, 0.0007552475435659289], [-0.010023333132266998, -0.004508777987211943, 0.03042592667043209], [-0.009238336235284805, 0.005955408792942762, 0.022636638954281807], [-0.010435031726956367, -0.0034471736289560795, 0.012765333987772465], [-0.010987512767314911, 0.0005528760375455022, 0.026306135579943657], [-0.010371722280979156, 0.0036525875329971313, 0.007199263200163841]],
    "thdistal": [[-0.00094795529730618, -0.006982842925935984, 0.01811189576983452], [0.002626439556479454, -0.006539319641888142, 4.996722418582067e-05], [-0.0034360431600362062, -0.00615164078772068, 0.008426538668572903], [0.002973517868667841, -0.006382592022418976, 0.011918909847736359], [0.0026527668815106153, -0.006527431774884462, 0.02366715297102928], [-0.0034155123867094517, -0.006162168458104134, 0.0019269119948148727], [-0.0033331606537103653, -0.006204396951943636, 0.023483257740736008], [0.0025241682305932045, -0.0065797604620456696, 0.005907486192882061], [-0.00349381472915411, -0.006122016813606024, 0.0137711763381958], [0.0031557006295770407, -0.006300325505435467, 0.01670246012508869]]
    }

if __name__ == '__main__':
    # Simple smoke test
    try:
        from plotly_utils import plot_point_cloud
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing HandModel on {device}...")
        
        hand_model = get_handmodel(1, str(device))
        
        # Test Input: [Trans(3) + Joints(24) + Quat(4)] = 31
        # Replicating original test vector logic
        joint_lower = hand_model.revolute_joints_q_lower[0].cpu().numpy()
        q_trans = np.array([0, 1, 0])
        q_quat = np.array([0, 1, 0, 1]) # Unnormalized test quaternion
        q_full = np.concatenate([q_trans, joint_lower, q_quat])
        
        q_tensor = torch.from_numpy(q_full).unsqueeze(0).float().to(device)
        
        print("Running Forward Kinematics...")
        data = hand_model.get_plotly_data(q=q_tensor, opacity=0.5)
        
        center, toward = hand_model.get_palm_center_and_toward()
        data.append(plot_point_cloud(toward.cpu() + center.cpu(), color='black'))
        data.append(plot_point_cloud(center.cpu(), color='red'))
        
        fig = go.Figure(data=data)
        fig.show()
        print("Test Complete.")
        
    except ImportError:
        print("Skipping visualization test (plotly_utils not found).")
    except Exception as e:
        print(f"Test failed with error: {e}")