"""Graph structure configuration and management."""

from typing import Dict, List
import torch
import torch.nn as nn

from .velocity_strategies import build_rigid_groups


class GraphConfig:
    """Encapsulates graph structure parsing and buffer registration.
    
    Handles parsing of graph constants, registration as module buffers,
    and computation of derived properties like rigid groups.
    
    Args:
        module: Parent nn.Module to register buffers on
        graph_consts: Dictionary containing graph structure tensors
    """
    
    def __init__(self, module: nn.Module, graph_consts: Dict[str, torch.Tensor]):
        self.module = module
        self._parse_and_register(graph_consts)
        self._setup_rigid_groups(graph_consts)
    
    def _parse_and_register(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        """Parse graph constants and register as module buffers."""
        self._register_graph_topology(graph_consts)
        self._register_geometry(graph_consts)
        self._register_norm_bounds(graph_consts)
        # Normalized canonical xyz will be populated later when normalization is applied.
        self.module.norm_canonical_xyz = None
        self.norm_canonical_xyz = None

        # Normalized approx std will be populated later when normalization is applied.
        self.module.norm_approx_std = None
        self.norm_approx_std = None

    def _register_graph_topology(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        finger_ids = graph_consts["finger_ids"].long()
        joint_type_ids = graph_consts["joint_type_ids"].long()
        edge_index = graph_consts["edge_index"].long()
        edge_type = graph_consts["edge_type"].long()
        edge_rest_lengths = graph_consts["edge_rest_lengths"].float()
        
        self.N = finger_ids.shape[0]
        self.num_fingers = int(finger_ids.max().item()) + 1
        self.num_joint_types = int(joint_type_ids.max().item()) + 1
        self.num_edge_types = int(edge_type.max().item()) + 1
        
        self.module.register_buffer("finger_ids_const", finger_ids)
        self.module.register_buffer("joint_type_ids_const", joint_type_ids)
        self.module.register_buffer("edge_index", edge_index)
        self.module.register_buffer("edge_type", edge_type)
        self.module.register_buffer("edge_rest_lengths", edge_rest_lengths)
        
    def _register_geometry(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        template_xyz = graph_consts.get("template_xyz", None)
        if template_xyz is not None:
            self.module.register_buffer("template_xyz", template_xyz.float())
            self.template_xyz = self.module.template_xyz
        else:
            self.module.template_xyz = None
            self.template_xyz = None

        canonical_xyz = graph_consts.get("canonical_xyz", None)
        if canonical_xyz is not None:
            self.module.register_buffer("canonical_xyz", canonical_xyz.float())
            self.canonical_xyz = self.module.canonical_xyz
        else:
            self.module.canonical_xyz = None
            self.canonical_xyz = None

        approx_std = graph_consts.get("approx_std", None)
        if approx_std is not None:
            self.module.register_buffer("approx_std", approx_std.float())
            self.approx_std = self.module.approx_std
        else:
            self.module.approx_std = None
            self.approx_std = None
    
    def _register_norm_bounds(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        """Register normalization bounds as buffers if provided."""
        norm_min = graph_consts.get("norm_min", None)
        norm_max = graph_consts.get("norm_max", None)
        if norm_min is None or norm_max is None:
            self.module.norm_min = None
            self.module.norm_max = None
            self.norm_min = None
            self.norm_max = None
            return

        def _reshape_bound(x: torch.Tensor) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            x = x.float()
            if x.numel() == 3:
                return x.view(1, 1, 3)
            if x.dim() == 2 and x.shape[-1] == 3:
                return x.view(1, 1, 3)
            if x.dim() == 3 and x.shape[-1] == 3 and x.shape[0] == 1 and x.shape[1] == 1:
                return x
            raise ValueError(f"Normalization bounds must have shape (3,), (1,3), or (1,1,3); got {tuple(x.shape)}")

        norm_min_t = _reshape_bound(norm_min)
        norm_max_t = _reshape_bound(norm_max)
        self.module.register_buffer("norm_min", norm_min_t)
        self.module.register_buffer("norm_max", norm_max_t)
        self.norm_min = self.module.norm_min
        self.norm_max = self.module.norm_max
    
    def _setup_rigid_groups(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        """Setup rigid groups from graph constants or build from edge topology."""
        rigid_groups = graph_consts.get("rigid_groups", None)
        if rigid_groups is not None and len(rigid_groups) > 0:
            self.rigid_groups = [g.clone().detach().long() for g in rigid_groups]
        else:
            self.rigid_groups = build_rigid_groups(self.module.edge_index, self.N)
    
    def get_edge_index(self) -> torch.Tensor:
        """Get edge index tensor."""
        return self.module.edge_index
    
    def get_edge_rest_lengths(self) -> torch.Tensor:
        """Get edge rest lengths tensor."""
        return self.module.edge_rest_lengths
