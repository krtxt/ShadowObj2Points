from .point_transformer_backbone import PointTransformerBackbone
try:
    from .pointnet2 import Pointnet2Backbone
    from .pointnet2_3sa import Pointnet2Backbone_3sa
except ModuleNotFoundError:
    Pointnet2Backbone = None
    Pointnet2Backbone_3sa = None
# from .pointnext_backbone import PointNextBackbone
from .point_patch_embedding import PointPatchEmbeddingBackbone
from .ptv3_backbone import PTV3Backbone
from .ptv3_sparse_encoder import PTv3SparseEncoder
from .resnet import build_resnet_backbone
from .sonata_backbone import SonataBackbone

import hydra

def build_backbone(backbone_cfg):
    if "_target_" in backbone_cfg:
        return hydra.utils.instantiate(backbone_cfg)
        
    if backbone_cfg.name.lower() == "resnet":
        return build_resnet_backbone(backbone_cfg)
    elif backbone_cfg.name.lower() == "pointnet2":
        if Pointnet2Backbone is None:
            raise ImportError(
                "Backbone 'pointnet2' requires the 'pointnet2' package. "
                "Install it or choose a different backbone."
            )
        return Pointnet2Backbone(backbone_cfg)
    elif backbone_cfg.name.lower() == "pointnet2_3sa":
        if Pointnet2Backbone_3sa is None:
            raise ImportError(
                "Backbone 'pointnet2_3sa' requires the 'pointnet2' package. "
                "Install it or choose a different backbone."
            )
        return Pointnet2Backbone_3sa(backbone_cfg)
    # elif backbone_cfg.name.lower() == "pointnext":
        # return PointNextBackbone(backbone_cfg)
    elif backbone_cfg.name.lower() in ("ptv3", "ptv3_light", "ptv3_no_flash"):
        return PTV3Backbone(backbone_cfg)
    elif backbone_cfg.name.lower() in ("ptv3_sparse", "ptv3_tokens"):
        return PTv3SparseEncoder(backbone_cfg)
    elif backbone_cfg.name.lower() == "point_transformer":
        return PointTransformerBackbone(backbone_cfg)
    elif backbone_cfg.name.lower() in ("point_patch_embedding", "point_patch"):
        return PointPatchEmbeddingBackbone(backbone_cfg)
    elif backbone_cfg.name.lower() in ("sonata", "sonata_backbone", "sonata_small"):
        return SonataBackbone(backbone_cfg)
    else:
        raise NotImplementedError(f"No such backbone: {backbone_cfg.name}")
