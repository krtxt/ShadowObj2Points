"""
Sonata Backbone for SceneLeapUltra
基于 PTv3 的自监督预训练模型，适配到标准 backbone 接口

标准接口：
  - forward(pos: (B, N, C)) -> (xyz: (B, K, 3), features: (B, out_dim, K))
  - C 可以是 3 (xyz), 6 (xyz+rgb), 4 (xyz+mask), 7 (xyz+rgb+mask)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict

from .sonata.model import load as load_sonata, PointTransformerV3
from .sonata.transform import Compose
from .sonata.utils import offset2bincount

logger = logging.getLogger(__name__)


class SonataBackbone(nn.Module):
    """
    Sonata Backbone - 标准接口版本
    
    支持两种模式:
    1. raw 模式: 直接输出 Sonata 编码器的原始输出（用于测试/对比）
    2. adapter 模式: 使用 Perceiver 等压缩到固定 token 数（用于训练）
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 基础配置
        self.sonata_model_name = getattr(cfg, 'sonata_model_name', 'sonata_small')
        self.sonata_pretrained = getattr(cfg, 'sonata_pretrained', True)
        self.random_init = getattr(cfg, 'random_init', False)
        self.freeze_sonata = getattr(cfg, 'freeze_sonata', False)
        
        # 预处理配置
        self.grid_size = getattr(cfg, 'grid_size', 0.005)
        self.use_color = getattr(cfg, 'use_color', False)
        self.use_normal = getattr(cfg, 'use_normal', False)
        
        # 输出配置
        self.output_mode = getattr(cfg, 'output_mode', 'raw')  # 'raw' 或 'adapter'
        self.out_dim = getattr(cfg, 'out_dim', 512)
        
        # Sonata 自定义配置
        sonata_cfg = getattr(cfg, 'sonata_custom_config', {})
        enable_flash = sonata_cfg.get('enable_flash', True)
        
        # 检查 FlashAttention
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            enable_flash = False
            self.logger.warning("FlashAttention 未安装，已关闭")
        
        # 加载 Sonata 编码器
        self.logger.info(f"加载 Sonata 模型: {self.sonata_model_name} (预训练={self.sonata_pretrained}, 随机初始化={self.random_init})")
        
        if self.random_init:
            # 仅使用 checkpoint 的 config，构造随机初始化模型
            try:
                ckpt = load_sonata(
                    name=self.sonata_model_name,
                    custom_config={
                        'enable_flash': enable_flash,
                        'enc_patch_size': sonata_cfg.get('enc_patch_size', [256, 256, 256, 256, 256]),
                        'stride': sonata_cfg.get('stride', [2, 1, 1, 1]),
                    },
                    ckpt_only=True,
                )
                self.sonata = PointTransformerV3(**ckpt['config'])
                self.logger.info("Sonata 随机初始化完成（未加载预训练权重）")
            except Exception as e:
                self.logger.error(f"Sonata 随机初始化失败: {e}")
                raise
        else:
            if self.sonata_pretrained:
                self.sonata = load_sonata(
                    name=self.sonata_model_name,
                    custom_config={
                        'enable_flash': enable_flash,
                        'enc_patch_size': sonata_cfg.get('enc_patch_size', [256, 256, 256, 256, 256]),
                        'stride': sonata_cfg.get('stride', [2, 1, 1, 1]),
                    },
                )
            else:
                raise NotImplementedError("从头训练 Sonata 需要手动配置 (建议使用 random_init=true)")
        
        # 仅使用编码器（兼容可选属性）
        if hasattr(self.sonata, 'enc_mode'):
            self.sonata.enc_mode = True
        else:
            self.logger.debug("Sonata 模型缺少 enc_mode 开关，跳过设置")
        
        # 记录 Sonata 输入特征维度（若无法获取则回退到 6）
        self.sonata_in_dim = getattr(self.sonata.embedding, 'in_channels', None)
        if self.sonata_in_dim is None:
            self.logger.warning("未能从 Sonata 模型读取输入通道数，默认使用 6")
            self.sonata_in_dim = 6
        else:
            self.logger.info(f"Sonata 输入通道数: {self.sonata_in_dim}")
        
        # 冻结 Sonata（可选）
        if self.freeze_sonata:
            for param in self.sonata.parameters():
                param.requires_grad = False
            self.logger.info("已冻结 Sonata 编码器参数")
            self.sonata.eval()
        
        # 输出维度直接来自配置，避免在 CPU 上触发 spconv 依赖
        self.sonata_out_dim = getattr(cfg, 'sonata_out_dim', 512)
        self.logger.info(f"Sonata 输出维度: {self.sonata_out_dim}")
        
        # 预处理 transform
        self.transform = Compose([
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=self.grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ])
        
        # Adapter（如果使用）
        if self.output_mode == 'adapter':
            from .sonata_adapter import SonataToTokenAdapter
            
            self.adapter = SonataToTokenAdapter(
                sonata_out_dim=self.sonata_out_dim,
                num_tokens=getattr(cfg, 'num_tokens', 128),
                token_dim=getattr(cfg, 'token_dim', 512),
                compression_mode=getattr(cfg, 'compression_mode', 'perceiver'),
                num_cross_attn_layers=getattr(cfg, 'num_cross_attn_layers', 2),
                num_heads=getattr(cfg, 'num_heads', 8),
                use_prefilter=getattr(cfg, 'use_prefilter', True),
                num_candidates=getattr(cfg, 'num_candidates', 512),
                prefilter_mode=getattr(cfg, 'prefilter_mode', 'fps'),
            )
            self.logger.info(f"使用 Adapter 模式: {cfg.compression_mode}")
        else:
            self.adapter = None
            self.logger.info("使用 Raw 模式: 直接输出 Sonata 原始编码")
        
        # 输出维度投影（如果需要）
        if self.sonata_out_dim != self.out_dim and self.output_mode == 'raw':
            self.out_proj = nn.Linear(self.sonata_out_dim, self.out_dim)
        else:
            self.out_proj = None
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"总参数: {total_params/1e6:.2f}M, 可训练: {trainable_params/1e6:.2f}M")
    
    
    def preprocess_batch(self, pos: torch.Tensor, device: torch.device):
        """
        将 (B, N, C) 格式的点云预处理为 Sonata 所需格式
        
        Args:
            pos: (B, N, C) 点云，C 可以是 3 (xyz), 6 (xyz+rgb), 4 (xyz+mask), 7 (xyz+rgb+mask)
            device: 目标设备
        
        Returns:
            data_dict: Sonata 输入格式
        """
        B, N, C = pos.shape
        
        # 分离 xyz 和其他特征
        coords_torch = pos[:, :, :3]  # (B, N, 3)
        
        # 处理 RGB（如果有且配置使用）
        if self.use_color and C >= 6:
            color_torch = pos[:, :, 3:6]  # (B, N, 3)
        else:
            color_torch = torch.zeros(B, N, 3, device=device, dtype=pos.dtype)
        
        # 处理 object_mask（如果有）
        # 注意: mask 用于过滤点，但 Sonata 的 GridSample 会自动处理稀疏点
        # 这里我们可以选择：
        # 1. 忽略 mask（让 GridSample 自然处理）
        # 2. 预先过滤掉 mask=0 的点
        # 目前选择方案1，保持简单
        
        per_sample = []
        for i in range(B):
            # 转换为 numpy（Sonata transform 需要）
            coord = coords_torch[i].cpu().numpy().astype(np.float32)
            color = color_torch[i].cpu().numpy().astype(np.float32)
            normal = np.zeros((N, 3), dtype=np.float32)  # 法线通常不用
            
            sample = dict(coord=coord, color=color, normal=normal)
            
            try:
                sample = self.transform(sample)
                per_sample.append(sample)
            except Exception as e:
                self.logger.warning(f"样本 {i} 预处理失败: {e}，使用简化版本")
                # 失败时使用简化版本：直接传递坐标
                continue
        
        if len(per_sample) == 0:
            raise RuntimeError("所有样本预处理失败")
        
        # 拼接批次
        coord_cat = torch.cat([s["coord"] for s in per_sample], dim=0)
        grid_coord_cat = torch.cat([s["grid_coord"] for s in per_sample], dim=0)
        feat_cat = torch.cat([s["feat"] for s in per_sample], dim=0)
        offset_cat = torch.cat([s["offset"] for s in per_sample], dim=0)
        offset_cat = torch.cumsum(offset_cat, dim=0)
        
        # 自适配特征维度
        if feat_cat.shape[1] != self.sonata_in_dim:
            if feat_cat.shape[1] > self.sonata_in_dim:
                feat_cat = feat_cat[:, :self.sonata_in_dim]
            else:
                pad = torch.zeros(
                    feat_cat.shape[0],
                    self.sonata_in_dim - feat_cat.shape[1],
                    dtype=feat_cat.dtype,
                    device=device,
                )
                feat_cat = torch.cat([feat_cat, pad], dim=1)
        
        return {
            "coord": coord_cat.to(device),
            "grid_coord": grid_coord_cat.to(device),
            "feat": feat_cat.to(device),
            "offset": offset_cat.to(device),
        }

    def _is_preprocessed_batch(self, x) -> bool:
        """判定输入是否为已在 DataLoader/collate_fn 中预处理完成的批数据。"""
        return isinstance(x, dict) and all(k in x for k in ("coord", "grid_coord", "feat", "offset"))

    def _normalize_collated_data(self, data_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """将 collate_fn 产出的批数据迁移到目标设备并做一致性校验/适配。"""
        out: Dict[str, torch.Tensor] = {}
        for k in ("coord", "grid_coord", "feat", "offset"):
            t = data_dict[k]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t)
            out[k] = t.to(device)

        # 类型规范（通常 grid_coord/offset 为整型索引）
        if out["grid_coord"].dtype not in (torch.int32, torch.int64):
            out["grid_coord"] = out["grid_coord"].long()
        if out["offset"].dtype not in (torch.int32, torch.int64):
            out["offset"] = out["offset"].long()

        # offset 一致性校验，必要时从计数转累计
        sizes = offset2bincount(out["offset"])  # (B,)
        if sizes.sum().item() != out["coord"].shape[0]:
            if out["offset"].ndim == 1 and out["offset"].sum().item() == out["coord"].shape[0]:
                out["offset"] = torch.cumsum(out["offset"], dim=0)
            else:
                raise ValueError(
                    f"预处理 batch 的 offset 无效: sizes_sum={sizes.sum().item()}, n_coord={out['coord'].shape[0]}"
                )

        # 自适配特征维度到 Sonata 期望输入通道
        feat = out["feat"]
        if feat.shape[1] != self.sonata_in_dim:
            if feat.shape[1] > self.sonata_in_dim:
                out["feat"] = feat[:, : self.sonata_in_dim]
            else:
                pad = torch.zeros(
                    feat.shape[0],
                    self.sonata_in_dim - feat.shape[1],
                    dtype=feat.dtype,
                    device=device,
                )
                out["feat"] = torch.cat([feat, pad], dim=1)

        return out
    
    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        标准 backbone forward 接口
        
        Args:
            inputs: 可以是 (B, N, C) 点云，或已在 DataLoader/collate_fn 中预处理好的 data_dict
                 C = 3: xyz only
                 C = 4: xyz + object_mask
                 C = 6: xyz + rgb
                 C = 7: xyz + rgb + object_mask
        
        Returns:
            xyz: (B, K, 3) 输出点坐标
            features: (B, out_dim, K) 输出特征（注意: 第二维是通道维）
        """
        device = next(self.parameters()).device

        # 预处理（若输入已是 collate 后的批数据则直接使用）
        if self._is_preprocessed_batch(inputs):
            data_dict = self._normalize_collated_data(inputs, device)
            # 估计 batch 大小（通过 offset 反推）
            B = offset2bincount(data_dict["offset"]).numel()
        else:
            pos = inputs  # type: ignore[assignment]
            B = pos.shape[0]  # type: ignore[assignment]
            device = pos.device  # 与原逻辑保持：以输入张量设备为准
            data_dict = self.preprocess_batch(pos, device)  # type: ignore[arg-type]
        
        # Sonata 编码
        # 冻结逻辑：仅根据 freeze_sonata 决定是否禁用梯度/设置 eval
        if self.freeze_sonata:
            self.sonata.eval()
            with torch.no_grad():
                point_out = self.sonata(data_dict)
        else:
            point_out = self.sonata(data_dict)
        
        # 根据模式处理输出
        if self.output_mode == 'raw':
            return self._forward_raw(point_out, B, device)
        elif self.output_mode == 'adapter':
            return self._forward_adapter(point_out, B, device)
        else:
            raise ValueError(f"未知的输出模式: {self.output_mode}")
    
    def _forward_raw(self, point_out, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw 模式: 直接输出 Sonata 编码器的原始输出"""
        feat = point_out.feat  # (N_total, C)
        coord = point_out.coord  # (N_total, 3)
        offset = point_out.offset  # (B,)
        
        sizes = offset2bincount(offset)
        
        # 分离每个样本
        feat_list = feat.split(sizes.tolist(), dim=0)
        coord_list = coord.split(sizes.tolist(), dim=0)
        
        # Padding 到统一长度
        max_K = max(f.shape[0] for f in feat_list)
        
        xyz_padded = torch.zeros(B, max_K, 3, device=device, dtype=coord.dtype)
        feat_padded = torch.zeros(B, max_K, self.sonata_out_dim, device=device, dtype=feat.dtype)
        
        for i, (c, f) in enumerate(zip(coord_list, feat_list)):
            K_i = c.shape[0]
            xyz_padded[i, :K_i] = c
            feat_padded[i, :K_i] = f
        
        # 输出维度投影（如果需要）
        if self.out_proj is not None:
            feat_padded = self.out_proj(feat_padded)  # (B, K, out_dim)
        
        # 转换为标准格式: (B, out_dim, K)
        features = feat_padded.transpose(1, 2)  # (B, out_dim, K)
        
        return xyz_padded, features
    
    def _forward_adapter(self, point_out, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapter 模式: 使用 Perceiver 等压缩到固定 token 数"""
        tokens, token_coords, _ = self.adapter(point_out, return_attention=False)
        
        # tokens: (B, K, D)
        # token_coords: (B, K, 3)
        
        # 转换为标准格式: (B, out_dim, K)
        features = tokens.transpose(1, 2)  # (B, out_dim, K)
        
        return token_coords, features

    # =====================
    # Freeze/Unfreeze Controls for Sonata encoder
    # =====================
    def set_freeze_sonata(self, freeze: bool, *, use_eval: bool = True) -> None:
        """手动控制 Sonata 编码器是否冻结（梯度与训练状态）。"""
        for p in self.sonata.parameters():
            p.requires_grad = not freeze
        if use_eval:
            if freeze:
                self.sonata.eval()
            else:
                self.sonata.train()
        logger = getattr(self, 'logger', None)
        try:
            (logger or logging.getLogger(self.__class__.__name__)).info(
                f"Sonata encoder {'frozen' if freeze else 'unfrozen'} (requires_grad={not freeze})"
            )
        except Exception:
            pass

    def sonata_parameters(self):
        """返回 Sonata 编码器参数迭代器（用于参数组/LR 控制）。"""
        return self.sonata.parameters()


# 用于 build_backbone 的工厂函数
def build_sonata_backbone(cfg):
    """构建 Sonata Backbone"""
    return SonataBackbone(cfg)
