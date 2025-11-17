## 1. 任务设定（Problem Setting）

我们要解决的是这样一个任务：

> **给定一个三维物体点云，生成一组满足物理约束的灵巧手抓取姿态。**

在这个设定下，一次抓取的表示拆成两部分：

1. **局部关键点坐标（local keypoints on a kinematic tree）**
    
    - 手被建模为一个“关节树”上的 23 个关键点（指尖、指节、掌心等）。
        
    - 在一个固定的 **canonical 手坐标系** 下，每个关键点有一个局部坐标 yi∈R3y_i \in \mathbb{R}^3yi​∈R3。
        
    - 这些点通过图结构连接起来（edge_index, edge_type），边上有固定的 rest length（边长约束）。
        
2. **全局平移 + 旋转（global SE(3) pose）**
    
    - 整只手在世界坐标系下的全局位姿用一个刚体变换描述：
        
        R∈SO(3),t∈R3R \in SO(3),\quad t \in \mathbb{R}^3R∈SO(3),t∈R3
    - 最终世界坐标下的关键点：
        
        xi=Ryi+tx_i = R y_i + txi​=Ryi​+t

**目标**：学习一个条件生成模型

p(y,R,t∣scene_pc)p(y, R, t \mid \text{scene\_pc})p(y,R,t∣scene_pc)

其中 scene_pc\text{scene\_pc}scene_pc 是物体点云（可能是局部的物体表面点，也可以是场景点云），输出是这套**局部关键点 + 全局位姿**，同时满足手部骨架的几何/物理约束（边长、刚体组等）。

---

## 2. 数据说明（Data）

每个样本大致包含以下信息：

### 2.1 输入：物体点云

- `scene_pc ∈ ℝ^{B × P × 3}`
    
    - B: batch size
        
    - P: 点云点数
        
    - 典型来源：模拟 / 数据集里的物体表面点，或场景点云下的物体子集。
        

### 2.2 输出：手部抓取（世界坐标）

- `xyz_world ∈ ℝ^{B × N × 3}`
    
    - N=23，手上 23 个关键点的**世界坐标**（关节树上的节点：掌心、指节各级、指尖等）。
        
- `hand_model_pose ∈ ℝ^{B × D}`
    
    - 前 3 维：全局平移 `t_global`
        
    - 后 6 维：全局旋转的 6D 表征 `r6d_global`（转成 R∈SO(3)R\in SO(3)R∈SO(3) 使用）
        

利用 `hand_model_pose` 我们可以从世界坐标 `xyz_world` 反算出局部坐标：

yi=R⊤(xi−t)y_i = R^\top (x_i - t)yi​=R⊤(xi​−t)

### 2.3 图结构常量（Graph constants）

这些是一次性预先定义的，与具体样本无关（来自 datamodule，比如 `HandEncoderDataModule.get_graph_constants()`）：

- `finger_ids ∈ ℕ^N`：每个关键点所属的手指 ID（拇、食、中、无、小、掌等）。
    
- `joint_type_ids ∈ ℕ^N`：关键点在指上的角色（根关节、近节、中节、远节、指尖…）。
    
- `edge_index ∈ ℕ^{2×E}`：图边 (i, j) 列表，描述骨架/刚体/星形连接关系。
    
- `edge_type ∈ ℕ^E`：边类型（关节边、跨指边、掌心星形边等）。
    
- `edge_rest_lengths ∈ ℝ^E`：每条边在 rest pose 下的基准长度，用于长度约束。
    
- `template_xyz ∈ ℝ^{N×3}`：模板手姿态下的关键点坐标。
    
- `rigid_groups`：刚体组的定义（比如某些 phalanx 的点应随着同一个 SE(3) 变换刚体运动），可用于更强的刚体约束。
    

---

## 3. 技术要点概览（Key Ideas）

1. **Flow Matching / Rectified Flow**
    
    - 不是直接在数据空间做 diffusion，而是用 flow matching 学习一个速度场 vθv_\thetavθ​，把简单先验（高斯）逐步变成抓取分布。
        
    - 网络预测的是 **速度**，训练目标是**匹配解析速度**（我们这里用简单的 v\*=x1−x0v^\* = x_1 - x_0v\*=x1​−x0​ 直线插值设定）。
        
2. **局部关键点 + 全局 SE(3) 的解耦建模**
    
    - 用局部关键点 yyy 表示手的内部构型，用 SE(3) (R,t)(R,t)(R,t) 表示整手的位置和朝向。
        
    - 相比直接在世界坐标上建模，这种拆分更利于**约束建模**和**泛化**。
        
3. **Graph-aware DiT 作为条件速度场网络**
    
    - 关节树天然是一个图；我们用 DiT（Diffusion Transformer）结构，让**手部 token**做 self-attn，同时用图结构构造 attention bias，强化邻接与骨桿结构。
        
    - 物体点云通过 PointNet / PointTransformer backbone 抽成 **scene tokens**，再用 cross-attn 融入手部表示。
        
4. **物理约束（骨长 / 刚体 / 关节结构）**
    
    - 训练时：在速度场上加入**切向约束 loss**，鼓励速度位于「保持边长的流形」的切空间。
        
    - 推理时：在每一步数值积分之后，对局部关键点做若干步 **PBD（Position-Based Dynamics）式投影**，把边长拉回到 rest length 附近。
        
    - 有需要可以进一步在参数化中引入 FK / 刚体参数，使部分约束“天然满足”。
        

---

## 4. 模型设计（Model Design）

### 4.1 表示与 Flow Matching 变量

我们在训练时显式引入以下变量：

- 局部关键点：
    
    - y1∈RB×N×3y_1 \in \mathbb{R}^{B\times N\times 3}y1​∈RB×N×3：由 `xyz_world` 和 `hand_model_pose` 反算得到：
        
        y1=Rglobal⊤(xworld−tglobal)y_1 = R_{\text{global}}^\top (x_{\text{world}} - t_{\text{global}})y1​=Rglobal⊤​(xworld​−tglobal​)
- 全局位姿：
    
    - 平移 t1∈RB×3t_1 ∈ ℝ^{B×3}t1​∈RB×3：来自 `hand_model_pose`。
        
    - 旋转 r1∈RB×6r_1 ∈ ℝ^{B×6}r1​∈RB×6：6D 表征，通过 `rot6d_to_matrix` 转为 R1R_1R1​。
        

Flow Matching 用一个简单的 **直线插值**路径：

- 采样 base：
    
    y0∼N(0,I),t0∼N(0,I),r0∼N(0,I)y_0 \sim \mathcal{N}(0,I),\quad t_0 \sim \mathcal{N}(0,I),\quad r_0 \sim \mathcal{N}(0,I)y0​∼N(0,I),t0​∼N(0,I),r0​∼N(0,I)
- 采样时间 τ∼U(0,1) \tau \sim U(0,1)τ∼U(0,1)，构造路径上的点：
    
    yτ=(1−τ)y0+τy1,tτ=(1−τ)t0+τt1,rτ=(1−τ)r0+τr1y_\tau = (1-\tau)y_0+\tau y_1,\quad t_\tau = (1-\tau)t_0+\tau t_1,\quad r_\tau=(1-\tau)r_0+\tau r_1yτ​=(1−τ)y0​+τy1​,tτ​=(1−τ)t0​+τt1​,rτ​=(1−τ)r0​+τr1​
- 解析目标速度：
    
    vy\*=y1−y0,vt\*=t1−t0,vr\*=r1−r0v_y^\* = y_1-y_0,\quad v_t^\* = t_1-t_0,\quad v_r^\* = r_1-r_0vy\*​=y1​−y0​,vt\*​=t1​−t0​,vr\*​=r1​−r0​

网络预测 v^y,v^t,v^r\hat v_y, \hat v_t, \hat v_rv^y​,v^t​,v^r​，用 MSE 拟合这些解析速度。

> 未来你也可以升级为 conditional flow matching (CFM) 那类更「diffusion-like」的路径定义，这套代码结构是兼容的。

---

### 4.2 Hand encoder：局部关键点 → per-point tokens (B, N, D)

文件：`components/hand_encoder.py`

输入：

- `xyz_local` 或 flow 中间状态 `y_tau ∈ (B,N,3)`
    
- `finger_ids ∈ (N,)`
    
- `joint_type_ids ∈ (N,)`
    
- `edge_index ∈ (2,E)`
    
- `edge_type ∈ (E,)`
    
- `edge_rest_lengths ∈ (E,)`
    

核心做法：

1. 每个点的基础几何特征：
    
    - xyz 的 Fourier embedding：γ(yi)\gamma(y_i)γ(yi​)
        
    - 相对模板偏移：yi−templateiy_i - \text{template}_iyi​−templatei​（可选）
        
2. 离散属性嵌入：
    
    - finger embedding：`Embedding(finger_ids)`
        
    - joint type embedding：`Embedding(joint_type_ids)`
        
3. 拼接后用 MLP + LayerNorm 投到 `d_model` 维：  
    得到 `hand_tokens ∈ (B,N,d_model)`。
    

在此基础上，有两套结构可选（我们默认用 Transformer 版）：

- **HandPointTokenEncoderEGNNLite**
    
    - 用类似 EGNN 的 Graph Message Passing（不更新坐标，只更新特征），显式引入边长 / rest length 等几何量，输出 per-point tokens。
        
- **HandPointTokenEncoderTransformerBias**（推荐）
    
    - 用若干层 Transformer encoder，每层自注意力加入一个 graph attention bias（来自边的结构和 rest length），输出 per-point tokens。
        

输出接口统一为：

`hand_tokens = hand_encoder(     xyz=y_tau,                     # (B,N,3)     finger_ids=(B,N),     joint_type_ids=(B,N),     edge_index=(2,E),     edge_type=(E,),     edge_rest_lengths=(E,), )  # -> (B,N,d_model)`

---

### 4.3 Scene backbone：点云 → scene tokens (B, K, D)

文件：`models/backbone`（通过 `build_backbone` 工厂函数实例化）

输入：

- `scene_pc ∈ (B,P,3)`：物体点云
    

输出：

- `scene_tokens ∈ (B,K,d_model)`：K 个场景 token，用于 cross-attn
    
    - K 可以是 1（全局向量）或 32/64（下采样后的局部 token），由 backbone 结构决定。
        
    - backbone 可以是 PointNet++, Point Transformer 或自定义网络。
        

---

### 4.4 Graph-aware DiT：hand tokens + scene tokens

文件：`components/graph_dit.py`

核心组件：

1. **GraphAttentionBias**
    
    - 初始化时绑定图结构：
        
        - `num_nodes=N`
            
        - `edge_index, edge_type, edge_rest_lengths`
            
    - 每次 `forward(xyz)` 时，用当前坐标 xyzxyzxyz 计算边几何：
        
        dist,dist2,δ=dist−restrest\text{dist}, \text{dist}^2, \delta = \frac{\text{dist}-\text{rest}}{\text{rest}}dist,dist2,δ=restdist−rest​
    - 把 `[dist², δ, edge_type_emb]` 过一个 MLP → 标量 `bias_ij`，填进 `(B,N,N)` 矩阵，非边位置用可学习的 `non_edge_bias`。输出 `attn_bias ∈ (B,1,N,N)`。
        
2. **HandSceneGraphDiTBlock**
    
    - 结构借鉴 Hunyuan DiT Block：
        
        - AdaLayerNormShift(temb) + **GraphSelfAttention**（self-attn on hand tokens）
            
        - LayerNorm + **CrossAttention**（hand tokens attend to scene tokens）
            
        - LayerNorm + **FeedForward**（来自 diffusers.attention.FeedForward）
            
    - self-attn 中使用上一步的 `graph_attn_bias` 作为 logits 加性偏置，注入骨架拓扑与 rest length 信息。
        
3. **HandSceneGraphDiT**
    
    - 堆叠 L 个上述 Block：
        
        `h = hand_tokens for block in blocks:     h = block(         hand_tokens=h,         scene_tokens=scene_tokens,         temb=temb,         graph_attn_bias=attn_bias,     )`
        
    - 输出 `h ∈ (B,N,d_model)`，作为预测速度的输入。
        

---

### 4.5 速度场预测头（Flow Matching Heads）

在完整的“局部 + 全局”版本里，我们通常有两支/三支 head：

1. **局部关键点速度 head：**
    
    - 输入：`hand_tokens_out ∈ (B,N,d_model)`
        
    - 输出：`v_hat_y ∈ (B,N,3)`
        
    - 直接用一个 `nn.Linear(d_model → 3)` 即可。
        
2. **全局平移/旋转 head（两种实现可选）：**
    
    - 简洁方案：在 DiT 之后对 hand tokens 做全局池化（mean 或注意力），得到一个 grasp-level token：
        
        `grasp_token = hand_tokens_out.mean(dim=1)  # (B,d_model) v_hat_t = Linear(d_model→3)(grasp_token) v_hat_r = Linear(d_model→6)(grasp_token)`
        
    - 或者在 DiT 的输入中显式加一个 `[GRASP]` 全局 token，一路参与 self-attn / cross-attn，最后专门从该 token 上预测 `v_t, v_r`。
        

训练目标：

LFM=∥vy\*−v^y∥2+wt∥vt\*−v^t∥2+wr∥vr\*−v^r∥2\mathcal{L}_{\text{FM}} = \|v_y^\* - \hat v_y\|^2 + w_t \|v_t^\* - \hat v_t\|^2 + w_r \|v_r^\* - \hat v_r\|^2LFM​=∥vy\*​−v^y​∥2+wt​∥vt\*​−v^t​∥2+wr​∥vr\*​−v^r​∥2

---

### 4.6 物理约束建模

1. **切向约束损失（Training）**
    
    对每条边 (i,j)(i,j)(i,j)，我们希望在 flow 上边长保持不变：
    
    ddt∥yi−yj∥2=2(yi−yj)⋅(vi−vj)≈0\frac{d}{dt}\|y_i - y_j\|^2 = 2(y_i - y_j)\cdot(v_i - v_j) \approx 0dtd​∥yi​−yj​∥2=2(yi​−yj​)⋅(vi​−vj​)≈0
    
    损失写作：
    
    `i, j = edge_index diff_y = y_tau[:, i, :] - y_tau[:, j, :]  # (B,E,3) diff_v = v_hat_y[:, i, :] - v_hat_y[:, j, :] residual = (diff_y * diff_v).sum(-1)      # (B,E) L_tangent = (residual ** 2).mean()`
    
    整体 loss：
    
    `loss = L_FM + lambda_tangent * L_tangent`
    
2. **PBD 投影（Inference）**
    
    推理时，我们在每一个 time-step 更新后，对局部坐标 `y` 做若干次 PBD 式的投影，让边长回到 rest length 附近：
    
    `for _ in range(num_proj_iters):     p = y[:, i, :] - y[:, j, :]          # (B,E,3)     dist = (p.square().sum(-1, keepdim=True) + 1e-9).sqrt()     corr = (1.0 - rest / dist).clamp(-cmax, cmax)  # (B,E,1)     delta = corr * p * 0.5     y.index_add_(1, i, -delta)     y.index_add_(1, j,  delta)`
    
    这样可以把相对误差控制在很小的范围，同时不会破坏全局 SE(3)。
    
3. **刚体组/关节树结构（可选）**
    
    如果要更严格地保证某些点构成刚体，可以把它们参数化成同一个 SE(3) 作用在模板点上，这部分可以放在解码器或一个专门的几何模块中。目前框架已预留 `rigid_groups` 等信息，未来可以在此基础上加入 FK/硬性刚体约束。
    

---

## 5. 训练与推理流程（Lightning 视角）

### 5.1 训练流程

LightningModule：例如 `HandFlowMatchingDiT`

1. 从 batch 中取：
    
    - `xyz_world = batch["xyz"]`
        
    - `hand_model_pose = batch["hand_model_pose"]`（如果你需要）
        
    - `scene_pc = batch["scene_pc"]`
        
2. 拆出全局 SE(3) 并回到局部坐标：
    
    - 用 `hand_model_pose` 得到 `t_global, r6d_global → R_global`
        
    - `y1 = R_global^T (xyz_world - t_global)`
        
    - `t1 = t_global, r1 = r6d_global`
        
3. 构造 flow path：
    
    - 调 `_flow_matching_path(y1, t1, r1)`，得到 `(y0,t0,r0, tau, y_tau,t_tau,r_tau, v*_y, v*_t, v*_r)`。
        
4. 前向预测：
    
    - `scene_tokens = encode_scene_tokens(scene_pc)`
        
    - `hand_tokens = encode_hand_tokens(y_tau)`
        
    - `temb = time_embed(tau)`
        
    - `hand_tokens_out = dit(hand_tokens, scene_tokens, temb, xyz=y_tau)`
        
    - `v_hat_y, v_hat_t, v_hat_r = heads(hand_tokens_out 或 grasp_token)`
        
5. 损失与日志：
    
    - `L_FM = mse(v_hat_y, v*_y) + ...`
        
    - `L_tangent` 如上
        
    - 总 loss 回传；同时 log 边长相对误差等指标。
        

### 5.2 推理流程

给定 `scene_pc`，生成抓取：

1. 初始化：
    
    - y(0)∼N(0,I)y(0) \sim \mathcal{N}(0,I)y(0)∼N(0,I)，t(0)∼N(0,I)t(0)\sim \mathcal{N}(0,I)t(0)∼N(0,I)，r(0)∼N(0,I)r(0)\sim \mathcal{N}(0,I)r(0)∼N(0,I)
        
    - 构造时间网格 0=t0<...<tK=10=t_0 < ... < t_K=10=t0​<...<tK​=1
        
2. 每个 step：
    
    - 设当前 tkt_ktk​，构造 `tau = t_k`；
        
    - `v_hat_y, v_hat_t, v_hat_r = predict_velocity(y, t, r, scene_pc, tau)`；
        
    - 数值积分（Euler/Heun）：
        
        `y = y + dt * v_hat_y t = t + dt * v_hat_t r = r + dt * v_hat_r`
        
    - 对 y 做 PBD 投影，保证边长近似保持。
        
3. 最后一步：
    
    - 把 `r` 转成旋转矩阵 `R`（6D→SO(3)）；
        
    - `xyz_world_pred = R @ y + t`。
        

输出 `xyz_world_pred ∈ (B,N,3)` 就是一组对物体点云的物理可行抓取。

---

## 6. 代码结构示意（简化版）

Project/
├─ conf/
│  ├─ config. Yaml                  # Hydra 默认
│  ├─ datamodule/hand_encoder. Yaml # HandEncoderDataModule 配置
│  ├─ model/hand_flow_dit. Yaml     # FlowMatching + DiT 模型配置
│  └─ trainer/default. Yaml         # PL Trainer 配置
├─ src/
│  ├─ datamodules/
│  │  └─ hand_encoder_datamodule. Py  # 提供 xyz_world / scene_pc / graph_consts
│  ├─ models/
│  │  ├─ backbone/                 # 点云 backbone（build_backbone）
│  │  ├─ components/
│  │  │  ├─ hand_encoder. Py        # HandPointTokenEncoder*（per-point tokens）
│  │  │  └─ graph_dit. Py           # GraphAttentionBias + DiT blocks
│  │  └─ hand_flow_matching_dit. Py # LightningModule: 整个 flow + loss + 推理
│  ├─ train. Py                     # Hydra + Lightning 启动脚本
│  └─ ...
└─ shadowobj2points. Md
