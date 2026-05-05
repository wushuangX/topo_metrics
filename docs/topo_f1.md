# TOPO F1 — Topological F1 Score

## 算法原理

TOPO F1 由 Biagioni & Eriksson (2012) 提出，用于评估道路网络推断结果的拓扑质量。其核心思想是：在真实图上选取一组**种子点**，对每个种子点在预测图和真实图上分别执行 BFS 局部子图漫游，得到两组采样点（分别称为 marbles 和 holes），然后通过**二分图匹配**判断局部拓扑一致性。

具体流程：
1. 沿真实图（GT）的边按 `seed_density` 间距生成种子点
2. 对每个种子点，在预测图中找到最近边（需满足距离和角度约束），作为匹配锚点
3. 从匹配的锚点出发，在预测图和真实图上分别执行 BFS 局部漫游（半径 `radius`），收集采样点
4. 预测图采样点称为 **marbles**，真实图采样点称为 **holes**
5. 通过二分图匹配（匈牙利算法）计算局部 Precision 和 Recall
6. 对所有种子点的结果去重后取平均，得到全局 Precision、Recall 和 F1

## 数学定义

对每个种子点 $s_i$，提取 marbles 集合 $M_i$ 和 holes 集合 $H_i$。通过二分图匹配（距离阈值 $d_{\text{th}}$ 和角度阈值 $\theta_{\text{th}}$）找到匹配对 $(m, h)$：

$$
\text{matched}(M_i, H_i) = \{(m, h) \mid m \in M_i, h \in H_i, \|p_m - p_h\| < d_{\text{th}}, |\cos^{-1}(\vec{d}_m \cdot \vec{d}_h)| < \theta_{\text{th}}\}
$$

其中匹配采用匈牙利算法最小化总距离，每个 $m$ 和 $h$ 最多匹配一次。

局部 Precision 和 Recall：

$$
P_i = \frac{|M_i^{\text{matched}}|}{|M_i|}, \quad R_i = \frac{|H_i^{\text{matched}}|}{|H_i|}
$$

对去重后的所有种子点取平均：

$$
P = \frac{1}{N} \sum_i P_i, \quad R = \frac{1}{N} \sum_i R_i
$$

最终 F1：

$$
\text{F1} = 2 \cdot \frac{P \cdot R}{P + R}
$$

分数范围 $[0, 1]$，值越高表示拓扑一致性越好。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prediction` | `TopoGraph` | — | 预测图结构。节点存储 (N, 2) 坐标，边存储 (E, 2) 节点索引对。 |
| `ground_truth` | `TopoGraph` | — | 真实标注图结构。格式与 `prediction` 相同。 |
| `radius` | `float` | `300.0` | 局部子图采样半径（物理单位）。BFS 在此半径范围内采集采样点。 |
| `sample_step` | `float` | `5.0` | 边采样步长（物理单位）。沿边每隔此距离采集一个点作为局部子图样本。 |
| `match_threshold` | `float` | `10.0` | 匹配距离阈值（物理单位）。marble 与 hole 距离超过此值则不匹配。 |
| `angle_threshold` | `float` | `30.0` | 匹配角度阈值（度）。边方向夹角超过此值则不匹配。 |
| `seed_density` | `float` | `50.0` | 种子点密度（物理单位）。沿真实图边每隔此距离生成一个种子点。 |

## 返回值

- **类型**: `tuple[float, float, float]`
- **内容**: `(precision, recall, f1)`
- **范围**: 三个值均在 $[0, 1]$ 区间
- **特殊情况**:
  - 两图均为空 → 返回 `(1.0, 1.0, 1.0)`
  - 仅一图为空或无边 → 返回 `(0.0, 0.0, 0.0)`

## 实现细节

### 1. 种子点生成

调用 `_generate_seeds()` 在真实图的每条链上以 `seed_density` 为间隔生成种子点。种子点均匀分布在链上，避开链两端（使用 `(k+1)/(n+1)` 的比例策略）。每个种子点记录其物理坐标、所在边的两个节点索引、以及到两端节点的距离。

### 2. 种子点匹配（Seed-to-Pred Matching）

对每个种子点，调用 `_match_seed_to_pred()`：
1. 使用 KDTree 在预测图中查找最近邻节点（top-8）
2. 对每个候选节点，检查其邻接边是否已被使用
3. 计算候选边与种子点所在 GT 边的方向余弦相似度，低于 `cos(angle_threshold)` 的边被过滤
4. 计算种子点到候选边的投影距离，取最近且满足 `match_threshold` 的边
5. 匹配后的预测边被标记为已使用，避免后续种子点重复匹配

### 3. 局部子图漫游（Topo Walk）

调用 `_topo_walk()` 以 BFS 方式从匹配边出发，在 `radius` 范围内采集采样点：
- 从种子点所在的边开始，沿边以 `sample_step` 间距收集点
- 通过队列向邻接节点扩散，递归收集邻接边上的采样点
- 记录每个采样点的物理坐标和所在边方向向量
- 使用 `edge_covered` 字典避免同一条边被反复遍历

Holes 的 `radius` 增加 `sample_step`，确保覆盖范围略大于 marbles。

### 4. 二分图匹配

调用 `_bipartite_match()` 对 marbles 和 holes 执行匹配：
1. 构建 cost 矩阵：marbles 为行，holes 为列
2. 使用 KDTree 空间索引加速候选对筛选（`query_ball_point`）
3. 距离超过 `match_threshold` 或角度超过 `angle_threshold` 的对 cost 设为大值
4. 调用 `scipy.optimize.linear_sum_assignment`（匈牙利算法）求解最优匹配
5. cost 小于大值的匹配视为有效匹配

### 5. 种子点去重

调用 `_deduplicate_seeds()` 去除过于接近的种子点：
- 计算种子点之间的图距离
- 距离小于 `radius` 的种子点只保留 F1 更高的那个
- 避免局部区域被过度采样导致偏差

### 6. 最终聚合

- 去重后的所有种子点 precision/recall 取算术平均
- F1 由平均 precision 和 recall 计算得到

### 7. Marbles & Holes 命名由来

术语源自 Biagioni & Eriksson 的原始论文：预测图上的采样点称为 **marbles（弹珠）**，真实图上的采样点称为 **holes（孔洞）**。弹珠要"落入"孔洞才算匹配成功，形象地比喻了两组采样点的匹配关系。若某区域的弹珠找不到对应孔洞，则为假阳性（FP）；反之，孔洞没有弹珠落入则为假阴性（FN）。

## 使用示例

```python
import numpy as np
from topo_metrics import TopoGraph, topo_f1

# 构建预测图：十字交叉结构
pred_nodes = np.array([
    [0.0, 0.0], [100.0, 0.0], [200.0, 0.0],
    [100.0, -50.0], [100.0, 50.0],
])
pred_edges = np.array([
    [0, 1], [1, 2],
    [3, 1], [1, 4],
])
pred = TopoGraph(nodes=pred_nodes, edges=pred_edges, spacing=(1.0, 1.0))

# 构建真实图：与预测图结构一致
gt_nodes = np.array([
    [0.0, 0.0], [100.0, 0.0], [200.0, 0.0],
    [100.0, -50.0], [100.0, 50.0],
])
gt_edges = np.array([
    [0, 1], [1, 2],
    [3, 1], [1, 4],
])
gt = TopoGraph(nodes=gt_nodes, edges=gt_edges, spacing=(1.0, 1.0))

# 计算 TOPO F1
precision, recall, f1 = topo_f1(pred, gt)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"TOPO F1:   {f1:.4f}")
```

## 参考文献

Biagioni, J. & Eriksson, J. "Map Inference in the Face of Noise and Disparity." In *Proceedings of the 20th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, 2012.

```bibtex
@inproceedings{biagioni2012map,
  title={Map Inference in the Face of Noise and Disparity},
  author={Biagioni, James and Eriksson, J{\'a}nos},
  booktitle={Proceedings of the 20th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems},
  year={2012}
}
```
