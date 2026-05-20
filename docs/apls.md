# APLS — Average Path Length Similarity

## 算法原理

APLS（Average Path Length Similarity）通过比较预测图与真实图中**控制点对之间的最短路径长度**来评估图结构的拓扑相似度。其核心思想是：如果两个图拓扑等价，那么任意两个位置之间的最短路径长度应当相同。

APLS 采用**双向计算**策略：
1. 在真实图上采样控制点，映射到预测图，计算路径相似度（GT → Pred）
2. 在预测图上采样控制点，映射到真实图，计算路径相似度（Pred → GT）
3. 最终结果为两个方向的平均值

这使得 APLS 对预测图中缺失或多余的结构都敏感，是一种对称的拓扑一致性度量。

## 数学定义

对于一对匹配的控制点 $(i, j)$，定义路径惩罚：

$$
\text{penalty}(i, j) = \min\left(\frac{|d_{\text{gt}}(i, j) - d_{\text{pred}}(i, j)|}{d_{\text{gt}}(i, j)},\ 1.0\right)
$$

其中 $d_{\text{gt}}(i, j)$ 是真实图中 $i$ 到 $j$ 的最短路径长度，$d_{\text{pred}}(i, j)$ 是预测图中对应映射点的最短路径长度。路径对满足 $d_{\text{gt}}(i, j) > \text{min\_path\_length}$ 才参与计算。

单方向 APLS 值为：

$$
\text{APLS}_{\text{dir}} = 1 - \frac{1}{N} \sum_{(i,j)} \text{penalty}(i, j)
$$

其中 $N$ 为有效控制点对总数。最终 APLS 分数为双向平均：

$$
\text{APLS} = \frac{\text{APLS}_{\text{GT}\to\text{Pred}} + \text{APLS}_{\text{Pred}\to\text{GT}}}{2}
$$

分数范围 $[0, 1]$，$1$ 表示完美匹配。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prediction` | `TopoGraph` | — | 预测图结构。节点存储 (N, 2) 坐标，边存储 (E, 2) 节点索引对。 |
| `ground_truth` | `TopoGraph` | — | 真实标注图结构。格式与 `prediction` 相同。 |
| `snapping_tol` | `float` | `10.0` | 控制点映射的物理距离阈值。源图控制点与目标图最近节点距离超过此值时视为未匹配。 |
| `min_path_length` | `float` | `100.0` | 最短路径长度下限。短于此值的路径对不参与计算，避免短路径噪声干扰。 |
| `sample_interval` | `float` | `50.0` | 控制点采样间距（物理单位）。沿链等距放置控制点，同时包含所有交叉点和端点。 |
| `densify_interval` | `float` | `2.0` | 边加密间距（物理单位）。计算前在长边上插入中间节点，确保采样精度。 |

## 返回值

- **类型**: `float`
- **范围**: $[0, 1]$
- **含义**: $1$ 表示预测图与真实图拓扑完全一致；$0$ 表示完全不一致。
- **特殊情况**:
  - 两图均为空 → 返回 `1.0`
  - 仅一图为空 → 返回 `0.0`

## 实现细节

### 1. 边加密（Densification）

调用 `TopoGraph.densify(interval)` 在长边上按 `densify_interval` 间距插入中间节点。加密后的图拥有更均匀的节点分布，保证短路径采样的精度。

### 2. 控制点采样

流程：
1. 调用 `get_chains()` 提取图中所有链（介于两个非度-2 节点之间的度-2 节点序列）
2. 对每条链，按 `sample_interval` 等距放置控制点
3. 将所有度不等于 2 的节点（交叉点、端点）也加入控制点集合
4. 使用空间网格去重和 4-hop 覆盖避免过密采样

### 3. 控制点映射（Snapping）

使用 KDTree 为每个控制点在目标图中查找最近邻节点。映射规则：
- 距离超过 `snapping_tol` 的视为未匹配（标为 -1）
- 采用**一对一匹配**：每个目标节点最多被匹配一次
- 匹配后，该目标节点的 4-hop 邻域被排除，避免多个控制点映射到同一区域

### 4. 最短路径计算

调用 SciPy 的 `scipy.sparse.csgraph.shortest_path`，使用 Dijkstra 算法计算控制点之间的最短路径。邻接矩阵边权为物理距离（已考虑 `spacing`）。

### 5. 惩罚聚合

- 未匹配的控制点对直接贡献惩罚 $1.0$
- 匹配的控制点对计算归一化路径差
- 短路径对（$d_{\text{gt}} \leq \text{min\_path\_length}$）被跳过
- 如果目标图中路径不可达，视 $d_{\text{tgt}} = 0$，此时惩罚为 $1.0$

### 6. 双向平均

单向 APLS 可能不对称（例如预测图比真实图稠密时，Pred → GT 方向得分可能高于反方向）。取双向平均提供公平的全局度量。

## 使用示例

```python
import numpy as np
from topo_metrics import TopoGraph, apls

# 构建预测图：两条交叉的线段
pred_nodes = np.array([
    [0.0, 0.0], [100.0, 0.0], [200.0, 0.0],  # 水平线
    [100.0, -50.0], [100.0, 50.0],             # 垂直线
])
pred_edges = np.array([
    [0, 1], [1, 2],  # 水平
    [3, 1], [1, 4],  # 垂直
])
pred = TopoGraph(nodes=pred_nodes, edges=pred_edges, spacing=(1.0, 1.0))

# 构建真实图：与预测图相同的拓扑结构
gt_nodes = np.array([
    [0.0, 0.0], [100.0, 0.0], [200.0, 0.0],
    [100.0, -50.0], [100.0, 50.0],
])
gt_edges = np.array([
    [0, 1], [1, 2],
    [3, 1], [1, 4],
])
gt = TopoGraph(nodes=gt_nodes, edges=gt_edges, spacing=(1.0, 1.0))

# 计算 APLS
score = apls(pred, gt)
print(f"APLS: {score:.4f}")  # 预期输出接近 1.0
```

## 从分割掩码直接计算

当输入为二值分割图而非图结构时，可使用 `apls_from_mask` 一步完成：

```python
import numpy as np
from topo_metrics import apls_from_mask

# prediction 和 ground_truth 均为 (H, W) 二值数组
score = apls_from_mask(prediction, ground_truth)
print(f"APLS: {score:.4f}")
```

该函数内部依次调用 `mask_to_graph` 将掩码转为 `TopoGraph`，再调用 `apls` 计算。
所有 `apls` 和 `mask_to_graph` 的参数均可通过关键字参数传递。

## 参考文献

Van Etten, A., Lindenbaum, D., & Bacastow, T.M. "SpaceNet: A Remote Sensing Dataset and Challenge Series." *arXiv preprint arXiv:1807.01232*, 2018.

```bibtex
@article{vanetten2018spacenet,
  title={SpaceNet: A Remote Sensing Dataset and Challenge Series},
  author={Van Etten, Adam and Lindenbaum, Dave and Bacastow, Todd M.},
  journal={arXiv preprint arXiv:1807.01232},
  year={2018}
}
```
