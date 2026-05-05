<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 请你生成你的回答

下面给出的是一份更聚焦“落地工具库设计”的版本（在前一份调研基础上，删去冗余、保留关键设计信息），可以直接作为你规划开源 Python 拓扑度量库的蓝本。

***

## 1. 总览：适合工具库实现的指标 \& 损失（精选）

### 1.1 推荐优先实现的拓扑指标（metrics）

| 名称 | 领域 | 任务类型 | 输入 | 需要 graph/skeleton | 可微 | 用途 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| clDice / centerline Dice | 医学/道路 | 细长结构 segmentation | mask / prob + skeleton | skeleton（soft/hard） | 有 soft 版 | metric + loss[^1][^2][^3][^4] |
| Skeleton Precision / Recall / Skel-F1 | 医学/神经/道路 | skeleton/中心线质量 | skeleton | 是 | 否 | metric[^5][^6][^7] |
| Connected Components / CC-based metrics | 通用 | segmentation | mask | 否 | 否 | metric |
| APLS | 遥感 | road graph | node-edge graph | 是 | 否 | metric[^8][^9][^10][^11] |
| TOPO-F1 / Graph IoU | 遥感 | road graph | graph | 是 | 否 | metric[^12][^13][^14] |
| Crack Topology Score (可泛化为 branch/junction metric) | 裂缝/河网 | skeleton graph | skeleton graph | 是 | 否 | metric[^15][^16] |
| Betti / PH-based errors（简化版） | 通用 | segmentation 拓扑 | mask / distance field | 否 | 否 | metric[^17][^18][^19][^20][^21] |

### 1.2 推荐优先实现的拓扑损失（losses）

| 名称 | 领域 | 输入 | 可微 | skeleton/PH 依赖 | 用途 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| clDice loss / soft-clDice / smooth clDice | 医学/道路 | prob + soft skeleton | 是 | 依赖 skeleton | 训练 tubular segmentation[^1][^2][^3][^22][^4] |
| Centerline CE / Centerline Dice loss | 医学 | prob + DT(centerline) | 是 | 不必 hard skeleton | 训练中心线贴合[^23][^24][^25][^26][^27] |
| Skeleton Recall Loss | 医学 | prob + soft skeleton | 是 | skeleton | 连接性保持[^5][^6][^7] |
| 简化 PH / Betti-based loss | 通用 | prob / mask | 近似可微 | PH 后端 | 严格拓扑约束[^17][^28][^29][^30][^18][^19][^20][^21] |
| Geodesic / DT-based topology loss | 医学/道路 | prob + DT / geodesic | 是 | 不需要 skeleton | 对断裂/边界贴合敏感[^31][^32][^33][^34] |
| CAPE 这类 path-enforcement loss（抽象版） | 医学/道路 | prob + endpoints | 局部可微 | shortest path/geodesic | 路径连通约束[^35][^36][^37] |


***

## 2. Taxonomy（为代码结构服务的划分）

### 2.1 Representation 视角划分

- **mask-based：**
    - 连通分量、简单 Betti error。
    - PH-based metric/loss（TopoLoss / Betti Matching 的 wrapper）[^17][^28][^29][^30][^18][^19][^20][^21]。
    - DT / geodesic distance-based loss（boundary/centerline/Topo-DT）[^31][^32][^33][^34]。
- **skeleton-based：**
    - clDice / soft-clDice / smooth clDice。
    - centerline Dice / centerline-based CE。
    - Skeleton Precision/Recall / Skeleton Recall Loss。[^1][^2][^3][^22][^4][^5][^6][^38][^7]
- **graph-based：**
    - APLS / TOPO-F1 / Graph IoU / 连接率、junction/branch 统计。[^8][^9][^10][^11][^12][^13][^14]
    - Crack Topology Score、river network/road boundary topology metrics。[^15][^16][^39][^40]
- **path-based：**
    - shortest-path length similarity（APLS）。
    - path-level matching precision/recall（TOPO-F1 / CAPE family）。[^8][^41][^12][^13][^14][^35][^36]
- **persistent-homology-based：**
    - PH-based metric/loss（TopoLoss、Betti Matching / Efficient Betti Matching）。[^17][^18][^19][^29][^30][^20][^42][^43][^21][^44][^45][^46]
- **distance-field-based：**
    - Differentiable DT / oriented DT / boundary-aware losses。[^31][^32][^33][^34][^47]
- **differentiable-optimization-based：**
    - diff shortest path / diff graph matching / blackbox-backprop。[^48][^49][^50][^51][^52][^53][^54][^55]

***

## 3. 版本规划：v0.1 / v0.2 / v1.0

### v0.1：可直接落地的“最小可用拓扑库”

**目标：只引入 PyTorch / NumPy / SciPy / networkx，避免 PH / C++。**

1. **mask-based：**
    - 连通分量计数 \& 最大连通分量占比 / 每个 component 的 size 分布。
    - 简单洞统计（2D：用 binary closing/connected components 近似 Betti1）。
2. **skeleton-based（高优）：**
    - clDice metric + loss（2D/3D，参考 jocpae / MONAI 实现）[^1][^2][^3][^4]。
    - centerline Dice metric（利用 distance transform 实现“centerline 重叠率”）。
    - skeleton precision/recall（输入 skeleton GT \& pred，匹配点集/线段）。
3. **graph-based：**
    - 定义统一 Graph schema（下节）。
    - APLS（基于 CosmiQ/apls 重写，完全 Python + networkx）[^10][^11]。
    - 基础 graph stats：connected components、edge length分布、平均度数。

> 工程建议：v0.1 尽量无外部 C++ / PH 依赖，让用户“pip 安装即可用”，先让 clDice + APLS + basic connectivity 真正可复现主流论文指标。

### v0.2：增强版拓扑指标 + skeleton/graph 扩展

在 v0.1 稳定基础上：

1. **skeleton-based 扩展：**
    - soft-clDice / smooth clDice。
    - Skeleton Recall Loss（兼容 2D/3D，需 soft skeletonization 栈）[^5][^6][^7]。
2. **graph-based：**
    - TOPO-F1（明确采用 Sat2Graph 定义的一个 variant，写清楚 protocol）[^12][^56][^13][^14]。
    - Graph IoU（edge matching-based IoU，支持道路/裂缝/河网）。
    - Crack Topology Score（从 CTS 通用化为“branch/junction consistency score”）[^15][^16]。
3. **distance-field-based / connectivity loss：**
    - boundary-aware / DT-based loss（简化版 BAE loss）[^47][^31][^32][^33][^34]。
    - 简单 connectivity-aware loss：对预测的“断裂处”施加高权重 penalty（不必引入 full shortest path）。

### v1.0：高级拓扑模块（PH + diff combinatorial）

**重点是插件化，而不是强依赖。**

1. **PH / Betti 模块（可选安装）：**
    - 封装 Hu TopoLoss / Betti Matching：
        - `ph_betti_metric`, `ph_betti_loss`（需安装 `gudhi`/`ripser`/PH 库）。
    - 提供简单 API：`betti_error(pred, gt, dims=[0,1])`。
2. **path-based / diff shortest path：**
    - 完整 CAPE-like path-enforcement loss：支持给定端点/种子，生成 geodesic path mask，并用 softmin / LogSumExp 近似 shortest path[^35][^36][^37]。
    - `diff_shortest_path(cost, method='perturb-and-map'/'softmin')` wrapper，内部调用黑盒组合优化梯度[^48][^49][^50][^51][^52][^53]。
3. **可选 graph-matching / PH 可视化：**
    - 封装 LP-SparseMAP / SG-PGM 等方法作为“图相似度损失”接口[^54][^55]。

***

## 4. 实现难度 \& 运行效率（具体到候选方法）

### 4.1 实现难度（对你团队的预估）

- **低：**
    - 连通分量、Betti0 approximate、centerline Dice、clDice metric/loss、basic graph stats。
- **中：**
    - APLS（需要路径采样 + 多次最短路）、Graph IoU、Crack Topology Score。
    - Skeleton Recall Loss（需要 soft skeletonization 和鲁棒的 skeleton matching）。
- **高：**
    - TOPO-F1（path generation / matching 方案多且细节多）。
    - PH-based loss/metric（TopoLoss / Betti Matching）。
    - diff shortest path / diff graph matching / CAPE full 版本。


### 4.2 运行效率（是否适合大规模 batch）

- **适合 batch：**
    - clDice / centerline Dice / skeleton P-R（图像尺寸中等，特别是 2D）。
    - 基础 connectivity / Betti0 / graph stats。
- **需要加速或谨慎使用：**
    - APLS / TOPO-F1：大图、大量起终点时需要多进程、C++ 或 GPU 实现。
    - PH / Betti Matching：3D 大尺度时成本极高；建议只用于 patch 或 small region。
    - diff shortest path / combinatorial：建议充当 regularizer，不每个 iteration 都算，或者只对 subset 计算。

***

## 5. 与 APLS / TOPO-F1 的互补性（你设计指标体系的核心）

构建工具库的指标组合时，可以这样分层：

1. **图级拓扑（APLS / TOPO-F1）：**
    - 强调全局路网连通与路径质量。
2. **中心线/skeleton 级拓扑（clDice / Skeleton P-R / centerline Dice）：**
    - 弥补 APLS/TOPO 对局部几何细节不敏感的问题；反映“断裂是否发生在局部”。
3. **mask 拓扑（Betti / PH / CC-based）：**
    - 弥补 graph extraction 依赖：即便没做 vectorization，也能在 mask 级检测 holes / branches 的缺失或冗余。
4. **几何-拓扑混合（DT/geodesic-based metric/loss）：**
    - 兼顾边界/中心线偏移 + 连通性（对 small shift / thickness 不敏感，对大断裂敏感）。

在工具库文档中，明确写出“推荐组合”：

- 道路：`IoU + centerline Dice + clDice + APLS + TOPO-F1`。
- 血管/气道：`Dice + clDice + Skeleton Recall + PH/Betti error (可选)`。

***

## 6. 仓库结构 \& API 设计（可以直接用来搭骨架）

### 6.1 推荐目录结构

```text
topometrics/
  __init__.py

  masks/
    connectivity.py      # components, Betti0 approx
    betti.py             # 简易 Betti/PH wrapper（可选）
    distance_field.py    # DT/geodesic metrics & losses

  skeletons/
    skeletonize.py       # hard/soft skeletonization
    metrics.py           # clDice, centerline Dice, Skel-P/R
    losses.py            # clDice loss, SkeletonRecall loss, CC-CE

  graphs/
    schema.py            # Graph dataclass & IO
    apsl.py              # APLS metric
    topo_f1.py           # TOPO-F1, Graph IoU
    crack.py             # CTS & generic branch/junction metrics

  losses/
    connectivity.py      # CAPE-like, simple connectivity loss
    ph.py                # TopoLoss/BettiMatching wrappers (optional)

  vis/
    overlay.py           # mask+graph/skeleton overlay
    graph_plot.py        # networkx/mpl

  batch/
    evaluator.py         # Dataset-wide evaluation helper

  datasets/
    spacenet.py
    topoboundary.py
    vessels_demo.py
```


### 6.2 Graph schema 设计

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Graph:
    nodes: np.ndarray         # (N, D) D=2/3
    edges: np.ndarray         # (M, 2) 每条边的起止 node id
    node_attr: dict = None    # e.g. {"is_junction": bool array, ...}
    edge_attr: dict = None    # e.g. {"length": array, "width": array}
```

并提供：

```python
def graph_from_skeleton(skel: np.ndarray, spacing=None) -> Graph: ...
def graph_from_geojson(path: str) -> Graph: ...
```


### 6.3 Metric \& Loss API 示例

```python
# skeleton-based
def cldice_metric(pred: torch.Tensor, target: torch.Tensor,
                  soft: bool = True, reduction: str = "mean") -> torch.Tensor:
    ...

class ClDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, soft=True, reduction="mean"):
        ...

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...

def centerline_dice(pred: torch.Tensor, target: torch.Tensor,
                    spacing=None, reduction="mean") -> torch.Tensor:
    ...

# graph-based
def apsl_score(ref: Graph, pred: Graph,
               node_pairs=None,
               directed=False,
               normalize=True) -> float:
    ...

def topo_f1(ref: Graph, pred: Graph,
            path_type="junction-end",
            tol=5.0) -> dict:
    # 返回 {"precision": p, "recall": r, "f1": f}
    ...
```


### 6.4 Batch evaluator

```python
class Evaluator:
    def __init__(self, metrics, config=None):
        ...

    def evaluate(self, dataset, preds):
        # dataset 返回 (image, gt_mask/gt_graph)；preds 同结构
        # 内部调用 metrics 并做汇总
        ...
```


***

## 7. 哪些方法“论文引用多但工程坑大”（工具库需要标红）

在文档里建议明确标注以下为“advanced/experimental”：

- TOPO-F1（不同论文定义略有差异，开源实现分散，不同参数对结果影响大）[^12][^13][^14][^57][^58][^59]。
- 所有 PH-based loss（TopoLoss / Betti Matching / Efficient Betti Matching），因为依赖外部 C++ / PH 库且 3D 成本高[^17][^28][^18][^19][^29][^30][^20][^42][^43][^21][^44][^45][^46]。
- Homotopy warping / topology-preserving warping error（实现长、debug 难）[^60][^61][^62][^63][^64]。
- diff shortest path / diff graph matching（blackbox-backprop/perturb-and-MAP），在大图上训练成本高、数值复杂[^48][^49][^50][^51][^52][^53][^54][^55]。


<div align="center">⁂</div>

[^1]: https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf

[^2]: https://www.catalyzex.com/paper/cldice-a-topology-preserving-loss-function

[^3]: https://github.com/dmitrysarov/clDice

[^4]: https://docs.monai.io/en/latest/_modules/monai/losses/cldice.html

[^5]: https://arxiv.org/abs/2404.03010

[^6]: https://arxiv.org/html/2404.03010v1

[^7]: https://github.com/MIC-DKFZ/Skeleton-Recall

[^8]: https://arxiv.org/abs/1807.01232

[^9]: http://arxiv.org/pdf/1807.01232.pdf

[^10]: https://github.com/CosmiQ/apls

[^11]: https://github.com/CosmiQ/cw-eval

[^12]: https://arxiv.org/abs/2007.09547

[^13]: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690052.pdf

[^14]: https://mftp.mmcheng.net/Papers/20CvprVecRoad.pdf

[^15]: https://arxiv.org/html/2601.10762v1

[^16]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12982177/

[^17]: https://papers.neurips.cc/paper/2019/file/2d95666e2649fcfc6e3af75e09f5adb9-Paper.pdf

[^18]: https://arxiv.org/abs/1910.01877

[^19]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9721526/

[^20]: https://proceedings.mlr.press/v202/stucki23a/stucki23a.pdf

[^21]: https://arxiv.org/abs/2407.04683v1

[^22]: https://www.lrde.epita.fr/dload-new/techreps/2025-01-14.techrep.oscar-morand.pdf

[^23]: https://arxiv.org/abs/2407.01517

[^24]: https://papers.miccai.org/miccai-2024/paper/0458_paper.pdf

[^25]: https://papers.miccai.org/miccai-2024/paper/1081_paper.pdf

[^26]: https://papers.miccai.org/miccai-2024/770-Paper1081.html

[^27]: https://papers.miccai.org/miccai-2024/129-Paper0458.html

[^28]: https://arxiv.org/abs/1906.05404

[^29]: https://github.com/HuXiaoling/TopoLoss/blob/master/README.md

[^30]: https://github.com/HuXiaoling/TopoLoss

[^31]: https://arxiv.org/pdf/2209.08355.pdf

[^32]: https://arxiv.org/pdf/2101.08910.pdf

[^33]: https://www.sciencedirect.com/science/article/abs/pii/S0031320323007252

[^34]: https://papers.ssrn.com/sol3/Delivery.cfm/12a81f3c-7f46-4898-914d-2a467dddad80-MECA.pdf?abstractid=4373078\&mirid=1\&type=2

[^35]: https://www.arxiv.org/pdf/2504.00753.pdf

[^36]: https://arxiv.org/abs/2504.00753v2

[^37]: https://openaccess.thecvf.com/content/WACV2025W/CV4EO/papers/Shojaei_Adaptive_Structure-Aware_Connectivity-Preserving_Loss_for_Improved_Road_Segmentation_in_Remote_WACVW_2025_paper.pdf

[^38]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11757899/

[^39]: https://arxiv.org/abs/2103.17119

[^40]: https://paperswithcode.com/paper/topo-boundary-a-benchmark-dataset-on

[^41]: https://adinh26101.github.io/en/posts/apls/

[^42]: https://arxiv.org/html/2407.04683v1

[^43]: https://ui.adsabs.harvard.edu/abs/arXiv:2407.04683

[^44]: https://github.com/AlexanderHBerger/multiclass-BettiMatching

[^45]: https://github.com/nstucki/Betti-matching

[^46]: https://github.com/nstucki/Betti-Matching-3D

[^47]: https://hal.science/hal-03417244v1/file/BMVC21_yizi.pdf

[^48]: https://openreview.net/forum?id=BkevoJSYPB

[^49]: https://arxiv.org/abs/1912.02175

[^50]: https://arxiv.org/pdf/2002.08676.pdf

[^51]: https://arxiv.org/html/2405.04923v2

[^52]: https://arxiv.org/html/2405.04923v1

[^53]: https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/shortestpath_icde2021.pdf

[^54]: http://proceedings.mlr.press/v119/niculae20a/niculae20a.pdf

[^55]: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Xie_SG-PGM_Partial_Graph_CVPR_2024_supplemental.pdf

[^56]: https://github.com/songtaohe/Sat2Graph

[^57]: https://arxiv.org/abs/2209.10150

[^58]: https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/papers/Hetang_Segment_Anything_Model_for_Road_Network_Graph_Extraction_CVPRW_2024_paper.pdf

[^59]: https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/html/Hetang_Segment_Anything_Model_for_Road_Network_Graph_Extraction_CVPRW_2024_paper.html

[^60]: https://arxiv.org/abs/2112.07812

[^61]: https://proceedings.neurips.cc/paper_files/paper/2022/file/98143953a7fd1319175b491888fc8df5-Paper-Conference.pdf

[^62]: https://papers.nips.cc/paper_files/paper/2022/file/98143953a7fd1319175b491888fc8df5-Paper-Conference.pdf

[^63]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html

[^64]: https://imagej.net/plugins/tws/topology-preserving-warping-error
