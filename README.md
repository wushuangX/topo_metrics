# topo-metrics

遥感图像与医学影像的拓扑结构评估指标库。

## 简介

在道路网络提取、血管分割等任务中，传统的像素级指标（如 IoU、F1）无法衡量预测结果的**拓扑一致性**——即图结构的连通性是否与真实标注一致。`topo-metrics` 提供了一系列专门用于评估拓扑结构质量的指标。

## 支持的指标

### APLS — Average Path Length Similarity

基于图最短路径的拓扑相似度指标。通过比较预测图与真实图中对应节点对之间的最短路径长度差异，评估拓扑结构的连通等价性。APLS 侧重于衡量**路由可达性**——预测的路网是否能支持与真实路网相同的导航路径。

- **原始论文**: Van Etten, A., Lindenbaum, D., & Bacastow, T.M. "SpaceNet: A Remote Sensing Dataset and Challenge Series", arXiv:1807.01232, 2018
- **适用场景**: 道路网络提取、管状结构分割

### TOPO F1 — Topological F1 Score

TOPO 指标由 Biagioni & Eriksson (2012) 提出，用于评估道路网络的拓扑质量。将预测路网和真实路网转成图后，通过可达性/局部子图比对得到拓扑意义上的 TP/FP/FN，进而计算拓扑 Precision、Recall 和 F1。其中的 F1 即 TOPO F1（也写作 TOPO(F1) 或 TOPO-F1），侧重于衡量预测路网的**几何连通性与结构完整性**。

- **原始论文**: Biagioni, J. & Eriksson, J. "Map Inference in the Face of Noise and Disparity", ACM SIGSPATIAL GIS 2012
- **适用场景**: 道路网络提取、地图推断

> **APLS 与 TOPO F1 的关系**：APLS 评估全局路径连通性（长距离路由质量），TOPO F1 评估局部几何与拓扑重叠度（细粒度结构保真度）。两者互补，通常一起报告。

## 安装

```bash
# 使用 uv（推荐）
uv add topo-metrics

# 或使用 pip
pip install topo-metrics
```

要求 Python >= 3.12。

## 快速开始

```python
from topo_metrics import apls, topo_f1

# APLS 示例
score = apls(prediction, ground_truth)
print(f"APLS: {score:.4f}")

# TOPO F1 示例
f1 = topo_f1(prediction, ground_truth)
print(f"TOPO F1: {f1:.4f}")
```

## 开发

```bash
# 克隆仓库
git clone git@github.com:wushuangX/topo_metrics.git
cd topo_metrics

# 安装开发依赖
uv sync

# 运行测试
uv run pytest
```

## 引用

如果这些指标对你的研究有帮助，请引用相应的原始论文：

```bibtex
@article{vanetten2018spacenet,
  title={SpaceNet: A Remote Sensing Dataset and Challenge Series},
  author={Van Etten, Adam and Lindenbaum, Dave and Bacastow, Todd M.},
  journal={arXiv preprint arXiv:1807.01232},
  year={2018}
}

@inproceedings{biagioni2012map,
  title={Map Inference in the Face of Noise and Disparity},
  author={Biagioni, James and Eriksson, J{\'a}nos},
  booktitle={Proceedings of the 20th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems},
  year={2012}
}
```

## License

MIT
