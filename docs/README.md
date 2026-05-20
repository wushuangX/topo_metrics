# docs/ — 指标技术文档

本目录存放各拓扑评估指标的详细技术文档，包括算法原理、实现细节、参数说明等。

## 文档索引

| 文档 | 指标 | 说明 |
|------|------|------|
| [`apls.md`](apls.md) | APLS（Average Path Length Similarity） | 基于双向最短路径比对的全局拓扑相似度指标 |
| [`topo_f1.md`](topo_f1.md) | TOPO F1（Topological F1 Score） | 基于种子点局部子图比对的拓扑精度、召回率和 F1 指标 |
| [`apls_verification_report.md`](apls_verification_report.md) | APLS 验证报告 | Python 实现与 Go 参考实现的精度与性能对比报告 |
| [`sphinx/`](sphinx/) | Sphinx 文档站点源码 | Python 开源项目常用的 API + Guide + Technical Docs 文档结构 |

## 深度调研资料

`deepresearch_result/topo_metrics调研/` 目录包含项目启动前的多模型深度调研资料：

| 文件 | 说明 |
|------|------|
| `问题.md` | 调研问题的完整描述 |
| `ChatGPT.md` | ChatGPT 深度调研结果 |
| `gemini.md` | Gemini 深度调研结果 |
| `kimi.md` | Kimi 深度调研结果 |
| `pplx_deepresearch.md` | Perplexity 深度调研结果 |
| `综合评价与总结.md` | 四份调研结果的横向比较与综合评价 |
