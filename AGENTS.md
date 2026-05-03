# AGENTS.md — AI 协作规范

本文档为 AI 编码助手（如 Claude、Copilot 等）提供项目上下文和开发约定。

## 项目概述

`topo-metrics` 是一个 Python 库，提供遥感图像和医学影像领域的**拓扑结构评估指标**。核心目标是让研究者和工程师能方便地计算预测结果与真实标注之间的拓扑一致性。

当前支持：APLS（Average Path Length Similarity）、TopoF1（Topological F1 Score）。

## 技术栈

- **语言**: Python 3.12+
- **包管理**: uv
- **构建系统**: pyproject.toml（PEP 621）
- **测试**: pytest
- **代码风格**: ruff（lint + format）

## 项目结构（约定）

```
topo_metrics/
├── src/
│   └── topo_metrics/
│       ├── __init__.py          # 公共 API 导出
│       ├── apls.py              # APLS 指标实现
│       ├── topo_f1.py           # TopoF1 指标实现
│       └── _graph.py            # 图构建与操作工具函数
├── tests/
│   ├── test_apls.py
│   └── test_topo_f1.py
├── pyproject.toml
└── README.md
```

- 每个指标一个独立模块文件
- 公共 API 通过 `__init__.py` 统一导出
- 内部辅助函数以 `_` 前缀标识

## 代码风格

- 必须使用 **type hints**（函数签名、返回值）
- 所有公共函数必须有 **docstring**（Google 风格）
- import 顺序：标准库 → 第三方库 → 本地模块
- 变量命名：snake_case
- 避免使用 `Any`、`# type: ignore` 等类型逃逸

```python
def apls(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    sample_ratio: float = 0.1,
) -> float:
    """计算 Average Path Length Similarity。

    Args:
        prediction: 预测的二值图，shape (H, W)。
        ground_truth: 真实标注的二值图，shape (H, W)。
        sample_ratio: 采样节点对的比例。

    Returns:
        APLS 分数，范围 [0, 1]，1 表示完全一致。
    """
    ...
```

## 依赖策略

- **最小化外部依赖**。优先使用 numpy/scipy 等科学计算基础库。
- 图操作如需复杂算法（如 networkx），作为可选依赖引入。
- 新增依赖需说明理由。

## 测试要求

- 每个指标至少覆盖：
  1. **完全一致**：prediction == ground_truth → score 应为最优值
  2. **完全不同**：空白预测 vs 有标注 → score 应为最差值
  3. **典型场景**：部分匹配的中间值
- 测试中使用小型合成数据，不要依赖外部文件。
- 运行方式：`uv run pytest`

## 关键约束

1. **输入格式**：统一接受 numpy ndarray（二值图）作为输入，不做图像读取。
2. **返回值**：每个指标函数返回单一标量 float。
3. **性能**：对于大图（如 1024×1024），应考虑内存效率，避免不必要的全图复制。
4. **数值稳定性**：注意除零保护，使用 `np.finfo` 或小常数 eps。

## Git 约定

- 分支命名：`feat/<metric-name>`、`fix/<description>`
- Commit message：英文，简洁描述变更内容
- 不提交 `.venv/`、`__pycache__/`、IDE 配置等
