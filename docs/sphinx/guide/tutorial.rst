教程：完整评估流程
==================

本教程展示如何从零开始构建图并计算两个拓扑指标。

1. 构建真实标注图
-----------------

假设我们有一段道路的真实标注：

.. code-block:: python

   import numpy as np
   from topo_metrics import TopoGraph, apls, topo_f1

   gt_nodes = np.array([
       [100.0, 500.0],   # 起点
       [300.0, 500.0],   # 第一个交叉口
       [500.0, 500.0],   # 直行终点
       [300.0, 300.0],   # 分支终点 A
       [300.0, 700.0],   # 分支终点 B
   ])
   gt_edges = np.array([
       [0, 1],  # 起点 → 交叉口
       [1, 2],  # 交叉口 → 直行终点
       [1, 3],  # 交叉口 → 分支 A
       [1, 4],  # 交叉口 → 分支 B
   ])

   gt = TopoGraph(nodes=gt_nodes, edges=gt_edges, spacing=(1.0, 1.0))

2. 构建预测图
-------------

模型输出了一段道路预测：

.. code-block:: python

   pred_nodes = np.array([
       [100.0, 500.0],
       [300.0, 500.0],
       [500.0, 505.0],   # 轻微偏移
       [300.0, 300.0],
   ])
   pred_edges = np.array([
       [0, 1],
       [1, 2],
       [1, 3],
   ])

   pred = TopoGraph(nodes=pred_nodes, edges=pred_edges, spacing=(1.0, 1.0))

3. 计算 APLS（全局路径连通性）
-------------------------------

.. code-block:: python

   score = apls(
       prediction=pred,
       ground_truth=gt,
       snapping_tol=10.0,
       min_path_length=100.0,
       sample_interval=50.0,
       densify_interval=2.0,
   )
   print(f"APLS: {score:.4f}")

APLS 评估全局路由可达性——预测路网是否支持与真实路网相同的导航路径。

4. 计算 TOPO F1（局部几何拓扑重叠度）
--------------------------------------

.. code-block:: python

   precision, recall, f1 = topo_f1(
       prediction=pred,
       ground_truth=gt,
       radius=300.0,
       sample_step=5.0,
       match_threshold=10.0,
       angle_threshold=30.0,
       seed_density=50.0,
   )
   print(f"TOPO Precision: {precision:.4f}")
   print(f"TOPO Recall:    {recall:.4f}")
   print(f"TOPO F1:        {f1:.4f}")

TOPO F1 评估局部几何与拓扑重叠度——预测路网的细粒度结构保真度。

5. 综合解读
-----------

两个指标互补使用：

- **APLS 高、TOPO F1 低**: 全局路径连通但局部几何偏差大
- **APLS 低、TOPO F1 高**: 局部匹配好但全局路由受阻
- **两者都高**: 拓扑结构高度一致
