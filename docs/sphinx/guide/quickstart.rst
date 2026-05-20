快速开始
========

基本用法
--------

.. code-block:: python

   import numpy as np
   from topo_metrics import TopoGraph, apls, topo_f1

   # 构建图（节点坐标 + 边索引）
   gt = TopoGraph(
       nodes=np.array([[0, 0], [200, 0], [400, 0]]),
       edges=np.array([[0, 1], [1, 2]]),
       spacing=(1.0, 1.0),
   )

   pred = TopoGraph(
       nodes=np.array([[0, 0], [200, 5], [400, 0]]),
       edges=np.array([[0, 1], [1, 2]]),
       spacing=(1.0, 1.0),
   )

   # 计算 APLS
   score = apls(pred, gt)
   print(f"APLS: {score:.4f}")

   # 计算 TOPO F1
   precision, recall, f1 = topo_f1(pred, gt)
   print(f"TOPO F1: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")

TopoGraph 构造
--------------

``TopoGraph`` 是所有指标的输入类型，存储节点坐标和边连接关系：

- **nodes**: ``(N, 2)`` float64 数组，节点坐标（像素或物理坐标）
- **edges**: ``(E, 2)`` int64 数组，边的节点索引对
- **spacing**: ``(sy, sx)`` 元组，物理尺寸与像素单位的比率，默认 ``(1.0, 1.0)``

从 NetworkX 创建
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import networkx as nx
   from topo_metrics import TopoGraph

   g = nx.Graph()
   g.add_node(0, pos=(0, 0))
   g.add_node(1, pos=(200, 0))
   g.add_edge(0, 1)

   topo = TopoGraph.from_networkx(g, pos_attr="pos", spacing=(1.0, 1.0))
