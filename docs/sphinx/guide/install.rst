安装
====

要求
----

- Python >= 3.12

使用 uv 安装（推荐）
---------------------

.. code-block:: bash

   uv add topo-metrics

使用 pip 安装
-------------

.. code-block:: bash

   pip install topo-metrics

开发安装
--------

.. code-block:: bash

   git clone git@github.com:wushuangX/topo_metrics.git
   cd topo_metrics
   uv sync

运行测试：

.. code-block:: bash

   uv run pytest

构建文档
--------

.. code-block:: bash

   uv sync --group docs
   cd docs/sphinx && uv run make html

依赖
----

- numpy >= 1.26
- scipy >= 1.12
- networkx >= 3.2
