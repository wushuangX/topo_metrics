# **拓扑结构评估指标与拓扑保持损失函数系统性深度调研报告**

## **引言**

在计算机视觉与模式识别领域，针对管状、网状或树状拓扑结构（如城市道路网、医学血管、气道树、神经元分支及地表裂缝）的自动提取与分割，一直是极具挑战性的核心任务。传统的基于像素级重叠度的评估指标（如 Intersection over Union (IoU) 和 Dice 系数）与损失函数（如 Cross-Entropy），假设像素之间相互独立，严重忽略了结构的全局连通性与拓扑一致性。在实际场景中，仅仅几个像素的预测错误便可能切断一条主干道或主动脉，导致预测结果在拓扑功能上完全失效，而传统的像素级指标对此类致命错误却极不敏感 1。

为了弥补这一鸿沟，学术界在遥感图像处理、医学图像分割和计算拓扑学等领域，发展出了一系列拓扑感知评估指标（Topology-aware Evaluation Metrics）与拓扑保持损失函数（Topology-preserving Loss Functions）。本报告系统性地深度调研并提炼了这些前沿方法，对其核心思想、数学定义、计算复杂度及工程实现特性进行了详尽的解构，旨在为构建一个支持大规模端到端训练与多任务评估的开源 Python 拓扑结构评估工具库提供完备的理论基础与架构设计指导。

## ---

**第一部分：拓扑感知评估指标 (Topology-aware Evaluation Metrics) 深度解析**

拓扑评估指标的核心在于量化预测结构与真实结构在连通性、分支一致性和空间几何特征上的差异。根据应用领域的不同，这些指标在图结构提取（如遥感道路）和体素骨架化（如医学管状组织）上展现出不同的侧重点。

### **1\. 遥感道路提取与道路图重建领域**

遥感领域的道路提取不仅要求像素级别的精确，更强调生成的道路图（Road Network Graph）在导航与路径规划上的功能可用性。

#### **Average Path Length Similarity (APLS)**

* **适用场景**：遥感领域；道路网络图重建、中心线提取任务；输入形式为 node-edge graph（节点-边缘图）；需要从二值掩码通过骨架化或后处理生成图结构 3。  
* **工程属性**：不可微；仅用于 evaluation metric；官方开源实现为 Python/Go 5；采用 MIT/Apache 2.0 许可证 7。  
* **算法特性**：APLS 是一种基于图论和最短路径的指标，旨在衡量图的逻辑拓扑与物理物理几何的综合质量。其核心思想是计算真实图 ![][image1] 和预测图 ![][image2] 中对应节点之间的最短路径差异。对于真实图中的节点对，算法使用 Dijkstra 算法计算最短路径 ![][image3]，并在预测图中找到匹配的节点计算 ![][image4]。数学定义通常包含如下形式的累加惩罚：![][image5] 9。  
* **性能特征**：APLS 内在包含了“中介中心性”（Betweenness Centrality）的概念。高流量的十字路口节点若缺失，将切断大量最短路径，从而受到极重的惩罚；而末端死胡同的缺失则惩罚较轻 4。计算复杂度极高，瓶颈在于全源最短路径的计算。对断裂、漏检极度敏感，但对适度的几何偏移有一定容忍度（依赖于节点吸附阈值）。  
* **入库建议**：强烈建议纳入（v0.2）。这是 SpaceNet、CityScale 等数据集的黄金标准 10。实现时需特别注意节点吸附（Node Snapping）算法的稳健性，建议利用 KD-Tree 优化匹配效率。

#### **TOPO / TOPO-F1 (Topological Precision and Recall)**

* **适用场景**：遥感领域；图提取与道路网重建；输入形式为基于拓扑距离的 node-edge graph；需后处理生成图。  
* **工程属性**：不可微；仅用于评估；在 Sat2Graph、SAM-Road 等项目中有 Python 实现 1。  
* **算法特性**：TOPO 指标侧重于局部图结构的相似度。它在真实图中随机采样候选顶点，在预测图中寻找对应点，并比较从该顶点出发，在特定半径（如 16 米）内可达的子图（Reachable Subgraphs）的相似性 11。通过计算可达节点的重叠度，输出拓扑精确率（Precision）、召回率（Recall）和 F1 分数。  
* **性能特征**：对局部断裂和伪连接（Spurious links）具有极高的惩罚力度 11。其复杂度低于 APLS，因为仅需计算局部广度优先搜索（BFS）而非全图最短路径。然而，TOPO 忽略了长距离的路由连通性。  
* **入库建议**：必须纳入（v0.2）。与 APLS 形成完美互补，用于衡量局部图结构的连通准确性。

#### **Graph IoU**

* **适用场景**：遥感领域；道路/车道线图评估；输入为 Polyline graph；需要将图渲染为栅格化掩码。  
* **工程属性**：不可微；仅用于评估。  
* **算法特性**：通过为向量图（Graph）的边缘应用一个固定的缓冲宽度（如 5 到 10 像素），将其渲染为二值化图像，随后计算真实图渲染掩码与预测图渲染掩码非零像素的交并比（IoU） 9。  
* **性能特征**：计算极快，高度并行化。对几何偏移和整体空间对齐极其敏感。缺点是具有“拓扑盲区”：如果渲染线宽过大，内部微小的拓扑断裂会被掩盖，导致图结构实质已断开但 Graph IoU 依然很高 14。  
* **入库建议**：基础指标，易于实现（v0.1）。可作为空间几何对齐的基准校验工具。

### **2\. 医学图像分割领域**

医学分割（如血管、气道、神经元）通常输出概率图或二值掩码，拓扑评估重点在于分支的完整性和管状结构的同伦等价性。

#### **clDice (Centerline Dice)**

* **适用场景**：医学领域；管状组织分割（血管、神经等）；输入形式为二值掩码（Binary Mask）；评估时需要通过硬骨架化（Hard Skeletonization）算法提取中心线 7。  
* **工程属性**：不可微（作为纯指标时）；仅用于评估；GitHub 有官方 Python/PyTorch 实现 7；MIT 许可证。  
* **算法特性**：计算预测掩码的骨架在真实掩码中的召回率（Centerline Sensitivity，即 Skeleton Recall），以及真实掩码的骨架在预测掩码中的精确率（Centerline Precision，即 Skeleton Precision）。clDice 定义为二者的调和平均数：![][image6] 17。  
* **性能特征**：理论证明 clDice 能够保证二值图像分割的同伦等价性（Homotopy Equivalence），确保网络结构的连通性 7。计算复杂度受限于传统骨架化算法（如 skimage.morphology.skeletonize），对管径粗细不均引发的骨架分支错误（毛刺）较为敏感 19。  
* **入库建议**：核心指标，必须纳入（v0.1）。实现成本极低，广泛应用于血管与气道树的评估 2。

#### **Betti Matching Error**

* **适用场景**：计算拓扑与医学分割；二值掩码或距离场；无需提取显式图结构。  
* **工程属性**：可微（存在对应的 Loss 变体）；主要作为高阶评估指标；有 C++ 和 Python 实现 21；计算极耗时。  
* **算法特性**：传统的贝蒂数误差（Betti Number Error）仅比较图像中连通分量（![][image7]）和拓扑环（![][image8]）的全局数量，这在空间结构上存在致命缺陷（例如：数量正确但位置全错）。Betti Matching 引入了持续同调（Persistent Homology）中的诱导匹配（Induced Matchings）理论，在真实与预测的持续条形码（Persistence Barcodes）之间建立具有空间对应关系的特征匹配 22。误差量化了在两幅图像中无法建立空间映射的拓扑特征数量。  
* **性能特征**：对伪连接、漏检、拓扑环具有绝对的解析精确度，是目前衡量拓扑一致性最严谨的数学工具 22。瓶颈在于建立多维图像的复形（Cubical Complex）及计算 Wasserstein 距离，3D 数据上常因内存溢出而失败。  
* **入库建议**：高阶特性，建议后期纳入（v1.0）。需要通过 GUDHI 或 Giotto-tda 等 C++ 库进行 Python 绑定以提升性能 25。

#### **Tree Edit Distance (TED) 与 Skeleton Precision/Recall**

* **适用场景**：医学神经元追踪、气道树提取；树状图与骨架。  
* **工程属性**：不可微；Python 生态中有 apted 或 zss 库支持 27。  
* **算法特性**：TED 衡量将一棵有序树转换为另一棵树所需的最小节点插入、删除和替换操作的代价 28。Skeleton Precision/Recall 则直接衡量骨架分支的像素级覆盖率 2。  
* **性能特征**：TED 对层级依赖和树形末端分支极其敏感，被广泛集成于 PyNeval 等神经元评估工具中 30。致命弱点是无法处理存在环路（如血管吻合支）的通用图结构。

### **3\. 通用图结构与几何结构评估**

#### **Graph Edit Distance (GED)**

* **算法特性与瓶颈**：GED 计算将一个图通过编辑操作转化为目标图的最小成本。作为最完备的图相似度度量，GED 是 NP-hard 问题 31。实际应用中通常采用二分图匹配（Bipartite Matching）或逆最优传输（Inverse Optimal Transport, 比如 GEDIOT）进行近似计算 31。  
* **库实现预警**：**极其不建议作为核心内置指标**。GED 的评分高度依赖于对不同编辑操作（如增删节点 vs. 更改属性）设定的启发式代价矩阵，导致不同论文间的结果无法横向对比 33。计算速度也极难支持大规模批处理 3。

#### **Hausdorff Distance 与 Chamfer Distance (基于骨架/中心线)**

* **算法特性**：将骨架或中心线转化为点云，计算点集间的距离。Chamfer Distance 衡量两个点集之间最近邻距离的平均值；而 95% Hausdorff Distance (95HD) 则衡量最大距离的第 95 百分位点 35。  
* **性能特征**：95HD 对极端的几何偏离（如一根游离的错误分支伸向背景）极其敏感，而 Chamfer Distance 提供平滑的整体几何对齐评估 38。由于忽略了点与点之间的连通边，它们属于距离场几何指标，对局部断裂不敏感。适合作为 v0.1 的基准度量补充。

## ---

**第二部分：拓扑保持损失函数 (Topology-aware Loss Functions) 深度解析**

标准的交叉熵和 Dice 损失函数是逐像素优化的，无法产生拓扑约束。拓扑保持损失函数通过引入微分优化和代数拓扑特征，强制神经网络在训练时学习结构的连通性。

### **1\. 医学图像分割中的拓扑损失**

#### **soft-clDice Loss 与 Soft Skeletonization Loss**

* **适用场景**：医学管状结构分割（视网膜、气道等）；输入为概率图（Probability Map）；无需后处理 39。  
* **工程属性**：完全可微；适合端到端训练；开源于 PyTorch/Keras 7。  
* **核心思想与公式**：由于传统的形态学骨架化算法不可微，作者提出了一种“软骨架化”（Soft Skeletonization）替代方案。它通过迭代的最小池化（Min-pooling，近似形态学腐蚀）和最大池化（Max-pooling，近似形态学膨胀）来提取微分骨架 7。其损失函数形式通常为 ![][image9] 42。  
* **性能特征**：迭代次数 ![][image10] 必须覆盖管状结构的最大像素半径，否则骨架化不完全 39。缺点是最大/最小池化产生的梯度较为粗糙（锯齿状），且针对粗大血管时 ![][image10] 值较大，引入了极大的计算开销和显存占用 16。  
* **入库建议**：核心损失函数（v0.1）。实现容易，通过循环池化操作即可完成。

#### **Skeleton Recall Loss**

* **适用场景**：管状组织与裂缝检测；为解决 soft-clDice 效率低下的改进方案。  
* **核心思想**：避免在模型的前向/反向传播中进行极其昂贵的可微骨架化。相反，在数据加载阶段（Dataloader），针对 Ground Truth 进行离线的硬骨架化，并进行一定的膨胀生成“管状骨架”（Tubed Skeleton）。训练时，计算网络预测概率图在预计算骨架上的“软召回率”（Soft Recall） 43。  
* **优缺点**：计算开销极低，极大提升了训练速度并解决了锯齿梯度问题 43。但仅侧重于召回（Recall），需与 Cross-Entropy 强绑定以抑制假阳性。强烈建议纳入（v0.1）。

#### **Persistent Homology (PH) Loss 与 Differentiable Euler Characteristic**

* **适用场景**：医学图像复杂拓扑分割；输入为似然图（Likelihood map）。  
* **核心思想**：通过下星形滤波（Lower-star filtration）构建图像的单纯复形，提取持续条形码。损失函数定义为预测图与真实图的条形码之间的 Wasserstein 距离，通过将梯度回传至引起拓扑特征生灭的“关键像素点”来实现可微 44。  
* **性能特征与预警**：对修正断裂和填补孔洞具有无可匹敌的理论精确度 44。然而，**强烈预警**：计算拓扑图的 Wasserstein 匹配在 3D 体素上具有近乎 ![][image11] 的复杂度，极耗内存，通常只能在极小的 Patch（如 16x16 或 32x32）上运算，难以适应大规模工业级模型 25。其轻量化替代方案为**可微欧拉示性数损失（Differentiable Euler Characteristic, DEC）**，通过规避完整的同调配对，极大降低了计算复杂度 47。

### **2\. 遥感道路提取与图学习中的可微优化损失**

在道路图重建中，网络往往预测代价图（Cost Map），而后求解最短路径。直接求解最短路径是离散操作，无法反向传播。

#### **LogSumExp Shortest Path (Softmin Path Loss)**

* **适用场景**：图提取、路由代价预测；端到端可微。  
* **核心思想**：使用 LogSumExp（平滑最大值/最小值）函数，对经典的动态规划最短路径算法（如 Bellman-Ford）进行连续松弛（Continuous Relaxation） 49。由于 ![][image12]，它为组合优化提供了平滑的梯度 49。  
* **缺点**：当路径变长时，指数累加会导致极其严重的数值不稳定，极易引发梯度消失或梯度爆炸问题 50。

#### **DataSP 与 Newton Losses (利用海森矩阵)**

* **核心思想**：针对 LogSumExp 和其他受扰动最短路径算法的梯度失效问题，DataSP 引入了可微的全源最短路径（Floyd-Warshall）并在循环内嵌入约束 52。而 Newton Losses 则利用二阶曲率信息（如通过随机平滑估计的 Hessian 矩阵或 Fisher 矩阵），将高度非凸的算法损失局部转化为二次近似，使得标准的梯度下降能够顺利训练 50。这使道路提取精度提高了惊人的 20% 以上 50。  
* **入库建议**：架构复杂（v1.0）。需要专门构建“算法层”（Algorithmic Layer），处理动态规划的反向传播。

## ---

**第三部分：核心指标的深度横向对比（APLS 与 TOPO-F1）**

构建拓扑工具库时，必须认清不同基准指标的局限性，以便提供完善的评价闭环。

**APLS 与 TOPO-F1 的不足：**

1. **APLS 的局限**：APLS 依赖于全局最短路径搜索，对“未连接到主图上的孤立伪连接环（Isolated Spurious Loops）”缺乏惩罚能力，因为随机采样的节点对很难落入孤立环中。此外，APLS 对节点吸附（Snapping Buffer）的阈值极其敏感，轻微的几何漂移若超出缓冲半径，会导致原本连通的路径被判为死胡同 4。  
2. **TOPO-F1 的局限**：TOPO 专注于固定半径内的局部可达性计算 11。它严厉惩罚局部的微小断裂，但完全丧失了对跨越整个城市的干线网络连通效率（全局路由能力）的评价维度 9。

**如何互补：**

* **空间对其互补 \-\> Graph IoU**：利用 Graph IoU 来强制校验图结构在宏观几何上的准确覆盖 14。  
* **孤立假阳性互补 \-\> Skeleton Precision**：通过衡量预测骨架在真实掩码中的命中率，消除 APLS 无法惩罚的孤立假阳性分支。  
* **精准空间错位互补 \-\> Betti Matching Error**：弥补 TOPO 在局部半径外的感知缺失，通过持续同调精准定位并计数预测中所有的虚假拓扑环与异常断裂点 22。

## ---

**第四部分：方法论分类体系 (Taxonomy)**

为了在工具库中进行合理的模块化设计，我们将上述方法按照输入和计算范式进行分类归纳：

1. **Mask-based (基于掩码重叠)**：Graph IoU, Connected Component Error。计算效率最高，支持大规模批处理，仅适合几何与连通体量评估。  
2. **Skeleton-based (基于形态学骨架)**：clDice, soft-clDice Loss, Skeleton Recall Loss, Skeleton Precision/Recall。在连续管状流形上评估能力强，但容易受“毛刺”和边界噪声干扰。  
3. **Graph-based (基于图组合优化)**：Graph Edit Distance, Tree Edit Distance, TOPO, Junction Matching。针对离散顶点和边，精确匹配微观连接状态，但图同构复杂度高。  
4. **Path-based (基于路径路由)**：APLS。以导航与流量的视角审视网络，强调结构的功能性。  
5. **Persistent-Homology-based (基于代数拓扑)**：Betti Number / Matching Error, PH Loss。基于严格的数学保证，洞察数据空间的本征孔洞与连通分量，计算代价极高。  
6. **Distance-field-based (基于距离场与几何)**：Geodesic Distance Loss, Chamfer Distance, Hausdorff Distance。捕获非欧几何面上的距离扭曲与空间几何偏离。  
7. **Differentiable-optimization-based (基于可微算法)**：LogSumExp Path Loss, Newton Losses。将经典图算法嵌入神经网络前向传播，是实现端到端图生成的核心技术。

## ---

**第五部分：开源 Python 工具库开发评估**

基于上述分类与剖析，在设计该 Python 工具库（暂命名为 TopoMetrics）时，需对各算法的实现难度与运行效率有清晰预判。

### **实现难度评估**

* **低难度 (Low)**：Graph IoU，Skeleton Precision/Recall，Chamfer Distance，Skeleton Recall Loss。此类方法仅涉及基础的 Tensor 操作（交集、并集、距离变换），利用 torch 或 scipy 原生 API 即可无缝实现且易于保证数值稳定。  
* **中难度 (Medium)**：clDice, soft-clDice, TOPO-F1。需要实现迭代的形态学池化操作，或者借助 NetworkX 进行局部 BFS 遍历搜索。难点在于保持张量维度一致性以及边缘条件（如管径动态计算）的处理。  
* **高难度 (High)**：APLS, Betti Matching Loss, Differentiable Shortest Path (Newton Losses)。APLS 必须实现复杂的欧式空间节点与几何线段的精确吸附（Node Snapping）算法 4。Betti 系列涉及将像素网格构建为单纯复形并提取条形码，原生 Python 效率不可忍受，必须编写 C++ 并通过 pybind11 接入 26。可微最短路径则面临黑盒算法的梯度截断与自定义 torch.autograd.Function 的复杂推导。

### **运行效率评估**

* **适合 Batch Evaluation**：mask-based 与 skeleton-based（如 clDice、Skeleton Recall Loss）。张量运算高度契合 GPU CUDA 并行加速，可毫无阻力地融入训练 DataLoader 和 Validation 流程 43。  
* **不适合大规模评估 (需离线或 CPU 运算)**：GED, APLS, Betti Error。全源最短路径及持续同调计算在批处理（Batch Size \> 16）时将遭遇内存爆炸或 CPU 瓶颈。此类指标应设计为仅在 Epoch 结束时的最终评价阶段（Validation/Test Phase）调用。  
* **需要极致加速**：soft-clDice 的多次 Min/Max Pooling 极其消耗显存；PH Loss 需要依赖预编译的高性能 TDA 工具包（如 GUDHI、Giotto-tda 或 CubicalRipser）进行加速 25。

## ---

**第六部分：开源仓库 (TopoMetrics) 功能与架构设计建议**

结合理论调研与工程经验，建议的工具库架构如下，旨在提供学术界开箱即用且高度可扩展的拓扑验证基准。

### **1\. Package Structure (包结构设计)**

Python

topometrics/  
├── metrics/               \# 纯评估指标，支持 numpy 和 torch tensor  
│   ├── mask.py            \# Graph IoU, 连通分量误差  
│   ├── skeleton.py        \# clDice, Skeleton Precision/Recall  
│   ├── routing.py         \# APLS, TOPO, TOPO-F1 (自动调用 C++/Go 底层)  
│   └── algebraic.py       \# Betti Matching Error, 欧拉示性数  
├── losses/                \# 继承自 torch.nn.Module，用于网络训练  
│   ├── morphological.py   \# Soft-clDice Loss, Skeleton Recall Loss  
│   ├── differentiable.py  \# Differentiable Shortest Path, LogSumExp  
│   └── topology.py        \# PH Loss, Betti Loss  
├── graph/                 \# 核心数据流组件  
│   ├── builder.py         \# Mask转Graph, 骨架化提取算法  
│   └── schema.py          \# Unified TopoGraph Schema  
├── utils/  
│   ├── snapping.py        \# 节点吸附与几何匹配算法  
│   └── c\_ext/             \# 加速核心：C++/Cython 编译模块  
└── viz/                   \# 可视化：持续同调条形码、拓扑错误热力图

### **2\. Graph Schema Design (统一图架构)**

库的核心在于统一不同任务的输入。需设计一个 TopoGraph 数据结构。无论输入是遥感的多段线（Polylines）、医学的二值体素（Mask），还是预测的概率图，都首先由 graph.builder 转化为标准的 TopoGraph。

* 结构包含：nodes (坐标张量, Nx2 或 Nx3), edges (连接索引, Ex2), features (如血管半径、道路粗细)。  
* 兼容性：原生支持无缝导出至 NetworkX（用于经典指标算法）和 PyTorch Geometric (PyG)（用于图神经网络特征学习）。

### **3\. Metric 与 Loss API Design**

**Metric API**：保持与 scikit-learn 和 torchmetrics 高度一致的接口设计，确保用户无需重构评测脚本。

Python

\# 示例：评估医学血管的 clDice  
from topometrics.metrics import clDice  
score \= clDice(pred\_mask, gt\_mask, voxel\_spacing=(1.0, 1.0, 1.0))

\# 示例：评估遥感道路 APLS  
from topometrics.metrics import APLS  
apls\_score \= APLS(pred\_graph, gt\_graph, snapping\_tolerance=8.0) \# 8米吸附容差

**Loss API**：提供混合优化的灵活接口，允许与 BCE 或 Dice Loss 组合。

Python

from topometrics.losses import SoftClDiceLoss  
criterion \= SoftClDiceLoss(iterations=10, alpha=0.5)   
loss \= criterion(pred\_prob, gt\_mask)

### **4\. 优先级列表 (Release Roadmap)**

* **v0.1 (基础形态学版)**：覆盖 clDice, soft-clDice, Skeleton Precision/Recall, Graph IoU 与 Skeleton Recall Loss。这些无需图转化，仅基于张量，能迅速占领医学血管/神经与简单裂缝检测用户群。  
* **v0.2 (图与遥感路由版)**：覆盖 APLS, TOPO/TOPO-F1。引入核心的 TopoGraph 架构与空间吸附算法，解决遥感与自动驾驶道路提取的需求。  
* **v1.0 (代数拓扑与可微算法版)**：整合 Betti Matching Loss, PH Loss, 及 LogSumExp/Newton Losses。包含 C++ 加速拓展包，满足顶会论文的极限前沿研究需求。

### **5\. Benchmark Datasets & Validation Protocol**

工具库应当内置微型数据集加载器与标准验证协议，确保结果可复现：

* **遥感验证集**：SpaceNet (城市道路)、CityScale (复杂交互网络) 10。用于严格校对库中 APLS 和 TOPO 实现与官方基准的一致性。  
* **医学验证集**：DRIVE (视网膜 2D)、CREMI (神经元)、TopoCows (脑血管 3D) 22。  
* **Validation Protocol**：提供统一的 benchmark.py。特别提示：对 APLS 和 TOPO，必须在文档中明确由于图转化策略（如分支合并）与吸附阈值的微小差异，可能导致与原论文产生的百分比级波动。

## ---

**结语**

像素重叠度指标在连续性拓扑结构面前的失效，催生了百花齐放的拓扑感知方法。本报告深入拆解了涵盖物理路由、局部同构、形态同伦与代数同调等四个维度的评估指标体系，并剥析了软化池化、算法松弛与持续同调三条构建微分拓扑损失的主路径。通过构建基于上述体系架构设计的 TopoMetrics 开源工具库，将极大地赋能遥感道路重建与医学体素分割领域，使得“拓扑正确性”不再是一项难以调优的昂贵特性，而是可以被标准化度量、端到端学习的基础视觉范式。

#### **引用的著作**

1. Sat2Graph: Road Graph Extraction through Graph-Tensor Encoding \- ECVA | European Computer Vision Association, 访问时间为 五月 3, 2026， [https://www.ecva.net/papers/eccv\_2020/papers\_ECCV/papers/123690052.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690052.pdf)  
2. 3D vessel segmentation \- DLMA: Deep Learning for Medical Applications \- BayernCollab, 访问时间为 五月 3, 2026， [https://collab.dvb.bayern/spaces/TUMdlma/pages/73379955/3D+vessel+segmentation](https://collab.dvb.bayern/spaces/TUMdlma/pages/73379955/3D+vessel+segmentation)  
3. Biagioni and Eriksson 1 INFERRING ROAD MAPS FROM GPS TRACES: SURVEY AND COMPARATIVE, 访问时间为 五月 3, 2026， [http://www.biagioni.net/papers/biagioni-trb12.pdf](http://www.biagioni.net/papers/biagioni-trb12.pdf)  
4. SpaceNet Road Detection and Routing Challenge Part II — APLS ..., 访问时间为 五月 3, 2026， [https://medium.com/the-downlinq/spacenet-road-detection-and-routing-challenge-part-ii-apls-implementation-92acd86f4094](https://medium.com/the-downlinq/spacenet-road-detection-and-routing-challenge-part-ii-apls-implementation-92acd86f4094)  
5. CosmiQ/apls: Python code to evaluate the APLS metric \- GitHub, 访问时间为 五月 3, 2026， [https://github.com/CosmiQ/apls](https://github.com/CosmiQ/apls)  
6. htcr/sam\_road: Segment Anything Model for large-scale ... \- GitHub, 访问时间为 五月 3, 2026， [https://github.com/htcr/sam\_road](https://github.com/htcr/sam_road)  
7. jocpae/clDice · GitHub \- GitHub, 访问时间为 五月 3, 2026， [https://github.com/jocpae/clDice](https://github.com/jocpae/clDice)  
8. SpaceNet Roads Extraction and Routing Challenge Solutions are Released | by David Lindenbaum | The DownLinQ | Medium, 访问时间为 五月 3, 2026， [https://medium.com/the-downlinq/spacenet-roads-extraction-and-routing-challenge-solutions-are-released-48830c5184ef](https://medium.com/the-downlinq/spacenet-roads-extraction-and-routing-challenge-solutions-are-released-48830c5184ef)  
9. Learning and Aggregating Lane Graphs for Urban Automated Driving \- CVF Open Access, 访问时间为 五月 3, 2026， [https://openaccess.thecvf.com/content/CVPR2023/papers/Buchner\_Learning\_and\_Aggregating\_Lane\_Graphs\_for\_Urban\_Automated\_Driving\_CVPR\_2023\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Buchner_Learning_and_Aggregating_Lane_Graphs_for_Urban_Automated_Driving_CVPR_2023_paper.pdf)  
10. Solving SpaceNet Road Detection Challenge With Deep Learning | NVIDIA Technical Blog, 访问时间为 五月 3, 2026， [https://developer.nvidia.com/blog/solving-spacenet-road-detection-challenge-deep-learning/](https://developer.nvidia.com/blog/solving-spacenet-road-detection-challenge-deep-learning/)  
11. Segment Anything Model for Road Network Graph Extraction \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2403.16051v3](https://arxiv.org/html/2403.16051v3)  
12. \[2602.23290\] LineGraph2Road: Structural Graph Reasoning on Line Graphs for Road Network Extraction \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/abs/2602.23290](https://arxiv.org/abs/2602.23290)  
13. Reconstructing Road Network Graphs from both Aerial Lidar and Images \- CVF Open Access, 访问时间为 五月 3, 2026， [https://openaccess.thecvf.com/content\_WACV\_2020/papers/Parajuli\_Reconstructing\_Road\_Network\_Graphs\_from\_both\_Aerial\_Lidar\_and\_Images\_WACV\_2020\_paper.pdf](https://openaccess.thecvf.com/content_WACV_2020/papers/Parajuli_Reconstructing_Road_Network_Graphs_from_both_Aerial_Lidar_and_Images_WACV_2020_paper.pdf)  
14. DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2508.13669v2](https://arxiv.org/html/2508.13669v2)  
15. DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2508.13669v1](https://arxiv.org/html/2508.13669v1)  
16. Beyond Conventional Losses: Skeleton-Based Loss for Preserving Connectivity in Crack Segmentation \- MDPI, 访问时间为 五月 3, 2026， [https://www.mdpi.com/2673-7590/5/4/177](https://www.mdpi.com/2673-7590/5/4/177)  
17. clDice \- a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation, 访问时间为 五月 3, 2026， [https://www.researchgate.net/publication/355885480\_clDice\_-\_a\_Novel\_Topology-Preserving\_Loss\_Function\_for\_Tubular\_Structure\_Segmentation](https://www.researchgate.net/publication/355885480_clDice_-_a_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation)  
18. Segmentation-assisted vessel centerline extraction from cerebral CT Angiography \- PURE.EUR.NL., 访问时间为 五月 3, 2026， [https://pure.eur.nl/ws/files/200206484/Segmentation-assisted\_vessel\_centerline\_extraction\_from\_cerebral\_CT\_Angiography.pdf](https://pure.eur.nl/ws/files/200206484/Segmentation-assisted_vessel_centerline_extraction_from_cerebral_CT_Angiography.pdf)  
19. Centerline Boundary Dice Loss for Vascular Segmentation | MICCAI 2024 \- Open Access, 访问时间为 五月 3, 2026， [https://papers.miccai.org/miccai-2024/129-Paper0458.html](https://papers.miccai.org/miccai-2024/129-Paper0458.html)  
20. Impact of Connectivity-Preserving Loss Functions on the Segmentation of Thin Tubular Structures: Application to Coronary Arteries From CT \- Hannes Nickisch, 访问时间为 五月 3, 2026， [https://hannes.nickisch.org/students/krnjaca25master.pdf](https://hannes.nickisch.org/students/krnjaca25master.pdf)  
21. Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes, 访问时间为 五月 3, 2026， [https://proceedings.mlr.press/v202/stucki23a.html](https://proceedings.mlr.press/v202/stucki23a.html)  
22. nstucki/Betti-matching · GitHub \- GitHub, 访问时间为 五月 3, 2026， [https://github.com/nstucki/Betti-matching](https://github.com/nstucki/Betti-matching)  
23. Topologically faithful image segmentation via induced matching of persistence barcodes \[Quick Review\] \- Liner, 访问时间为 五月 3, 2026， [https://liner.com/review/topologically-faithful-image-segmentation-via-induced-matching-persistence-barcodes](https://liner.com/review/topologically-faithful-image-segmentation-via-induced-matching-persistence-barcodes)  
24. Pitfalls of topology-aware image segmentation \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2412.14619v1](https://arxiv.org/html/2412.14619v1)  
25. A Survey of Topological Machine Learning Methods \- Frontiers, 访问时间为 五月 3, 2026， [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.681108/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.681108/full)  
26. nstucki/Betti-Matching-3D \- GitHub, 访问时间为 五月 3, 2026， [https://github.com/nstucki/Betti-Matching-3D](https://github.com/nstucki/Betti-Matching-3D)  
27. Efficient Computation of the Tree Edit Distance \- ResearchGate, 访问时间为 五月 3, 2026， [https://www.researchgate.net/publication/277568990\_Efficient\_Computation\_of\_the\_Tree\_Edit\_Distance](https://www.researchgate.net/publication/277568990_Efficient_Computation_of_the_Tree_Edit_Distance)  
28. Three-Dimensional Neuron Tracing by Voxel Scooping \- PMC \- NIH, 访问时间为 五月 3, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC2753723/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2753723/)  
29. RTED: A Robust Algorithm for the Tree Edit Distance \- VLDB Endowment, 访问时间为 五月 3, 2026， [https://vldb.org/pvldb/vol5/p334\_mateuszpawlik\_vldb2012.pdf](https://vldb.org/pvldb/vol5/p334_mateuszpawlik_vldb2012.pdf)  
30. PyNeval: A Python Toolbox for Evaluating Neuron Reconstruction ..., 访问时间为 五月 3, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC8831325/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8831325/)  
31. Computing Approximate Graph Edit Distance via Optimal Transport \- Indiana University Bloomington, 访问时间为 五月 3, 2026， [https://homes.luddy.indiana.edu/qzhangcs/papers/sigmod25-EDviaOT.pdf](https://homes.luddy.indiana.edu/qzhangcs/papers/sigmod25-EDviaOT.pdf)  
32. Graph Edit Distance with General Costs Using Neural Set Divergence \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2409.17687v2](https://arxiv.org/html/2409.17687v2)  
33. Graph Edit Distance Evaluation Datasets: Pitfalls and Mitigation \- OpenReview, 访问时间为 五月 3, 2026， [https://openreview.net/forum?id=guapIeLs02](https://openreview.net/forum?id=guapIeLs02)  
34. An efficient algorithm for graph edit distance computation \- ResearchGate, 访问时间为 五月 3, 2026， [https://www.researchgate.net/publication/328300201\_An\_efficient\_algorithm\_for\_graph\_edit\_distance\_computation](https://www.researchgate.net/publication/328300201_An_efficient_algorithm_for_graph_edit_distance_computation)  
35. The Robust Vessel Segmentation and Centerline Extraction: One-Stage Deep Learning Approach \- MDPI, 访问时间为 五月 3, 2026， [https://www.mdpi.com/2313-433X/11/7/209](https://www.mdpi.com/2313-433X/11/7/209)  
36. Multiscale Cross-Axis Position Attention and Edge-Constrained Knowledge Distillation for Road Extraction in High-Resolution Remote Sensing Images \- IEEE Xplore, 访问时间为 五月 3, 2026， [https://ieeexplore.ieee.org/iel8/4609443/10766875/11207133.pdf](https://ieeexplore.ieee.org/iel8/4609443/10766875/11207133.pdf)  
37. RouteExtract: A Modular Pipeline for Extracting Routes from Paper Maps \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2509.11674v1](https://arxiv.org/html/2509.11674v1)  
38. A U-Net Deep Learning Framework for High Performance Vessel Segmentation in Patients With Cerebrovascular Disease \- PMC, 访问时间为 五月 3, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC6403177/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6403177/)  
39. clDice \- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation \- CVF Open Access, 访问时间为 五月 3, 2026， [https://openaccess.thecvf.com/content/CVPR2021/papers/Shit\_clDice\_-\_A\_Novel\_Topology-Preserving\_Loss\_Function\_for\_Tubular\_Structure\_CVPR\_2021\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)  
40. Centerline Boundary Dice Loss for Vascular Segmentation \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2407.01517v1](https://arxiv.org/html/2407.01517v1)  
41. JunMa11/SegLossOdyssey: A collection of loss functions for medical image segmentation \- GitHub, 访问时间为 五月 3, 2026， [https://github.com/JunMa11/SegLossOdyssey](https://github.com/JunMa11/SegLossOdyssey)  
42. SELF-SUPERVISED LEARNING TO IMPROVE TOPOLOGY-OPTIMIZED AXON SEGMENTATION AND CENTERLINE DETECTION \- PMC, 访问时间为 五月 3, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12153109/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12153109/)  
43. GitHub \- MIC-DKFZ/Skeleton-Recall: Skeleton Recall Loss for ..., 访问时间为 五月 3, 2026， [https://github.com/MIC-DKFZ/Skeleton-Recall](https://github.com/MIC-DKFZ/Skeleton-Recall)  
44. Topology-Aware Focal Loss for 3D Image Segmentation \- CVF Open Access, 访问时间为 五月 3, 2026， [https://openaccess.thecvf.com/content/CVPR2023W/TAG-PRA/papers/Demir\_Topology-Aware\_Focal\_Loss\_for\_3D\_Image\_Segmentation\_CVPRW\_2023\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023W/TAG-PRA/papers/Demir_Topology-Aware_Focal_Loss_for_3D_Image_Segmentation_CVPRW_2023_paper.pdf)  
45. Topology-Aware Loss for Segmentation in Computed Tomography Images \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2307.03137v3](https://arxiv.org/html/2307.03137v3)  
46. Time-Series Forecasting via Topological Information Supervised Framework with Efficient Topological Feature Learning \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2503.23757v1](https://arxiv.org/html/2503.23757v1)  
47. Differentiable Euler Characteristic Transforms for Shape Classification \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/html/2310.07630v3](https://arxiv.org/html/2310.07630v3)  
48. ICLR Poster Differentiable Euler Characteristic Transforms for Shape Classification, 访问时间为 五月 3, 2026， [https://iclr.cc/virtual/2024/poster/18824](https://iclr.cc/virtual/2024/poster/18824)  
49. LogSumExp \- Wikipedia, 访问时间为 五月 3, 2026， [https://en.wikipedia.org/wiki/LogSumExp](https://en.wikipedia.org/wiki/LogSumExp)  
50. Newton Losses: Using Curvature Information for Learning ... \- arXiv, 访问时间为 五月 3, 2026， [https://arxiv.org/pdf/2410.19055](https://arxiv.org/pdf/2410.19055)  
51. What Is the Log-Sum-Exp Function? \- Nick Higham, 访问时间为 五月 3, 2026， [https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/](https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/)  
52. A survey on traffic flow prediction and classification \- ResearchGate, 访问时间为 五月 3, 2026， [https://www.researchgate.net/publication/373229143\_A\_survey\_on\_traffic\_flow\_prediction\_and\_classification](https://www.researchgate.net/publication/373229143_A_survey_on_traffic_flow_prediction_and_classification)  
53. Learning Costs of Constrained Optimization Problems for Structured Inference \- DiVA, 访问时间为 五月 3, 2026， [https://oru.diva-portal.org/smash/get/diva2:2030588/FULLTEXT01.pdf](https://oru.diva-portal.org/smash/get/diva2:2030588/FULLTEXT01.pdf)  
54. Learning by Sorting: Self-supervised Learning with Group Ordering Constraints | Request PDF \- ResearchGate, 访问时间为 五月 3, 2026， [https://www.researchgate.net/publication/377425734\_Learning\_by\_Sorting\_Self-supervised\_Learning\_with\_Group\_Ordering\_Constraints](https://www.researchgate.net/publication/377425734_Learning_by_Sorting_Self-supervised_Learning_with_Group_Ordering_Constraints)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAyklEQVR4XmNgGJZAGogLgHgmECshiVshsTHAYiD+D8S3gdgbiFWBeBoQPwdiS6gcVgCS+A3E3OgSQFDJAJG/hC4BAn8Y8JgKBSD5IHTBD1AJZnQJNIBhuC5U8CG6BBaAofkvVJAXXYIYANKIYSKxAJ9mfyB2BmJ7IHYAYhcGtHABaXyNLIAEsoG4ngFhQTkQMyErwGczDIDkb6ELgsA1BogkO7oEFOQyQOTD0SVgAGY7ipOAQA5JDi/Yy4BQ+A5KN0Ll1sAUjYIhBwDP2zLKnm6VTgAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAYCAYAAAD6S912AAAA9ElEQVR4XmNgGAmABYgN0QVxAWkgLgDimUCshCRuhcQ+A8T/kfhYwWIGiKLbQOwNxKpAPA2InwOxJVQOBkDstUh8DABS8BuIudElgKCSASJ/CUkMxGdC4qOAPwyEnQ+SD4KyfRkglmMFHxggipnRJdAAsoWngdgDiQ8HugwQhQ/RJbAA9PDDCv4yQCR50SXIBSDDcNpGDsBnoD8QOwOxPRA7ALELA+FwBhv2Gl0QCrKBuJ4BYWk5A55kAgP4XAgDIPlb6IK4wDUGiAZ2dAkoyGWAyIejS+ADMFeie0cOSY5ksJcBofkdlG6Eyq2BKRoFo4DKAABjTDrRtIX4eAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAYCAYAAABeIWWlAAACXklEQVR4Xu2WPWgUQRTHn5hEg5GgCKaIWGhsBBFsRNJoKViEtGkU1EDAUhIVQRTttLBRhIhYiK2YFBY2IZBCQS20tNIgafwC0fj1/jcz3vO/M7tzuY2F3A8et/N7b2eHuZ2ZFenQYZTFP6BLYzPLurmnMcyyRro1drP0vNfYyDLGtMZHjV8mPmhctUXEYY1HLGtkSZpjSVGWK1DVmSW3rh3wjK8sDZc1nrCMsUZcZ085EeGMxjOWqwDGc5ElgRqMvZQJcYV43apA3X6WNbND3HN6OUH80LjEknkn+a9aTt0xjWuSMasJ7krzObjG2xLjhsZ3lkzuesM/W1a3S1x+u2//9O2TfyryCON54NtXxP1LzAEpH08DFCywjHBB0p2tF5c7btwR79YalwPueRhxe8kNeJ+kar0dMtd3JN3ZFynm7OuVy6C4ezYY1+PduHGB0v7DmZICZ17gtqRr4Rcj7jO5Km5K8RmnvNtEHnDtXyCZKjincdC0JyVei90TntcW3BS5Kt5K8RmYYHZgq8R9A6wFJJ9zQtkmxRtHIg5skaLf6d06jSGNoyZ3Wtw3YozHGq/IoR/8e0yY1CjYrpHEoC3YYuFjh3WqM/gxfx02l1D70v+C8Ip9Ms7CE4hl89q0LdclMh4s9G/ittewXYdAG2cHNggsbgY1+1iKO3BxH/Iz3r3x7ROhyIPBlp1P56U5nrLXelncMVEbsxrzLFdAYcZXQB19FGi3U6zDWyxb5KzGC5Z1gO+5+yxbIPa10SrtTnApcxp7WGbSx6JFcJ72s6yb2BfDaoONC8dUh/+K3/BVnz2Rg4soAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAYCAYAAACldpB6AAACtUlEQVR4Xu2YS4iOURjHH+RaJKZYkA1KZKPEZIOSa7lkIzUh94zlDNkoZIOFjRWhSBZWFClJSmpqbNgoSpKsXCJye/5zzpk5/u857/fM+71fWXy/eprv/M7/Pe/5zpz3MiPSpk3dbGFhpItFDazQGhm1N0efW8Z1rWUsDUzW+qM1nzsMTNWax9KDMW9GbfyCLkbtUhD8JG6QUB+1zsYhYq3WPZZGzok7x3BYp/Vd3HEPqA9MEdc3mvxTrW3kSgkLYMGaS/FL6wpLIzjvSpbiFvY3S495riPEhfu4I8FRrX6Ww8A8KWKX5I/Fwm5l6XmtdZ5lioPiToBt3gjklrA0MknrK0sjbyS/CDkP1kt5/yDvxRgUW26nuC2KHRZzUmspOSs470OtOVq3tBZ4P0HrbghlsMx5IGQJYqeU5eaK65/l27hO0d47mKhOmONqGbp8T/2TyINsw8VH6AnLBMclvwjjxPXtjtwG70ZFrgpd4saZHjk81XJzYZDbwTKm0f0ALyGBy5I/8Tcp9l1NuCq8lOI4uDTY5UDuNMuYD1I+GN4ZApckn4V/l3BfyFUB4/BOhcvNhUHuDMuYssGOaS2P2r2SzuJpAc/XPtwRclXAOBsT7hW5HMh2swzgWkXgGXcoM6X4hTclHOiQop/t3Vhxd/Sya3K81mGWERhnWtTe750VZDtZBsIrLL5czAXvUy9FuZPDb/efw00yZJ/7nzlCdjF3ePAyFBZpjLjsmqHuhiTnjBvWD3GDh8dYKLR/irvRzQgHRCCziKW43yaOQ/9t79769p4QyoC/+JDFjTfHZ3Fj4R6DnWdllWQWoRnuaD1mWQMTtXpY1sALrUMs66D2lZXyXdAMrZjrACe0brBsAvyf4D7LGsCYeMNsGY+0FrKsCJ4gdYOXvGssW8E+Fv8RB1i0aVPkL3SBrkuS2q4KAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARIAAAAjCAYAAABYZoWfAAAEwklEQVR4Xu3da8gUZRQH8BNpZdmNlPISEQSZXVTySwgGBdkH0aLoImIEoqWVF9TAQI3IgrKyC6EEEQQV3T4EBYGCn0RFFFGjD0plauWHEJKsLD1/n+dhzxxn35nZ2d333d7/Dw77zJmdeXdnZ88888zuviJERERERERERJRnmcaTPklEVOQjjb81TsdYkJ1NRFQNCwkR1cZCQkS1sZAQUW0sJERUGwrJQp8kIqoCheRpn2ziVo3JJWN4XIaIBgEUkmd8sol7NGaUjJFxGSIaBFBIFvskEVEVKCRLfJKIqKwREgrJq34GEVGRTzWOafyscSje/irhY/NEREREREREREREREREREREPegKCZ+nqgNfM/nNJ4na5WKNwz5JA8o7Ur+QYPltPtkifp6LMoZp/OuTNOCgCKz2yYqwjnZ+471uYaP/EewMKCbdMN8nCtyscYtPdtl+n2iz2RqX+GSOum9afNO92TqO+kQ0yiecmzRO+iQNPrs1vvXJDnrCJ4y3JezoQ+L0+RK+JjEQHPeJFm2Q7Jv5Gin3g11XSfiaSB1vaszzSQP7gjfaJ3KgCOG1o0FqrDQ/QnVK0f8mso+n0z2BKlZoTPTJFtnn+I9p92WdxhifrKjo9PVrObdnWqaQnCfhOeG2Z03S2O6TVMpfGh/6ZIf19bu74zT+NNPdLnJFfveJFqBHtjm2L9DYY+b1N1wV+srlyhQSOKix1yd7xSoJI9ksJNVhB8EbFacP3eS78dh5cTkS3epN0hhMxIDg++lO0RoJO+v4OI3Hf7nG1RpfxukiByR7v9R+QONZjdc0TjVmZ5RZfx6M8/ygcUTjF407Y366xmOxnXwjoWCl07tW/ia+JZ+eQ9VLxv6+ZQsJDgJ+2Z7yvNQrJNjQ92k8qPGQxsMaj2g8WiF6ES4B9scL/5Rppy5xgvZFsY2fwfQ/ZDVNwoHj5Tj9uWRPN4qez/3xFl38y2I7nVpg2dSt/07jk9i2itafB29Ee0ph14GihR51gvGSKzX2adwbczjSAwact0jxY1gfb/12hVckDHaj6DbrXfn1ly0kgGXxnHpS3UKCIx82AAIvFo52GIlGoI3cbRq3SziCvCih4qdl/IZvNxS6TsDjXuuTXWB/wPsnjffMtN2W+BEr9BI8e5+Zpg1lXwt7v13x9gOTQ/f+ezOd2OUwaInTsGaRCiKKCK7KJHYdOyQMonr2Pq+bNpR5juil2ful08X/TA7z88ar/PqrFJIyhW7AQiHBC1LHcgkb4A8/o8BKCcst9TPa5EIJ62/3i5MuA97gZzSxTPJ3ulbYQoLHcG1s47HYy4jouSwy04ndFviRK6vMdsIRGTt88oZpJ1jP9T4p5dbv2WXmSvZvf6YxwUzDjZL9oJe/klLmMeB/ZuN9kaDHDV+YHK5C4QqS59dfpZDMkXOX7xnYYDt9sgXYKbER3vIzCqBr3cmNhyOaHwCrKxXAvnTqH7jbQnLCtNGrfEHj3Tg9RWNjY/ZZ+PRtOhW5VOMOMw/yntN1kh3AfUnC2Bo8bvLJx9I4rfDy1l/ELoPTibslnDrBcxqzGrPPwulbGozFv1VB78LKewzomf1oprdqTI1t29OysJ688TG//iqFZKiE5e/yM/rLGWJnAKVWJr3cAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGgAAAAeCAYAAADAZ1t9AAACN0lEQVR4Xu2awUsVURTGb5ZGhpKSusg2GhluRUwLdBu0iXYu3EWKGxdBy5L+gUDwH1A3rl0ILgx04doWQq4ENQM3SqWYpudj7rUz38yzJ45v4L3zg493zncfOsOZmXvm3uecYRiGYeTPc9Gu6FS0IqqKDxt5MiGaVPlvFxWqTXlGjqAYPSkeZOTMXZdejDQvjUE2iGpRHZvG5fgkekZesQVqEe2x6UHxj9g0sgHF+ctmAVpFP8lDcf6QZ2TEqosKVMsDF/BQ9MvHKM6xGjMyBM0CitPEA0UQimTFuSYaXFSc2zxQJI0ueqwd8IBxdfBiyk3BFOUXEYoD7rvoXcrIkLSG4ISNAuDO44YARQpzknFF0AqHtpr1P+pFh2x6UCTu7oxL8sAlixJUzFwyxgaBl1QUyjAMwzAMwzAqGLTXX0WLPt4QLYj2fZ43r110HNiWX/bxkuiLjz+ff7MM+Si6qXKc8A2Vo2h5o1cn2l38oqkR9au87NhS8SOXvGNGKc+acTY8AyoeVvG06IfKn6i47JkR7bBZAtYo36RcgwtohM1KASevr9ZS8s1/bsfcJHyHVxRpJ98t6vDxB+X3uWjbG/PVG+WDLpd8NGJr452ol3xN2v/X4HFW6DtY/3svaiafj2XI/VsnxJoidoKZt6KnbOZNp0uePJoH/EAE/ivvIcbEfE/5WL3G5B3Gb/k4NBgvRbM+TtviAOHOWY+5cfA3vrMp3BHN+Vjv7PKxPFb+CxVrQj4fc0vIGSpzd9t6MV4hAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAABGElEQVR4XmNgGPGAF4inAHExugSpYBUQd0PZKkD8H0mOJFABxDZoYiDDatDEiAK30QUYIIY1oQsSAtOAmBtdkAFiGCO6ICEACxtRILaHsv8gsUkCB4FYkgFiKAiDDPqCooJI4MSA3QUgQ+vRBQmBQ+gCUHAAiL8h8UuA+A4Q30cSwwC40tI/IP4MZYNcPxtJ7hcSGwWcRBeAApAlU6Hsn0CsjCaHAVwYIE5HBwUMqBpAbFA2Q+abIvHB4AgQ/2VATUugyAAplkISA/E50fgRSHy4IAg8gLJBuBMuiwAgcR40vhUSHwxwxSQ6AGlGDzMuJD44vByRBfCAOiAuQ+JjRMBhdAEC4CYQzwHi30AsgybH4IsuMArwAgDvkTrHqxDJxAAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAA/ElEQVR4Xu2SOwrCQBRFn7VYWmjtKkQQxUpR3IVgJdhYCIKd2KqLcBX+GteghRuwED+Vv/fIjGSumUxS64ELj/OSa0yG6OdJcSacLi7iMueM1ZzjvHy7WPQ4BXBS1gcXiR0K8sqGKF3MOEmU5JUlULrQ7ybNKar57ptjseRkyCuVSNHFuCIiZQp+AikdoHSxQqFYcG7g2pwqOAPbWXpyzmqWI9Mi79rQsi0Khdw4DXDWsgpnj5LpUPATh5ZtOA8yz5J8DLkp63Ma8TWUGv3rBzVLRp/tN7Kvo9TYvqQNKWugFOR9lVA6kLImSmGNIoQ858Q5qlzNdch//xPIG+nhNXfkN/dtAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAAoElEQVR4XmNgGJSAEYhV0QWxgadA/B+KiQJXGEhQDFJ4DV0QFwApjkAXxAaiGDCd0ATE/mhiYHCTAaGYC4jvAzEfEH+Dq0ACIIW3gVgQiDdCxX5CxTEASHAnEM9El0AHMxgQJsyGslUQ0qgAPTJA7INQdj6SOBiAJKeh8VuQ2HDACRUQRRL7CMQbgLgHiA2RxMHAE10ACDyAmANdcBTAAACQdCSKrBERiwAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAACmElEQVR4Xu2XS6hOURTHl2ceIZNrhChDMpDnQCiPDBkYyM1AmTCRd5kpycAtAzKSoUdKmanLhAGFlFcZSPKOvB9h/e/a+377/M8657jfcT7S/dW/s/d/rb3P+fbZj/OJDPJP0Kt6qbpA/l9hORs1uaTaGcpHVZ+SGFhH9QExVazTI6rxFPPYrNrFZk32qg6F8mnVzyQGpqtuklfJYbGONoT6FNVz1ef+jDyTVU/YTMAUQ59RzEPJxu9lw33An8mm0qM6xqbHULFOMC08vqt+sBlAu1FsEudU18VyF1AMDFfdYVOZK/YDLnIgwRu0HEjCSBaxVCxnGfkLVV/I80DbkeH6lWJgi2o1mwlrpPiHnJSKqfpYihtH4hs+Rf43+b219zZckY9+8MZSnlIdXFNNC+XRYu2GtML9TJCS518sFuwln5kolveGfHi4eRmzVNtCeZFYm8utcB/eA8JbGcqrQr0IxNzNMI5o1RpaL5Z3I/HGBa+Ks6phSR1tuN0VqoM5YhvccbH1PyYbzoD+drMJvJt53BfLw3EQWRK8KjjnQPDiW0Wfdc9Q9HeCza4Q4Afw8PI2Op5HXH8paX/P0kCb4CPgKpuYNrgJfyEwa8Xy+AjpDn4Zs1Xb2VRui7WdEa51+SB2DOXw3gxTlDNffD/lvOR3TDBWrC3eLm847YC+sNZz4AZlD/lILD6CA9LaWcsoi2PjQBxruS7oZw+bEQRvsam8ENtly0BbHOAeW8Xi3tkFcASUDcBAQD8Y8ELi9yIWKtYkyvMyGT7Ii7thBFPynep10EfVikxGi1dstEGc7o2wQ/WezQ6Dfz1n2PyTYPS8jaRTNPb2IlhLD9jsEPtU+9lsgoOqTWw2zCTVXTabpJuNhsEuPch/yy/QhqleEBViEQAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,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>
