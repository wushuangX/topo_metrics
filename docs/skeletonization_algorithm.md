# CRESI 骨架化算法：从语义分割掩码到路网图

## 概述

本文档详细描述 `cresi/04_skeletonize.py` 实现的算法。其输入是道路分割掩码（模型预测或 Ground Truth），输出是像素坐标下的 WKT `LINESTRING` 集合——即从栅格（Raster）到矢量（Vector）的转变。

算法流程可概括为**四个阶段**：

```
输入掩码 .tif
  │
  ├── Phase 1: build_wkt_dir()        ← 多线程调度入口
  │     └── img_to_ske_G()            ← 每张图像（并行）
  │           ├── make_skeleton()      ← 形态学精炼 + 骨架化
  │           │     ├── preprocess()   ← 二值化 + 去噪/填洞
  │           │     └── skeletonize()  ← 单像素宽骨架提取
  │           ├── sknw.build_sknw()    ← 栅格骨架 → NetworkX 图
  │           └── 迭代毛刺修剪         ← remove_small_terminal()
  │
  └── Phase 2: G_to_wkt()             ← 图 → WKT（单线程）
        ├── graph2lines()             ← 边列表 → 节点路径
        ├── add_direction_change_nodes() ← cv2.approxPolyDP 简化
        ├── remove_duplicate_segments()  ← 去重
        └── add_small_segments()      ← 连接短间隙
```

---

## 第一阶段：build_wkt_dir() — 多线程调度

**位置**：第 950–1062 行  
**输入**：目录路径，内含 `.tif` 掩码文件  
**输出**：`DataFrame[ImageId, WKT_Pix]` → `.csv`

### 流程

1. **收集文件**：扫描 `indir` 中所有 `.tif` 文件，按字典序排序。
2. **构建任务参数元组**：对每张图像构建一个参数元组 `(img_loc, out_ske_file, out_gpickle, thresh, ...)`，包含后面所有阶段所需的配置参数。
3. **并行执行骨架化**：使用 `multiprocessing.pool.Pool.map()` 将 `img_to_ske_G()` 分配至 `n_threads` 个进程并行执行。
4. **回读 Pickle**：在 `out_gdir` 中列出所有生成的 `.gpickle` 文件。
5. **单线程图转 WKT**：对每个 `.gpickle` 文件调用 `G_to_wkt()`——WKT 序列化有大量字符串操作，不适合多线程，所以集中到单线程执行。
6. **写入 CSV**：将所有 `(ImageId, WKT_Pix)` 行收集到 `DataFrame`，保存为与 `config.wkt_submission` 对应的 CSV 文件。

### 图像 ID 提取

```python
if spacenet_naming_convention:
    im_root = 'AOI' + imfile.split('AOI')[-1].split('.')[0]
else:
    im_root = imfile.split('.')[0]
```

支持两种命名约定：SpaceNet 格式（保留 `AOI*` 前缀）和通用格式（不含 `.tif` 后缀的文件名）。

---

## 第二阶段：img_to_ske_G() — 单张图像的骨架化 + 建图

**位置**：第 714–847 行  
**输入**：单张掩码 `.tif` + 参数元组  
**输出**：`(G, ske, img_refine)` 三元组，其中 G 是 NetworkX `MultiGraph`

### 2.1 make_skeleton() — 形态学精炼 + 骨架化

**位置**：第 580–710 行  
**输入**：单张掩码图像路径  
**输出**：`(img_refine, ske)`——二值精炼掩码 + uint16 骨架

#### 步骤 1：读取掩码（第 601–621 行）

按 `num_classes` 参数分支：

**单通道（`num_classes == 1`）**：
```python
try:
    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
except:
    img = skimage.io.imread(img_loc, as_gray=True).astype(np.uint8)
```
- 优先使用 OpenCV 读取灰度图（速度快）。
- 若失败（非标准 TIFF 变体），回退 skimage。
- 输出：2D `(H, W)` uint8 数组，值域 [0, 255]。

**多通道（`num_classes > 1`）**：
```python
img_tmp = skimage.io.imread(img_loc).astype(np.uint8)
# 启发式判断通道轴：若 shape[0] > 20 则假定为 (C, H, W)
if img_tmp.shape[0] > 20:
    img_full = np.moveaxis(img_tmp, 0, -1)
else:
    img_full = img_tmp
# 选择骨架化波段
if type(skeleton_band) == str:   # "all"
    img = np.sum(img_full, axis=0).astype(np.int8)
else:
    img = img_full[skeleton_band, :, :]
```
- skimage 可能以 `(C, H, W)` 或 `(H, W, C)` 读入。规则：若 `shape[0] > 20` 则重排为 `(H, W, C)`。
- `skeleton_band` 参数决定使用的波段。默认值 7（对应 SpaceNet 5 掩码的速度通道）；取值 `"all"` 则对所有波段求和。
- 结果归约为单通道 `(H, W)`。

#### 步骤 2：超大图像裁剪（第 626–630 行）

```python
img = img[:max_out_size[0], :max_out_size[1]]
```

安全阀，防止内存溢出。默认 `max_out_size = (200000, 200000)`。常规 SpaceNet 瓦片（< 2000×2000）不触发此分支。

#### 步骤 3：边界复制（第 634–635 行）

```python
if fix_borders:
    img = cv2.copyMakeBorder(img, replicate, replicate, replicate, replicate,
                             cv2.BORDER_REPLICATE)
```

使用 `cv2.BORDER_REPLICATE` 向外复制 `replicate=5` 像素的边缘值。

**目的**：消除骨架化在图像边缘处的边界伪影。当道路恰好终止于瓦片边缘时，骨架化需要足够的邻域上下文来正确判定连通性。不加这个步骤，骨架在边缘会出现异常弯曲或断裂。

结果：图像从 `(H, W)` 变为 `(H+10, W+10)`。

#### 步骤 4：preprocess() — 二值化与形态学精炼（第 332–376 行）

**位置**：第 332–376 行

分为两个路径，以 image size 是否超 1e10 像素为界：

##### 路径 A — skimage 路径（默认，适用绝大多数情况）

```python
img = (img > (img_mult * thresh)).astype(np.bool)
remove_small_objects(img, hole_size, in_place=True)
remove_small_holes(img, hole_size, in_place=True)
```

1. **阈值二值化**：`img > 255 × 0.3` → 所有大于 76 的像素标记为 `True`（道路）。`img_mult=255` 将输入归一到 [0,255] 范围。
2. **`remove_small_objects(…, hole_size=300)`**：删除面积 < 300 像素的白色连通域（假阳性噪点斑块）。底层调用 `ndimage.label` → 统计各 label 面积 → 删除小面积。
3. **`remove_small_holes(…, hole_size=300)`**：填充面积 < 300 像素的黑色空洞（道路内部假阴性裂缝）。

300 像素在 GSD=0.3m/像素下对应约 90m²，是一个经验阈值：小于它的空隙不值得连接，小于它的噪点不值得保留。

##### 路径 B — OpenCV 路径（超大图像，避免 ndimage.label OOM）

```python
blur = cv2.medianBlur((img * img_mult).astype(np.uint8), kernel_blur)
glob_thresh_arr = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)[1]
glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
mask_thresh = glob_thresh_arr_smooth

closing_t = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)
img = opening_t.astype(np.bool)
```

1. **中值模糊**：`kernel_blur=7`，去除椒盐噪点。
2. **全局阈值**：`cv2.threshold(blur, 0.3, 1, THRESH_BINARY)`，与 skimage 路径等价。
3. **二次模糊**：平滑二值图边缘。
4. **MORPH_CLOSE**（先膨胀后腐蚀，7×7 全 1 核）：填充裂缝。
5. **MORPH_OPEN**（先腐蚀后膨胀，7×7 全 1 核）：去除小噪点。
6. **转 bool**。

##### 被注释掉的备选路径

```python
# img, _ = dl_post_process_pred(img)
```

`dl_post_process_pred()`（第 122–239 行）实现了一个更复杂的处理流程：
- 自适应阈值（`cv2.adaptiveThreshold`，Gaussian 模式）
- 轮廓检测 + 最小面积过滤（`min_area=2000`）
- 轮廓近似平滑
- `medial_axis` 骨架化

该函数是早期实验代码，主路径不使用。

#### 步骤 5：空掩码检查（第 649 行）

```python
if not np.any(img):
    return None, None
```

预处理后全图无前景 → 提前返回 `(None, None)`。调用方据此输出 `LINESTRING EMPTY`。

#### 步骤 6：骨架化（第 651–668 行）

两个分支，由 `use_medial_axis` 参数控制。

##### 分支 A — skeletonize（默认，`use_medial_axis=False`）

```python
ske = skeletonize(img).astype(np.uint16)
```

调用 `skimage.morphology.skeletonize()`，实现 **Lee et al. 1994** 并行迭代细化算法：

- **输入**：二值图（bool）。
- **算法本质**：每一轮迭代中，识别所有"可删除"的边界像素。判定条件是检查像素的 3×3/4×4 邻域是否匹配特定的八边形模板——这些模板编码了不影响拓扑连通性的边界像素模式。所有匹配模板的像素在这一轮中被并行标记删除。迭代不变式：不破坏连通分量数（Euler 数），不缩短分支端点到图像边缘。
- **终止条件**：没有更多像素可删除。
- **输出**：严格单像素宽的拓扑骨架（极少例外：在 2×2 方块中心可能出现 2 像素厚度，但 Lee 算法通过其子迭代模板抑制了这种情况）。
- **类型**：`astype(np.uint16)`——sknw 后续用 uint16 做缓冲区索引，需要无符号整型。

##### 分支 B — medial_axis（`use_medial_axis=True`）

```python
ske = skimage.morphology.medial_axis(img).astype(np.uint16)
```

采用基于距离变换的算法：

- 计算二值图的欧几里得距离变换（每个前景像素到最近背景的距离）。
- 追踪距离图的脊线（ridge），脊线上每点的距离值局部最大。
- 这些脊线天然位于形状中心。

| 特性 | skeletonize (Lee) | medial_axis |
|---|---|---|
| 算法范式 | 并行迭代细化 | 距离变换 + 脊线追踪 |
| 输出宽度 | 严格 1 像素 | 通常 1 像素，交叉口可能较厚 |
| 噪声敏感度 | 较低 | 较高（距离变换放大边界噪声） |
| 处理速度 | 中等 | 较快（单次距离变换而非多轮迭代） |
| 平滑度 | 有小锯齿 | 较平滑，中心定位更准 |
| 交叉口形态 | 干净 T/X 节点 | 交叉口处产生团块（medial 团） |

**生产配置**：`sn5_baseline.json` 设置 `use_medial_axis: 0`，使用 `skeletonize`（Lee）。medial_axis 在交叉口产生的团块会增加后续 sknw 图中节点合并的复杂度。

#### 步骤 7：fix_borders 后处理（第 670–680 行）

```python
if fix_borders:
    ske = ske[rec:-rec, rec:-rec]
    ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip,
                             cv2.BORDER_CONSTANT, value=0)
    img = img[replicate:-replicate, replicate:-replicate]
```

**三步操作**：

1. **裁剪复制边框**：`ske[7:-7, 7:-7]` 删除步骤 3 添加的 `replicate+clip=7` 像素外边框。`rec = replicate + clip = 5 + 2 = 7`。`clip=2` 的多余裁剪是为了切除复制边框处的伪骨架分支——复制边让骨架化"看到"一条沿图像边缘无限延伸的道路，可能因此产生虚假分支和端点。

2. **添加零边框**：用 `BORDER_CONSTANT, value=0` 填充 2 像素宽的零边框。目的：
   - **端点修剪**：骨架若延伸至裁剪后边缘，现在被零值截断为干净端点。
   - **sknw 兼容**：`sknw.build_sknw` 要求图像四周至少有 1 像素背景，否则邻域遍历会越界。

3. **同步裁剪精炼掩码**：`img[5:-5, 5:-5]` 去掉复制边框，保持尺寸与输入图像一致（仅用于可选的调试可视化）。

**为什么分成"裁剪 + 再填充"两步**？只裁剪 `replicate` 不足以去除复制边框内部的伪迹；全部裁剪掉 `replicate+clip` 再重新加纯零边框，保证边缘只有干净背景。

尺寸变化示踪：

| 阶段 | 尺寸 |
|---|---|
| 输入 | `H × W` |
| 复制边框后 | `(H+10) × (W+10)` |
| 骨架化后 | `(H+10) × (W+10)` |
| 裁剪 `rec` 后 | `(H-4) × (W-4)` |
| 再填充 clip 后 | `(H-4+4) × (W-4+4)` = `H × W` |
| 精炼掩码（裁剪 `replicate`） | `(H) × (W)` |

两个返回值尺寸并**不一致**：骨架尺寸 `H × W`，精炼掩码尺寸 `(H-10) × (W-10)`。调用方只将骨架传给 sknw，精炼掩码仅用于可选的 `visualize()` 调试。

#### 步骤 8：返回（第 688 行）

```python
return img, ske
```

- `img`：精炼二值掩码 (bool)，尺寸 `(H-2*replicate, W-2*replicate)`。
- `ske`：uint16 骨架，尺寸 `(H, W)`（比 `img` 在每边各多 `clip` 像素的零边框）。

---

### 2.2 sknw.build_sknw() — 栅格骨架到 NetworkX 图

**位置**：`cresi/utils/sknw.py`（第 113–120 行）  
**输入**：uint16 骨架数组  
**输出**：NetworkX `MultiGraph`

**算法细节**：

#### 步骤 1：buffer() — 添加零边框（sknw 第 108–111 行）

```python
def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape)+2), dtype=np.uint16)
    buf[tuple([slice(1,-1)]*buf.ndim)] = ske
    return buf
```

在骨架四周填充 1 像素零边框——确保邻域遍历不越界。

#### 步骤 2：mark() — 像素分类（sknw 第 17–26 行）

```python
@jit(nopython=True)
def mark(img, nbs):
    img = img.ravel()
    for p in range(len(img)):
        if img[p] == 0:
            continue
        if img[p] > 1:
            img[p] = 2
            continue
        s = 0
        for d in range(9):
            s += img[p + nbs[d]]
        if s < 3:
            img[p] = 2
        elif 3 <= img[p]:
            img[p] = 1
```

遍历所有骨架像素（Numba JIT 编译，接近 C 速度）。每个骨架像素（>0）分类为：
- **`2` = 节点**：端点（邻域骨架像素计数 < 3）或已有 label >1 的点。
- **`1` = 边像素**：邻域骨架像素计数 ≥ 3（即至少有两个骨架邻居——是线段内部点）。
- **`0` = 背景**。

这里的"邻域骨架像素计数"计算包含自身的 8 邻域窗口。若一个骨架像素邻域总和 < 3（即除自身外最多 1 个骨架邻居），它是端点；≥ 3 则至少有 2 个骨架邻居，属于边内部。

#### 步骤 3：fill() — 节点融合（sknw 第 38–55 行）

```python
@jit(nopython=True)
def fill(img, p, num, nbs, acc, buf):
    back = img[p]
    s = 0
    buf[s] = p; s += 1
    img[p] = num
    cur = 0
    while cur < s:
        p = buf[cur]
        for d in range(9):
            q = p + nbs[d]
            if img[q] == back:
                buf[s] = q; s += 1
                img[q] = num
        cur += 1
    return idx2rc(buf[:s], acc)
```

标记为 `2` 的节点像素可能形成一个**团块**（交叉口处多个像素连接在一起）。fill() 从种子像素开始 BFS，将所有连通的 label=2 像素合并为同一个节点 ID，并计算它们的重心坐标（坐标的平均值）。

#### 步骤 4：trace() — 边追踪（sknw 第 57–76 行）

```python
@jit(nopython=True)
def trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0
    ...
    return (c1-10, c2-10, idx2rc(buf[:cur], acc))
```

从节点出发，沿 label=1（边像素）追踪到另一个节点。使用缓冲区记录路径上的所有坐标。返回 `(start_id, end_id, path_array)`——path_array 是两个节点之间的 N×2 骨架像素坐标（不含端点，或根据实现可能包含）。

#### 步骤 5：parse_struc() + build_graph()（sknw 第 78–106 行）

```python
@jit(nopython=True)
def parse_struc(img, pts, nbs, acc):
    # 调用 mark → fill → trace，返回 nodes, edges
    ...

def build_graph(nodes, edges, multi=False):
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(nodes.shape[0]):
        graph.add_node(i, pts=nodes[i], o=nodes[i])  # o = (y, x) 重心
    for s, e, pts in edges:
        graph.add_edge(s, e, pts=pts, weight=len(pts))  # weight = 像素长度
    return graph
```

构建 `networkx.MultiGraph`，关键属性：
- **节点**：`'o'` = `(y, x)` 坐标（图像行优先格式）。
- **边**：`'pts'` = N×2 int32 数组（节点间骨架像素坐标）；`'weight'` = `len(pts)`（像素长度）。

**坐标约定**：sknw 全程使用 `(row, col)` / `(y, x)` 格式与 numpy 数组索引一致。后续的 WKT 序列化负责将其转换为 WKT 标准 `(x, y)`。

#### sknw_int64 备选

```python
if np.max(ske.shape) > 32767:
    G = sknw_int64.build_sknw(ske, multi=True)
else:
    G = sknw.build_sknw(ske, multi=True)
```

当骨架尺寸超过 int16 可表示范围（32,767）时，使用 `utils/sknw_int64.py` 版本，其缓冲区索引使用 int64 避免溢出。

---

### 2.3 迭代毛刺修剪 — remove_small_terminal()

**位置**：第 431–479 行

重复执行最多 8 轮：

```python
for itmp in range(8):
    ntmp0 = len(G.nodes())
    remove_small_terminal(G, weight='weight',
                          min_weight_val=min_spur_length_pix,
                          pix_extent=pix_extent)
    ntmp1 = len(G.nodes())
    if ntmp0 == ntmp1:
        break
```

**`remove_small_terminal()` 的内部逻辑**（第 431–479 行）：

1. **查找端点**：`deg = dict(G.degree())`，所有度为 1 的节点。

2. **遍历每条边 `(s, e)`**：
   - **自环处理**：若 `s == e` 且边总像素数 < 3，直接移除该边。
   - **边界保护**：检查边的四个端点坐标 `(sx, sy, ex, ey)` 是否在图像边缘 `edge_buffer=4` 像素内。若是，**保留该边**——图像边缘的端点很可能是道路自然终止而非伪毛刺。此保护防止误删真正的道路端点。
   - **毛刺判断**：若边端点中的任一节点是度为 1 的端点，且该边的 `weight`（像素长度）小于 `min_spur_length_pix`（从 `min_spur_length_m / GSD` 换算得到），则删除该端点节点。

3. **为什么需要多轮**：移除一个毛刺后，其相邻边的端点可能变为新的度为 1 的节点，暴露之前被隐藏的毛刺。多轮迭代确保收敛到稳定拓扑。

4. **收敛判断**：当一轮没有移除任何节点时终止。实际测试中通常 2–3 轮达到稳定。

**`min_spur_length_pix` 的确定**（在 `main()` 中）：

```python
min_spur_length_pix = int(np.rint(config.min_spur_length_m / config.GSD))
```

SpaceNet 5 默认：`10 m / 0.3 m/pixel ≈ 33 像素`。这是物理世界中有意义的最小道路段长度——任何短于此的分支末端被认为是毛刺而非真正的死胡同路。

---

### 2.4 自环移除（第 843 行）

```python
ebunch = nx.selfloop_edges(G)
G.remove_edges_from(list(ebunch))
```

移除 sknw 有时产生的起点=终点的伪边（骨架化过程中像素级环路的残余）。

---

## 第三阶段：G_to_wkt() — 图到 WKT 序列化

**位置**：第 851–945 行  
**输入**：NetworkX `MultiGraph`（来自 .gpickle 文件）  
**输出**：WKT `LINESTRING` 字符串列表

`build_wkt_dir()` 在并行阶段将所有图保存为 pickle 后，在**单线程**中逐一调用 `G_to_wkt()`——因 WKT 字符串操作密集，不适宜多线程。

### 3.1 graph2lines() — 边列表到节点路径（第 379–399 行）

```python
def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines
```

将无序边列表转换为有序节点序列（类似半欧拉路径）。贪心策略：当前边终点匹配下一边的起点时，将其并入同一个路径；否则开始新路径。结果如：

```
edges:  [(0,1), (1,2), (2,3), (4,5), (5,6)]
→ node_lines: [[0,1,2,3], [4,5,6]]
```

### 3.2 迭代节点行，构建坐标字符串（第 897–924 行）

**核心循环**——对每个节点行 `w`：

**Step A**：将节点序列成对分组：

```python
for s, e in pairwise(w):
    vals = flatten([[v] for v in G[s][e].values()])
    for ix, val in enumerate(vals):
        s_coord, e_coord = node[s]['o'], node[e]['o']
        pts = val.get('pts', [])
```

`pairwise` 产生 `(s0,e0), (e0,e1), (e1,e2), ...`。每条边的 pts 属性是两节点间的骨架像素坐标数组（来自 sknw）。

**Step B**：`add_direction_change_nodes()`（第 483–498 行）：

```python
def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)  # RDP 简化
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps
```

关键功能：

1. **RDP 简化**：`cv2.approxPolyDP(pts, epsilon=2, closed=False)` 是 OpenCV 对 Ramer-Douglas-Peucker 算法的实现。将密集的骨架像素序列压缩到仅保留"方向变化"的关键顶点。`epsilon=2` 意味着任何偏离当前线段超过 2 像素的点都会触发一个新顶点。长直路段→两个端点即可；急弯处→保留足够的点来表征曲线形状。

2. **方向修正**：RDP 返回的顶点顺序可能与原始节点方向一致也可能不一致（取决于 RDP 内部的起点选择）。通过比较 RDP 第一个顶点到 `s_coord` 和到 `e_coord` 的距离来判断是否需要翻转序列。

3. **端点对齐**：强制 `ps[0] = s_coord`、`ps[-1] = e_coord`——这是节点重心坐标，比 RDP 简化的边界像素坐标更精确地表示交点位置。

**Step C**：格式化坐标字符串：

```python
line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
```

其中 `c = (y, x)`（sknw 节点坐标格式），`c[0]=y, c[1]=x`。格式字符串 `"{1:.1f} {0:.1f}"` 产生 `x y`——WKT 标准顺序。每条 LINESTRING 形如：

```
LINESTRING (x1 y1, x2 y2, x3 y3, ...)
```

**Step D**：合并多属性边：若一条边有多个平行属性（MultiGraph 的多条平行边），额外路径作为独立 LINESTRING 添加，但先去重检查是否已被已有 WKT 覆盖。

### 3.3 remove_duplicate_segments() — 段去重（第 286–308 行）

**两步操作**：

1. **`remove_sequential_duplicates()`（第 275–282 行）**：删除坐标序列中连续重复的点。遍历序列，仅当 `elem != res[-1]` 时追加——简单但有效。

2. **`remove_duplicate_segments()`（第 286–308 行）**：更复杂的去重——检测重复的边段。如果 `(a, b)` 和 `(b, a)` 同时出现在路径中（图数据结构中的双向记录），在此处分割序列。这处理了 sknw 图有时将同一段道路注册为双向（`(0, 1)` 和 `(1, 0)`）的情况，避免 WKT 中出现来回折返的无效路径。

### 3.4 add_small_segments() — 连接短间隙（第 502–576 行）

重新连接预测掩码中由于弱信号而断裂的短道路间隙。

**算法**：

1. **识别端点**：所有度为 1 的节点及其坐标。
2. **计算距离矩阵**：`scipy.spatial.distance.pdist` 计算所有端点的成对距离。
3. **短距离配对**（`dist < dist1=24` 像素≈7.2 米）：
   - 所有在此范围内的端点对。
   - 约束：不能已有现成边连接。
4. **中距离配对**（`dist1=24 < dist < dist2=80` 像素≈24 米）：
   - 额外几何验证：端点的方向向量与另一个端点的点距离小于 `dist1`。
   - 角度范围约束：两段的方向角差必须在 `(-30°, 30°)`（平滑连接）或 `(150°, 210°)`（几乎平行的 U 形回弯）范围内。
5. **分配**：按距离升序排序的 `OrderedDict`。每对端点只能被分配一次（`added` 集合）。
6. **发出 WKT**：每对生成一个单段 `LINESTRING(x1 y1, x2 y2)`。

为什么要有角度约束？两个短线段末端朝向对方时，通常是断裂的同一条道路；朝向相反方向时，可能是两条不同的平行道路，不应连接。角度约束防止误连平行相邻道路。

---

## 第四阶段：输出格式

最终 DataFrame 保存为 CSV，包含两列：

| 列名 | 内容示例 |
|---|---|
| `ImageId` | `SN5_roads_train_AOI_8_Mumbai_PS-RGB_chip0` |
| `WKT_Pix` | `LINESTRING (10.0 20.0, 10.5 25.3, 15.0 30.0)` |

WKT 坐标值的像素坐标系原点在图像左上角，x 向右、y 向下（WKT 标准坐标系，与 numpy 数组索引的映射由序列化代码处理）。

---

## 与下游模块的连接

`05_wkt_to_G.py` 读取此 CSV，将像素 WKT 转换为地理参考 NetworkX 图：

| 04_skeletonize.py | 05_wkt_to_G.py |
|---|---|
| Mask → skeleton → sknw → NetworkX（像素坐标） | CSV(WKT_Pix) → wkt_list_to_nodes_edges() |
| G_to_wkt → CSV(WKT_Pix) | nodes_edges_to_G() → clean_sub_graphs() |
| | get_node_geo_coords() — GDAL 像素→地理坐标 |
| | get_edge_geo_coords() — 边几何地理参考 |
| | ox.project_graph() — 投影到 UTM |
| | ox.simplify_graph() — 拓扑简化 |
| | rdp.rdp() — 最终直线简化 |
| | → .gpickle 输出 |

**关于 OSM 与 OSMnx**：
- 04_skeletonize.py 完全不涉及 OSM 数据。它是纯图像处理 + 图论操作。
- 05_wkt_to_G.py 中的 `osmnx` 包仅用于投影（`project_graph`）和拓扑简化（`simplify_graph`），**不用于下载任何 OSM 地图数据**。
- `utils/osmnx_funcs.py` 是从 OSMnx v0.9 移植的纯算法代码（投影、简化、绘图辅助），也不执行任何数据下载。

---

## 参数速查表

| 参数 | 默认值 | 配置文件键 | 作用域 | 作用 |
|---|---|---|---|---|
| `thresh` | 0.3 | `skeleton_thresh` | preprocess | 二值化阈值（乘以 255 后与像素值比较） |
| `img_mult` | 255 | — | preprocess | 像素值乘数，默认假设输入 [0,1] → 255 |
| `hole_size` | 300 | — | preprocess | `remove_small_objects`/`remove_small_holes` 的最大像素面积 |
| `cv2_kernel_close` | 7 | — | preprocess (cv2 分支) | MORPH_CLOSE 核大小 |
| `cv2_kernel_open` | 7 | — | preprocess (cv2 分支) | MORPH_OPEN 核大小 |
| `replicate` | 5 | — | make_skeleton | 复制边框宽度（像素） |
| `clip` | 2 | — | make_skeleton | 骨架化后零边框宽度（像素） |
| `use_medial_axis` | False | `use_medial_axis` | skeletonize | 选择骨架化算法 |
| `skeleton_band` | `"all"`(str) 或 7(int) | `skeleton_band` | 多通道读取 | 多通道掩码的波段选择 |
| `max_out_size` | `(200000, 200000)` | — | 输入裁剪 | 超大图像的安全裁剪上限 |
| `min_weight_val` | 从 `min_spur_length_m` 换算 | `min_spur_length_m` | 毛刺修剪 | 毛刺最大像素长度 |
| `edge_buffer` | 4 | — | 毛刺修剪 | 图像边缘保护的像素范围 |
| `dist1` | 24 | — | add_small_segments | 短距离配对阈值（像素） |
| `dist2` | 80 | — | add_small_segments | 中距离配对阈值（像素） |
| `angle1` | 30 | — | add_small_segments | 平滑连接角度差上限（度） |
| `angle2` | 150 | — | add_small_segments | U 形回弯角度差下限（度） |
| `approx` | 2 | — | direction_change_nodes | cv2.approxPolyDP 的 epsilon 值（像素） |
| `min_subgraph_length_pix` | 20 | `min_subgraph_length_pix` | build_wkt_dir | G_to_wkt 后清理短子图的最小长度。此参数在 04 中传入 `build_wkt_dir()` 但不被 04 使用——它在 05_wkt_to_G.py 中被 `clean_sub_graphs()` 消费。 |
| `n_threads` | 12 | — | build_wkt_dir | 并行骨架化的进程数 |

---

## 算法设计选择总结

1. **两阶段并行/串行混合**：多线程执行计算密集型的 `img_to_ske_G()`（骨架化 + 建图），单线程执行 I/O 密集型的 `G_to_wkt()`（pickle 加载 + WKT 序列化）。避免了字符串格式化在 GIL 下的争用。

2. **sknw（Numba JIT）取代 ndimage.label**：Numba 编译的 `mark`/`fill`/`trace`/`parse_struc` 函数将核心的像素分类 → 节点合并 → 边追踪管道加速到接近 C 的性能，比纯 Python skimage 的连通域标记方法快一到两个数量级。

3. **cv2.approxPolyDP 作为矢量化简器**：在骨架像素层面进行 RDP 简化，比直接在 WKT 层面简化更可控。epsilon=2 像素的阈值在实践中保留了几何保真度（方向变化处保留顶点）和文件大小（长直路段仅两个端点）之间的良好平衡。

4. **迭代毛刺修剪**：不依赖单次完美修剪，而是用收敛格式的多轮迭代。这使算法对初始 sknw 图的质量鲁棒——即使第一次修剪遗漏了某些毛刺，后续轮次会在新暴露出的端点处捕获它们。

5. **多通道骨架波段选择**：对于 SpaceNet 5 的 8 通道掩码（7 个速度 bin + 1 个路网通道），`skeleton_band` 参数允许指定哪个通道用于骨架化。默认使用第 7 通道（速度 bin 6, 最高速度级别），因为它有最完整的道路覆盖。

6. **边界保护机制**：三重保障防止图像边缘道路被错误修剪：(a) fix_borders 的复制边框 (b) `remove_small_terminal` 中的 `edge_buffer` 保护 (c) `clip` 清零边框。这确保瓦片化评估中跨瓦片道路端点的正确拼接。
