## 操作说明

- 鼠标选取房子上一点并拖动可以施加拉力。
- 按 W 和 S 前后移动镜头，鼠标拖动房子外的点会绕坐标零点旋转镜头。按下 Space 反转重力。
- 所有房子共用的参数可在场景中 `Global Var` 对象中调节，主要包括
  - 地面摩擦系数以及是否启用静摩擦力
  - 每帧计算数（超弹性模型计算数太大时容易不流畅）
  - 重力（每帧计算数太低时加速碰撞流程）
- 不同的房子有一些独立的参数，主要是
  - 弹性模型（拖动 `model_idx` 轴进行选择，模型名称显示在 `model_name` 中）
  - 是否使用超弹性模型（`user_hyper_model` 复选框，只有 `model_idx` 为 0 即使用 StVK 模型时可取消勾选）

## 基础要求
  
### 1. StVK 的 FEM 显示积分仿真 & 3. 顶点速度拉普拉斯平滑 & 5. 基础的仿真参数调节优化 & Bonus 1
  
`class FVM` 中的 `get_first_pk_stress1()` 函数实现了 PPT 第七讲 P25 使用 Green Strain 的仿真流程。

`_Update()` 函数的最后部分实现了拉普拉斯平滑，即每个点的速度均匀贡献到所属四面体的 4 个点的速度中，因此用 `V_sum[]` 和 `ref_count[]` 数组统计了每个点的速度贡献。

使用拖动条可以改变弹性模型参数，而切换弹性模型也可使用拖动条，如果勾选 `auto_set_params_when_model_changed` 则会在切换模型时换上该模型的一套默认参数，防止爆炸💥。

### 2. 弹性体与地面间的碰撞处理

`particle_collision()` 和 `superficial_collision()` 函数实现了两种与地面碰撞的函数，后者带有静摩擦力。演示时默认使用的是带静摩擦力的。

Lab1 中的碰撞写法可能会导致粒子在碰撞表面来回振荡，使静摩擦不够明显。所以在 `superficial_collision()` 中，粒子仅在与碰撞面距离小于 $0$ 时收到法向方向的力，在距离小于 $0.01$ 时就会受到摩擦力。这样静摩擦就比较明显了。

### 4. 面向各向同性材料的特化算法 & Bonus 3. 更多超弹性模型

`get_first_pk_stress_hyper()` 函数根据用户选择的超弹性模型调用不同算法计算弹性张量，这些计算函数的参数都是形变梯度矩阵的特征值 $\lambda_1, \lambda_2, \lambda_3$:
- `get_stress_tensor_stvk()`: StVK 模型
- `get_stress_tensor_neo_hookean()`: Neo-Hookean 模型，能量公式参考 https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
- `get_stress_tensor_mooney_rivlin_peridyno()`: Rivlin-Peridyno 模型，参考 https://github.com/peridyno/peridyno/blob/7a32a01e33f13d299b77ffd9b6112e2bdff32c46/src/Dynamics/Cuda/Peridynamics/EnergyDensityFunction.h#L478
- `get_stress_tensor_mooney_rivlin_wiki()`: 维基百科给出的 Rilvin-Peridyno 能量公式形式与上面的不同，参见 https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid
- `get_stress_tensor_fung_peridyno()` Fung 模型，参见 https://github.com/peridyno/peridyno/blob/7a32a01e33f13d299b77ffd9b6112e2bdff32c46/src/Dynamics/Cuda/Peridynamics/EnergyDensityFunction.h#L557

能量 $W$ 公式正确性其中一个检验方法是，检验形变梯度 $F$ 为单位矩阵时，$W$ 和求得的弹性张量是否为 $0$，不是 $0$ 的话基本直接就会爆炸哦。这样看来 PPT 给的公式似乎有点问题，因此参考了维基百科和仓库 https://github.com/peridyno/peridyno 中不同模型对能量的定义，并对特征值求导得到弹性张量。

### Bonus 2. 交互式拖拽

鼠标点击可以选择物体的一个顶点，施加拉力，拉力与顶点到鼠标的距离成正比。可以同时选择两个房子（~bug~ feature）。如果鼠标发出的射线与所有顶点距离都较远，则不会选择物体，而会拖动屏幕，W, S 键则可以前后移动镜头。

## Bugs

- 大多数模型在拖拽拉力很大时有可能会爆炸。
- 单次更新时间间隔默认为 $\dfrac{1}{600}s$，调节到 2 - 3 倍就较容易爆炸。

## 思考

- 物体间碰撞

`FvmCollider.cs` 是尚未完成的物体间碰撞方法，利用第 8 讲关于碰撞的知识，大致思路是收集所有 `FVM` 类的三角元，为了优化碰撞检测效率，对整个空间 $x$, $y$, $z$ 轴分别对半分，按照八叉树形式划分 (`collide_split()` 函数) ，将三角元氛围与 3 个 **划分轴** 相交的 `on_line_triangles` 和在各个八分之一空间内部的 `off_line_triangles` 两个列表，两个列表进行碰撞检测后再递归检测各个子空间内部的碰撞，三角元之间的碰撞使用牛顿迭代法解三次方程。难点在于两个三角元之间的碰撞和碰撞相应，有待完成。
