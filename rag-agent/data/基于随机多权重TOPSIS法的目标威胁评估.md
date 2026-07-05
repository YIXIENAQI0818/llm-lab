# 基于随机多权重TOPSIS法的目标威胁评估

**作者：** 刘畅^1,2^，李江^1^，陈阳^1^，郭立红^1^，王烨^1^，于洋^1^

**作者单位：**

1. 中国科学院长春光学精密机械与物理研究所，长春 130033  
2. 中国科学院大学，北京 100049

**期刊信息：** 《兵器装备工程学报》，2024，45(7)：50-55，96  
**DOI：** 10.11809/bqzbgcxb2024.07.007

<!-- PDF_PAGE: 1/5 -->

## 摘要

在目标威胁评估问题中，为了解决目标各项特征权重选取可能存在主观决策误差的问题，提出了一种基于随机多权重 TOPSIS 法的空中目标威胁评估方法。基于层次分析法确定了各项特征之间的权重，对难以定量衡量的目标特性进行了特征提取，并基于随机多权重 TOPSIS 法对空中目标的威胁度进行了评估。仿真实例表明：随机多权重 TOPSIS 法威胁评估与 AHP、TOPSIS 方法排序一致，但给出了各目标威胁度的不确定范围，实验中不确定范围值最低为 0.08%，最高为 3.78%。战场指挥人员可以通过本文提出的威胁度不确定性范围得到更多参考信息。

## 关键词

目标威胁度评估；目标特征提取；TOPSIS 法；随机多权重；层次分析法

## 0 引言

现代战争中，目标威胁度评估是空中目标预警探测工作中的重要环节。目标威胁度评估属于典型的多属性决策问题，我方需要通过威胁估计对多源信息进行信息处理，从而从繁杂的交战信息中提取有用的目标信息，进而指导武器分配。合理的威胁估计方法，可以使武器资源利用更加合理，提高作战效能。

已有的目标威胁度评估方法主要基于两种思路，即主观法和客观法。主观法通过有经验的指挥人员，利用其专业经验和主观偏好对描述敌方空中目标的属性重要程度进行打分，再利用打分结果计算各属性权重，进而在实际战场条件下对各目标的各项指标进行综合打分，得到目标威胁度排序。由于存在专家主观参与，不同专家可能对同一属性有不同看法，因此这类方法得到的目标威胁度排序有时难以令人信服，限制了其使用范围。根据专家参与评价的程度和具体方式，主观法包括层次分析法（analytic hierarchy process，AHP）、模糊综合评价法等。已有文献多将这两种方法与其他方法结合，例如熵权法、直觉模糊集、多准则优化妥协决策法、启发式算法和优化模型等，以提高评价权重的可靠性。

基于客观法进行目标威胁度评估的研究较多，其理论依据包括信息熵、贝叶斯网络模型理论、D-S 证据理论、支持向量机以及神经网络等。这类方法理论相对严谨，但也存在两方面不足：一方面，算法所需的大量历史数据可能难以获得；另一方面，将实际作战背景融入数学模型可能存在困难。

部分研究采用组合权重与 TOPSIS 法完成目标威胁评估，也有研究将动态贝叶斯网络、灰色关联分析等与 TOPSIS 结合。这些方法通常需要使用者提供明确的权重分配表；若权重分配不准确，则可能得到错误结果。

针对主观法评价权重可能不准确的问题，本文引入随机多权重 TOPSIS 法。首先基于 AHP 法得到目标威胁度各项评价准则权重，在此基础上引入随机量，以模拟已有权重的不确定性，从而更加全面地评估目标威胁度排序。仿真结果表明，该方法不仅能够提供目标威胁度打分，还能给出打分的不确定带，为战场指挥人员提供更多参考信息。

## 1 目标威胁评估指标体系

目标威胁度评估是一个典型的多属性决策问题。影响目标威胁度的因素众多，且因素之间难以直接定量比较。AHP 法是一种经典决策算法，能够系统化整理专家经验，将复杂问题进行定性和定量相结合的处理，适合多目标、多准则决策，特别适用于目标威胁度评估问题。

AHP 将一个决策问题包含的因素自上而下分解为目标层、准则层和方案层。目标层为解决问题的目的，通常只有一个；准则层为达到目标而抽象出的各种因素；方案层为解决问题的具体对象，在目标威胁度评估问题中对应各个具体目标。目标威胁等级层次如图 1 所示。

**图 1 目标威胁等级层次**  
*Fig. 1 Hierarchy of target threat levels*

> 图像语义增强：
> - 图像类型：AHP 层次结构图。
> - 主要结构：顶层为“目标威胁度”，中间准则层包含距离、角度、高度、速度、机动能力、电子干扰能力和作战状态等准则，底层方案层为不同待评估目标。
> - 模块关系：准则层各指标共同指向目标威胁度，并用于比较方案层中的多个目标。
> - 可用于 RAG 检索的关键词：AHP，目标威胁度，指标体系，准则层，方案层，层次结构。

目标威胁度评估在选择评价准则时，既需要考虑评估准确度，也需要考虑某一项准则具体数值的获取难易程度。准则包括两类：一类是能够用数值描述的目标状态，即距离、角度、高度、速度；另一类是无法直接用数值描述的目标状态，即目标机动能力、目标电子干扰能力和目标作战状态。AHP 方法将各种因素放在统一框架下考虑。各准则编号及说明见表 1。

**表 1 准则层说明**  
*Table 1 Criterion layer description*

| 准则标号 | 准则 | 说明 |
|---|---|---|
| C1 | 距离 | 目标与我方武器的相对距离，单位 km。 |
| C2 | 角度 | 目标相对我方被保护地点的航向角，单位 °。 |
| C3 | 高度 | 目标相对地面的高度。 |
| C4 | 速度 | 目标飞行速度。 |
| C5 | 目标机动能力 | 按“很强、强、一般、较弱、弱”等等级描述。 |
| C6 | 电子干扰能力 | 按“很强、强、一般、较弱、弱”等等级描述。 |
| C7 | 作战状态 | 根据目标当前任务状态进行描述，如攻击、巡航、返航等。 |

> 表格语义说明：
> - 表格类型：威胁评估准则定义表。
> - 比较对象：7 项目标威胁评估准则。
> - 主要内容：C1-C4 为数值型准则，C5-C7 为非数值型准则。
> - 可用于 RAG 检索的关键词：目标威胁指标，距离，角度，高度，速度，机动能力，电子干扰，作战状态。

在分析准则层中各项因素对最终决策的影响程度时，如果将各项准则同时比较，可能难以得到令人信服的结果。Saaty 提出的 AHP 方法不把所有因素同时比较，而是进行两两比较，并采用相对尺度，以减少不同性质造成的不准确。AHP 使用判断矩阵描述不同因素之间的相对重要程度。设判断矩阵为 $A=[a_{ij}]_{n\times n}$，其中元素 $a_{ij}$ 表示准则 $C_i$ 相对于准则 $C_j$ 的重要程度。判断矩阵具有以下性质：

1. $a_{ii}=1$，对角线元素为 1，表示自身比较重要性相同；
2. $a_{ij}=1/a_{ji}$，成对比较的两个因素重要性互为倒数；
3. $a_{ij}>0$。

判断矩阵可能出现不一致。AHP 允许一定程度的不一致，但需要对不一致程度进行检验和控制。

<!-- PDF_PAGE: 2/5 -->

一致性指标定义为：

$$
CI=\frac{\lambda_{\max}-n}{n-1}
\tag{1}
$$

其中，$n$ 为矩阵维数，$\lambda_{\max}$ 为判断矩阵的最大特征值。$CI=0$ 时判断矩阵具有完全一致性，$CI$ 越大，不一致性越严重。

一致性比例定义为：

$$
CR=\frac{CI}{RI}
$$

当 $CR<0.1$ 时，认为判断矩阵的不一致程度处于可接受范围。Saaty 随机构造判断矩阵得到随机一致性指标 $RI$，见表 2。

**表 2 随机一致性指标 RI 统计**  
*Table 2 Random consistency index RI statistics*

| 阶数 $n$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $RI$ | 0.00 | 0.00 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 |

本文选取 7 种准则，因此判断矩阵规模为 $7\times 7$。利用 1-9 标度方法确定准则层判断矩阵，见表 3。

**表 3 准则层判断矩阵**  
*Table 3 Criterion level judgment matrix*

| 准则 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1 | 3 | 5 | 7 | 7 | 3 | 3 |
| C2 | 1/3 | 1 | 1 | 3 | 3 | 5 | 5 |
| C3 | 1/5 | 1 | 1 | 3 | 3 | 5 | 5 |
| C4 | 1/7 | 1/3 | 1/3 | 1 | 1 | 3 | 1 |
| C5 | 1/7 | 1/3 | 1/3 | 1 | 1 | 3 | 1 |
| C6 | 1/3 | 1/5 | 1/5 | 1/3 | 1/3 | 1 | 1 |
| C7 | 1/3 | 1/5 | 1/5 | 1 | 1 | 1 | 1 |

> 转换说明：表 3 由扫描图像重建，个别分数项建议结合原 PDF 复核。

经一致性检验，得到一致性指标 $CI=0.1088$，一致性比例 $CR=0.0824$，因此判断矩阵满足一致性要求。

将判断矩阵进行特征值分解，得到最大特征值对应的特征向量，并据此获得各项准则权重，见表 4。

**表 4 AHP 法得到的各项准则权重**  
*Table 4 The corresponding weights of each criterion obtained by the AHP method*

| 准则标号 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 准则 | 距离 | 角度 | 高度 | 速度 | 机动能力 | 电子干扰能力 | 作战状态 |
| 权重 | 0.4004 | 0.1856 | 0.1787 | 0.0797 | 0.0472 | 0.0508 | 0.0577 |

## 2 目标特征处理

### 2.1 数值型因素处理

对于准则 C1-C4，地面探测装备可以直接获得具体数值。为了进行两两比较，在评价数据前需要对数据进行处理。通常将数据分为正向指标、负向指标、单点指标和区间型指标四类。考虑目标威胁度越大、指标值越大的方向，本文将 C1 距离和 C3 高度作为负向指标，C2 角度作为单点指标，C4 速度作为正向指标，并按表 5 所示方式处理。

**表 5 数值型因素处理说明**  
*Table 5 Numerical factor processing instructions*

| 准则 | 指标类型 | 威胁值处理方式 |
|---|---|---|
| C1 距离 | 负向指标 | 按样本最大值与最小值归一化；距离越近，威胁值越大。 |
| C2 角度 | 单点指标 | 按与指定角度点的偏离程度计算威胁值。 |
| C3 高度 | 负向指标 | 按样本最大值与最小值归一化；高度越低，威胁值越大。 |
| C4 速度 | 正向指标 | 按样本最大值与最小值归一化；速度越大，威胁值越大。 |

> 转换说明：表 5 中的细小公式无法从扫描图像中完全可靠识别，已保留指标类型和可见处理含义，请结合原 PDF 复核具体归一化表达式。

### 2.2 非数值型因素处理

对于每个具体敌方飞行目标，地面探测装备可以对数值型因素进行测量；对于非数值型因素，只能根据既往经验数据进行估计。本文针对每项非数值准则整理威胁值计算规则，见表 6。

**表 6 非数值型因素处理说明**  
*Table 6 Non-numerical factor processing instructions*

| 准则标号 | 准则 | 评语集 | 威胁值 |
|---|---|---|---|
| C5 | 目标机动能力 | 很强、强、一般、较弱、弱 | 0.9、0.7、0.5、0.3、0.1 |
| C6 | 目标电子干扰能力 | 很强、强、一般、较弱、弱 | 0.9、0.7、0.5、0.3、0.1 |
| C7 | 目标作战状态 | 攻击、巡航、返航 | 1.0、0.5、0.1 |

## 3 随机多权重 TOPSIS 法

### 3.1 TOPSIS 算法

将上一节处理后的全部目标特征数值组成数据矩阵 $R=[r_{ij}]_{m\times n}$，其中 $m$ 为目标个数。TOPSIS 算法定义以下变量并据此计算评价得分。

1. 定义第 $j$ 个准则的最大值：

$$
r_j^+=\max(r_{1j},r_{2j},\ldots,r_{mj})
\tag{2}
$$

<!-- PDF_PAGE: 3/5 -->

2. 定义第 $j$ 个准则的最小值：

$$
r_j^-=\min(r_{1j},r_{2j},\ldots,r_{mj})
\tag{3}
$$

3. 定义第 $i$ 个目标与最大值向量的距离：

$$
d_i^+=\sqrt{\sum_{j=1}^{n}(r_{ij}-r_j^+)^2}
\tag{4}
$$

4. 定义第 $i$ 个目标与最小值向量的距离：

$$
d_i^-=\sqrt{\sum_{j=1}^{n}(r_{ij}-r_j^-)^2}
\tag{5}
$$

5. 第 $i$ 个目标的评价得分为：

$$
S_i=\frac{d_i^-}{d_i^++d_i^-}
\tag{6}
$$

其中 $0\leq S_i\leq 1$。当 $S_i$ 越大时，目标指标越接近最大值、越远离最小值，反之亦然。

> 公式说明：
> - 公式类型：TOPSIS 理想解、负理想解距离与贴近度计算。
> - 主要变量：$r_{ij}$ 为目标 $i$ 在准则 $j$ 下的处理值；$d_i^+$ 和 $d_i^-$ 分别为到正理想解和负理想解的欧氏距离；$S_i$ 为综合评价得分。
> - 公式作用：依据目标到正、负理想解的相对距离确定威胁度排序。

### 3.2 随机多权重处理

随机多权重 TOPSIS 法是 TOPSIS 法的一种拓展。考虑权重可能不准确，该方法在 TOPSIS 法基础上加入随机性，以模拟决策中的不确定性或随机性，从而更加全面地评估决策方案。特别是通过层次分析法获得的权重，本身可能存在人为因素造成的不合理性。

随机多权重 TOPSIS 法按以下步骤进行：

1. 利用层次分析法确定各项准则基础权重；
2. 对各项准则基础权重加入随机扰动，生成随机权重，以模拟人为打分的不确定性；
3. 将各目标指标乘以随机权重，并按随机权重分组；
4. 利用 TOPSIS 法对各组目标进行打分；
5. 统计各目标打分的均值和方差，得到各目标的综合评价。

## 4 仿真实例

假设战场同时出现 6 个目标，探测装备获得目标信息如表 7 所示，经过处理得到目标数据矩阵如表 8 所示。

**表 7 目标信息**  
*Table 7 Target information*

| 目标编号 | 距离/km | 角度/(°) | 高度/km | 速度/(m·s⁻¹) | 机动能力 | 电子干扰能力 | 作战状态 |
|---|---:|---:|---:|---:|---|---|---|
| T1 | 20 | 13 | 2.5 | 350 | 较弱 | 强 | 导引/攻击（原图细字需复核） |
| T2 | 150 | 20 | 1.8 | 280 | 很强 | 较弱 | 巡航 |
| T3 | 70 | 80 | 0.5 | 150 | 弱 | 强 | 返航 |
| T4 | 310 | -50 | 3.5 | 300 | 强 | 弱 | 攻击（原图细字需复核） |
| T5 | 160 | 30 | 6.0 | 350 | 较弱 | 强 | 巡航 |
| T6 | 40 | -20 | 1.1 | 260 | 强 | 一般 | 返航 |

> 转换说明：表 7 的部分中文状态词字号较小，已根据表 8 数值映射和可见文字重建，T1、T4 的作战状态建议对照原 PDF 复核。

**表 8 目标数据矩阵**  
*Table 8 Target data matrix*

| 目标编号 | 距离 | 角度 | 高度 | 速度 | 机动能力 | 电子干扰能力 | 作战状态 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 1.00 | 0.84 | 0.64 | 1.00 | 0.30 | 0.70 | 1.00 |
| T2 | 0.55 | 0.75 | 0.76 | 0.65 | 0.90 | 0.30 | 0.50 |
| T3 | 0.83 | 0.00 | 1.00 | 0.00 | 0.10 | 0.70 | 0.10 |
| T4 | 0.00 | 0.38 | 0.45 | 0.75 | 0.70 | 0.10 | 0.50 |
| T5 | 0.52 | 0.63 | 0.00 | 1.00 | 0.30 | 0.70 | 0.50 |
| T6 | 0.93 | 0.75 | 0.89 | 0.55 | 0.70 | 0.50 | 0.10 |

利用 AHP 法和 TOPSIS 法可以直接得到各目标威胁度打分。使用随机多权重 TOPSIS 法时，随机权重可服从均匀分布、正态分布或 Dirichlet 分布等。本文采用均匀分布生成随机权重值，各项随机权重与 AHP 法给出的基准权重偏差不超过 $\pm20\%$。随机生成权重后，将修正后的权重之和归一化为 1，以确保权重合法。

图 2 给出了 AHP 法、TOPSIS 法和随机多权重 TOPSIS 法的目标得分，以及由随机多权重 TOPSIS 统计标准差形成的不确定带。为便于比较，各目标得分均转化为百分制。

**图 2 各目标打分统计图**  
*Fig. 2 Statistical figure of scoring for each goal*

> 图像语义增强：
> - 图像类型：多方法折线对比图，带不确定区间。
> - 横轴：目标编号 T1-T6。
> - 纵轴：目标威胁评分，采用 0-100 分制。
> - 图例：AHP、TOPSIS、随机多权重 TOPSIS，以及随机多权重结果的不确定带。
> - 可见趋势：T1 得分最高，T6 次之，T4 最低；三种方法得到的整体排序一致。随机多权重方法在各目标处给出上下限，其中不同目标的不确定带宽度不同。
> - 与正文结论的关系：用于说明随机多权重 TOPSIS 在保持排序一致的同时，能够提供威胁评分的不确定范围。
> - 可用于 RAG 检索的关键词：目标威胁排序，AHP，TOPSIS，随机权重，不确定带，百分制评分。

为对比各种方法，表 9 给出了各方法的具体评分。

**表 9 各目标打分统计**  
*Table 9 Score statistics for each goal*

| 目标编号 | AHP | TOPSIS | 随机多权重 TOPSIS 均值 | 下限 | 上限 |
|---|---:|---:|---:|---:|---:|
| T1 | 100.00 | 100.00 | 99.99 | 99.92 | 100.07 |
| T2 | 74.68 | 80.50 | 80.64 | 79.25 | 82.04 |
| T3 | 62.80 | 65.99 | 66.26 | 64.69 | 67.83 |
| T4 | 33.67 | 43.02 | 43.07 | 41.44 | 44.70 |
| T5 | 56.98 | 61.73 | 61.57 | 60.73 | 62.42 |

<!-- PDF_PAGE: 4/5 -->

| 目标编号 | AHP | TOPSIS | 随机多权重 TOPSIS 均值 | 下限 | 上限 |
|---|---:|---:|---:|---:|---:|
| T6 | 89.79 | 89.24 | 89.44 | 87.85 | 91.03 |

不论采用何种方法，目标威胁度由高到低的排序均为：

$$
T1>T6>T2>T3>T5>T4
$$

随机多权重 TOPSIS 法除给出排序外，还给出每个目标威胁度的不确定范围。各目标的不确定度并不相同，其中不确定性最高的是 T4，不确定度为 3.78%；最低的是 T1，不确定度为 0.08%。指挥员应对不确定度较高的目标加以留意，避免威胁度变化导致排序改变；不确定度较低的目标，其威胁度结果相对更可信。

> 表格语义说明：
> - 表格类型：多方法目标威胁评分对比表。
> - 比较对象：AHP、TOPSIS、随机多权重 TOPSIS。
> - 主要指标：评分均值、随机权重评分下限与上限。
> - 数值关系：三种方法得到相同排序；T1 评分最高，T4 最低；T4 的随机评分区间相对更宽。
> - 可用于 RAG 检索的关键词：威胁评分，排序一致性，不确定度，随机多权重 TOPSIS，目标 T1-T6。

## 5 结论

针对主观法评价权重可能不准确的问题，本文引入随机多权重 TOPSIS 法解决目标威胁度评估问题。该方法不仅能够给出各目标威胁度评分，同时可得到评分的不确定范围。仿真实验中，不确定度最低为 0.08%，最高为 3.78%，代表不同目标威胁评估值的可信任范围。当两个或多个目标得分接近时，可以通过不确定度大小判断各目标威胁度评估的可信程度，并提醒指挥员关注不确定度较大的目标及其威胁变化。

## 参考文献

> 转换说明：本页参考文献字号较小，以下按可见内容转写；少量中文作者、题名及英文拼写无法可靠识别，已明确标注，建议据原 PDF 或数据库元数据复核。

[1] 杨丹，白苗. 基于熵权法和层次分析法相结合的目标威胁度评估仿真研究[C]//中国指挥与控制学会. 第九届中国指挥控制大会论文集. 兵器工业出版社，2021: 4-15. YANG Dan, BAI Miao. Simulation research on target threat assessment based on entropy weight method and analytic hierarchy process[C]//Chinese Institute of Command and Control. Proceedings of the 9th China Conference on Command and Control. The Publishing House of Ordnance Industry, 2021: 4-15.

[2] 张明双，徐克虎，李玲芝. 基于直觉模糊集和 VIKOR 的多目标威胁评估[J]. 兵器装备工程学报，2019，40(6): 62-67. ZHANG Mingshuang, XU Kehu, LI Lingzhi. Multi-target threat assessment based on intuitionistic fuzzy set and VIKOR[J]. Journal of Ordnance Equipment Engineering, 2019, 40(6): 62-67.

[3] KONG D, CHANG T, WANG Q, et al. A threat assessment method of group targets based on interval-valued intuitionistic fuzzy multi-attribute group decision-making[J]. Applied Soft Computing, 2018, 67: 350-369.

[4] 杨军佳，李凯. 基于参数维和时间维的空袭目标威胁二维评估[J]. 兵器装备工程学报，2021，42(5): 239-243. YANG Junjia, LI Kai. Two dimensional evaluation of air attack target threat based on parameter and time dimension[J]. Journal of Ordnance Equipment Engineering, 2021, 42(5): 239-243.

[5] LUO R, HUANG S, ZHAO Y, et al. Threat assessment method of low altitude slow small (LSS) targets based on information entropy and AHP[J]. Entropy, 2021, 23(10): 1292-1306.

[6] 毋嘉纬，周林，金勇，等. 基于主客观相结合的空中目标威胁评估[J]. 指挥信息系统与技术，2022，13(1): 22-29. WU Jiawei, ZHOU Lin, JIN Yong, LI Junwei, LIU Huanyu. Air target threat assessment based on subjective and objective combination[J]. Command Information System and Technology, 2022, 13(1): 22-29.

[7] 杜可可，张悦，赵凯. 结合相似性测度与随机森林的个人信用评估模型[J]. 重庆工商大学学报（自然科学版），2022，39(3): 54-60. DU Keke, ZHANG Yue, ZHAO Kai. Personal credit assessment model based on the combination of similarity measurement and random forest[J]. Journal of Chongqing Technology and Business University (Natural Science Edition), 2022, 39(3): 54-60.

[8] 孙和强，赵国林，唐菲菲，等. 电子对抗无人机作战目标威胁评估[J]. 舰船电子对抗，2020，43(5): 34-37. SUN Heqiang, ZHAO Guolin, TANG Feifei, et al. Threat evaluation of operational target of ECM UAV[J]. Shipboard Electronic Countermeasure, 2020, 43(5): 34-37.

[9] GAO Y, YU M. Target threat assessment method based on cloud model and entropy weight[C]//Proceedings of the 2nd International Conference on Innovation in Artificial Intelligence, 2018: 121-124.

[10] OEKLLP N, THOMS G. Threat assessment using Bayesian networks[C]//Proceedings of the 6th International Conference on Information Fusion, 2003: 1102-1109.（作者拼写需复核）

[11] DI R, GAO X, GUO Z, et al. A threat assessment method for unmanned aerial vehicle based on Bayesian networks under the condition of small data sets[J]. Mathematical Problems in Engineering, 2018: 1-17.

[12] 李江，郭立红. 基于改进 SVM 的目标威胁评估[J]. 光学精密工程，2014，22(5): 1354-1362. LI Jiang, GUO Lihong. Target threat assessment using improved SVM[J]. Optics and Precision Engineering, 2014, 22(5): 1354-1362.

[13] 黄轩，郭立红，李江，等. 基于磷虾群算法优化支持向量机的威胁评估[J]. 光学精密工程，2016，24(6): 1448-1455. HUANG Xuan, GUO Lihong, LI Jiang, et al. Threat assessment of support vector machine optimized by krill herd algorithm[J]. Optics and Precision Engineering, 2016, 24(6): 1448-1455.

[14] 孟祥国，赵金斌，张晓尉，等. 基于 D-S 证据理论的目标威胁评估模型[J]. 航天电子对抗，2022，38(5): 37-40. MENG Xiangguo, ZHAO Jinbin, ZHANG Xiaowei, et al. Threat assessment of air targets based on D-S evidence theory[J]. Aerospace Electronic Warfare, 2022, 38(5): 37-40.

[15] CAO Y, KOU Y X, XU A, et al. Target threat assessment in air combat based on improved glowworm swarm optimization and ELM neural network[J]. International Journal of Aerospace Engineering, 2021: 1-19.

[16] 郝英豪，张永利，李川，等. 基于组合赋权-TOPSIS 法的空中目标威胁评估[J]. 战术导弹技术，2015(5): 103-108. HAO Yinghao, ZHANG Yongli, LEI Chuan, et al. Target threat evaluation based on combination weighting-TOPSIS method[J]. Tactical Missile Technology, 2015(5): 103-108.

[17] 杨璐，刘付显，张涛，等. 基于组合赋权 TOPSIS 法的舰艇编队空中目标威胁评估模型[J]. 电光与控制，2019，26(8): 6-11. YANG Lu, LIU Fuxian, ZHANG Tao, et al. An aerial target threat assessment model based on combined-weighting TOPSIS method for warship formation[J]. Electronics Optics & Control, 2019, 26(8): 6-11.

[18] 刘芳，张勇，宫华，等. 基于 DBN-TOPSIS 法的空中目标融合威胁评估[J]. 兵器装备工程学报，2023，44(1): 136-143. LIU Fang, ZHANG Yong, GONG Hua, et al. Threat assessment of air target fusion based on DBN-TOPSIS[J]. Journal of Ordnance Equipment Engineering, 2023, 44(1): 136-143.

[19] YIN Y, ZHANG R, SU Q. Threat assessment of aerial targets based on improved GRA-TOPSIS method and three-way decisions[J]. Mathematical Biosciences and Engineering, 2023, 20(7): 13250-13266.

## English Title and Abstract

### Target threat assessment based on random multi-weights TOPSIS method

**Authors:** LIU Chang^1,2^, LI Jiang^1^, CHEN Yang^1^, GUO Lihong^1^, WANG Ye^1^, YU Yang^1^

**Affiliations:**  
1. Changchun Institute of Optics, Fine Mechanics and Physics, Chinese Academy of Sciences, Changchun 130033, China  
2. University of Chinese Academy of Sciences, Beijing 100049, China

### Abstract

In the problem of target threat assessment, in order to solve the shortcomings of possible subjective decision-making errors in the selection of weights of various characteristics of the target, an air target threat assessment method based on the random multi-weight TOPSIS method is proposed. This paper determines the weights between various features based on the analytic hierarchy process, extracts features of target characteristics that are difficult to quantitatively measure, and evaluates the threat of air targets based on the random multi-weighted TOPSIS method. The simulation results show that the threat assessment ranking of the random multi-weighted TOPSIS method is consistent with the AHP and TOPSIS methods, but the uncertainty range of each target's threat degree is given. The lowest uncertainty range value in the experiment is 0.08% and the highest is 3.78%. Combat commanders can obtain more reference information through the threat uncertainty range proposed in this article.

**Key words:** target threat assessment; target feature extraction; TOPSIS; random multi-weights; AHP

**收稿日期：** 2023-10-12  
**修回日期：** 2024-01-04  
**录用日期：** 2024-02-25  
**基金项目：** 国家自然科学基金项目（61977059）  
**作者简介：** 刘畅（1993-），男，博士研究生。电子邮箱在扫描页中无法可靠识别。  
**通信作者：** 李江（1982-），男，博士，研究员。电子邮箱在扫描页中无法可靠识别。

<!-- PDF_PAGE: 5/5 -->

**中图分类号：** TP274  
**文献标识码：** A  
**文章编号：** 2096-2304(2024)07-0050-06  
**科学编辑：** 李波 博士（西北工业大学教授）  
**责任编辑：** 唐定国

## 转换自检表

| 检查项 | 结果 |
|---|---|
| PDF 总页数 | 5 |
| Markdown 页码标记 | 5 |
| 页码是否连续 | 是，1/5 至 5/5 |
| 是否存在缺页、跳页或重复页 | 未发现 |
| 是否为单篇论文 | 是 |
| 是否为扫描型 PDF | 是，页面无可提取文本层 |
| 是否保留标题、作者、单位 | 是 |
| 是否保留摘要和关键词 | 是 |
| 是否保留正文全部章节 | 是 |
| 是否保留公式 | 核心公式已重建；表 5 细小公式需复核 |
| 是否保留图题 | 是 |
| 是否完成图像语义增强 | 已完成图 1、图 2 的语义说明 |
| 是否保留表格 | 是，共重建表 1-表 9 |
| 是否结构化表格 | 是 |
| 是否添加表格语义说明 | 关键表格已添加 |
| 是否保留参考文献 | 是，共 19 条；部分著录项需复核 |
| 是否保留英文摘要 | 是 |
| 是否存在非原文摘要 | 否；图表/公式语义说明单独标识 |
| 是否混入外部内容 | 否 |
| 是否适合后续语义切片 | 是，但建议复核标注项后入库 |
| 是否适合 RAG 知识库入库 | 基本适合，扫描细节复核后更稳妥 |
| 最终结论 | 仍需人工复核 |
