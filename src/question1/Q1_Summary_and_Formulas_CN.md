# 综合数学框架与算法规范

## 1. 术语与参数定义

为确保数学严谨性，我们首先定义模型中使用的符号和参数集。

| 符号 | 定义 | 类型/维度 | 描述 |
|:--- |:--- |:--- |:--- |
| $s$ | 赛季索引 | 整数 $\in [1, 34]$ | 赛季编号。 |
| $w$ | 周次索引 | 整数 $\in [1, 12]$ | 赛季内的周次编号。 |
| $N_w$ | 选手数量 | 整数 | 第 $w$ 周活跃选手的数量。 |
| $C_w$ | 选手集合 | 集合 $\{c_1, \dots, c_{N_w}\}$ | 第 $w$ 周参赛的具体选手。 |
| $S_j^{(i)}$ | 原始评委得分 | 实数 $\in [0, 30]$ (或 40) | 选手 $i$ 从评委处获得的原始分数。 |
| $\bar{S}_j^{(i)}$ | 平均评委得分 | 实数 $\in [0, 10]$ | 选手 $i$ 有效评委分数的算术平均值。 |
| $J^{(i)}$ | 归一化评委得分 | 实数 $\in [0, 1]$ | 评委对总分的贡献（占比或排名）。 |
| $F^{(i)}$ | 潜在观众投票 | 实数 $\in [0, 1]$ | **未知**的观众投票占比或排名（目标变量）。 |
| $T^{(i)}$ | 总综合得分 | 实数 | 用于淘汰的最终得分 ($J^{(i)} + F^{(i)}$)。 |
| $e_{obs}$ | 观测到的淘汰 | 索引 $\in C_w$ | 历史数据中实际被淘汰的选手（基准真值）。 |
| $\mathcal{M}$ | 蒙特卡洛迭代次数 | 整数 ($5000$) | 每周的模拟试验次数。 |
| $\alpha$ | Dirichlet 先验 | 向量 $\in \mathbb{R}^{N_w}$ | 观众投票分布的超参数（设为 $\mathbf{1}$ 表示均匀分布）。 |

---

## 2. 数据预处理与特征工程

在模拟之前，我们将原始记分卡转换为适合逆向优化问题的结构化格式。

### 2.1. 变量标准化（评委得分占比）

由于评委数量不同（3 或 4 人）且最高分值变化，我们将绝对分数归一化为**相对竞争占比**。这消除了尺度偏差。

$$
J_{share}^{(i)} = \frac{\bar{S}_j^{(i)}}{\sum_{k=1}^{N_w} \bar{S}_j^{(k)}}
$$

* **理由**：在“总分占比”系统中，重要的不是绝对分数（例如 27/30），而是选手相对于竞争对手获得的“总分蛋糕”的比例。

### 2.2. 基准真值提取

我们解析文本形式的 `Results` 列来定义约束条件：

* $e_{obs} \neq \emptyset$：如果在第 $w$ 周有选手被淘汰，其索引定义了模拟的**下界约束**。
* $W_{withdrew}$：如果 `is_withdrew = 1`，该选手将从建模集合 $C_w$ 中移除，因为他们的退出不是由评分函数 $T^{(i)}$ 决定的。

---

## 3. 投票系统的数学建模

我们将问题表述为：在给定观测结果 $e_{obs}$ 的情况下，寻找 $F^{(i)}$ 的后验分布。我们对两个不同规则的时代进行建模。

### 3.1. 模型 A：基于百分比的系统（第 3-27 赛季）

**机制**：分数是可加的百分比。
**约束**：总分**最低**的人被淘汰。

* **评委输入**：$J^{(i)} = J_{share}^{(i)}$
* **观众未知量**：$\mathbf{F} \sim \text{Dirichlet}(\alpha_1, \dots, \alpha_{N_w})$
* *假设*：在没有事先民意调查数据的情况下，我们假设平坦先验（$\alpha_k = 1, \forall k$），这意味着最初每种观众投票组合的可能性是相等的。
* **综合得分**：$T^{(i)} = J^{(i)} + F^{(i)}$
* **不等式约束**：
  为了使模拟有效，观测到的被淘汰选手 $e_{obs}$ 必须满足：
  
  $$
  T^{(e_{obs})} < T^{(k)} \quad \forall k \in C_w \setminus \{e_{obs}\}
  $$
  
  *平局规则*：如果 $T^{(e_{obs})} = T^{(k)}$，则观众投票 $F$ 较低的一方被淘汰。

### 3.2. 模型 B：秩和系统（第 1-2 赛季，第 28-34 赛季）

**机制**：分数是序数排名（1 = 最好）。
**约束**：排名总和**最高**（表现最差）的人处于危险之中。

* **评委输入**：$J^{(i)} = \text{Rank}(\bar{S}_j^{(i)})$（升序：最高分 $\to$ 排名 1）
* **观众未知量**：$\mathbf{F} \sim \text{Permutation}(\{1, 2, \dots, N_w\})$
* *假设*：观众排名是 $1$ 到 $N_w$ 整数的随机排列。
* **综合得分**：$T^{(i)} = J^{(i)} + F^{(i)}$
* **不等式约束 (S1-S2)**：
  
  $$
  T^{(e_{obs})} > T^{(k)} \quad \forall k \in C_w \setminus \{e_{obs}\}
  $$
* **不等式约束 (S28+ "倒数两名")**：
  评委拯救倒数两名中的一名。因此，被淘汰的人必须在倒数两名中：
  
  $$
  e_{obs} \in \{ i \mid \text{Rank}(T^{(i)}) \ge N_w - 1 \}
  $$

---

## 4. 算法规范：蒙特卡洛拒绝采样

由于解析推导 $\mathbf{F}$ 的可行区域很复杂（尤其是在高维情况下），我们采用随机方法。

### 4.1. 算法伪代码

```python
For each Season s, Week w:
    1. Retrieve Contestants C_w and Judge Scores J.
    2. Identify Target e_obs (Who was actually eliminated?).
    3. Initialize Valid_Samples = [].

    4. Loop m from 1 to M (M=5000):
        a. Generate Hypothesis Vector F_hat:
           IF Model == 'Percentage':
               F_hat ~ Dirichlet(1, 1, ..., 1)
           ELSE IF Model == 'Rank':
               F_hat ~ RandomPermutation(1..N)

        b. Compute Total Score T = J + F_hat

        c. Determine Hypothetical Elimination e_sim:
           e_sim = argmin(T) (or argmax for Rank system)

        d. Validation Check:
           IF e_sim == e_obs:
               Add F_hat to Valid_Samples
           ELSE:
               Discard F_hat (Rejection)

    5. Post-Processing:
       IF len(Valid_Samples) > 0:
           Estimated_Fan_Vote = Mean(Valid_Samples)
           Certainty = Calculate_Certainty(Valid_Samples)
       ELSE:
           Flag as "Unsolvable / Anomaly"
```

### 4.2. 参数论证

* **模拟次数 ($\mathcal{M} = 5000$)**：
  * 我们选择 5000 次迭代作为收敛性和计算成本之间的折衷。
  * 对于典型的有 10 名选手的周次，解空间是高维的。5000 个样本提供的均值标准误差 $< 1\%$，这对于稳健估计来说已经足够。
* **Dirichlet 先验 ($\alpha = \mathbf{1}$)**：
  * 这是“无信息”先验。它断言在观察到淘汰之前，我们认为 $(90\%, 10\%)$ 的观众投票分布与 $(50\%, 50\%)$ 具有同等的可能性。这防止了模型引入偏差。

---

## 5. 后处理指标

### 5.1. 观众投票估计 ($\hat{V}_f$)

选手 $i$ 的最终估计观众投票是有效后验分布的期望值：

$$
\hat{V}_f^{(i)} = \mathbb{E}[F^{(i)} \mid e_{obs}] \approx \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} v^{(i)}
$$

### 5.2. 确定性量化 ($\kappa$)

我们定义一个指标 $\kappa \in [0, 1]$ 来量化约束的“紧密度”。如果一名选手*必须*获得低观众投票才能被淘汰，则有效样本的方差将很低，确定性将很高。

$$
\kappa_i = 1 - \frac{\sigma_i}{\mu_i + \epsilon}
$$

* $\sigma_i$：选手 $i$ 有效观众投票样本的标准差。
* $\mu_i$：平均估计观众投票。
* $\epsilon = 0.1$：正则化项，防止低票选手的除零错误。

**解释**：

* $\kappa \approx 1.0$：确定性。只有当观众投票完全等于该值时，结果才会发生。
* $\kappa < 0.3$：高不确定性。观测到的淘汰与多种观众投票情景一致。
