# Causal Inference Enhanced Simulation Model for Voting Method Comparison
# 组合方法对比模型——因果推断增强型仿真模型

## 1. Nomenclature and Parameter Definitions (术语与参数定义)

To ensure mathematical rigor, we define the set of symbols and parameters used in the Question 2 model.
为确保数学严谨性，我们定义问题二模型中使用的符号和参数集。

| Symbol (符号) | Definition (定义) | Type/Dimension (类型/维度) | Description (描述) |
|:--- |:--- |:--- |:--- |
| $S_{J, i}$ | Judge Score | Real $\in [0, 40]$ | The raw judge score for contestant $i$. <br> 选手 $i$ 的原始评委得分。 |
| $V_{F, i}$ | Fan Vote Estimate | Real $\in [0, 1]$ | The estimated fan vote share (from Q1 Model). <br> 选手 $i$ 的粉丝投票估算值（源自问题一模型）。 |
| $R_{J, i}$ | Judge Rank | Integer $\in [1, N_w]$ | Rank based on judge score (1 = Highest Score). <br> 基于评委分数的排名（1为最高分）。 |
| $R_{V, i}$ | Fan Rank | Integer $\in [1, N_w]$ | Rank based on fan vote (1 = Highest Vote). <br> 基于粉丝投票的排名（1为最高票）。 |
| $C_{Rank, i}$ | Rank Combined Score | Integer | Combined score under Rank Method ($R_{J, i} + R_{V, i}$). <br> 排名法下的组合得分。 |
| $C_{Pct, i}$ | Pct Combined Score | Real $\in [0, 2]$ | Combined score under Percentage Method. <br> 百分比法下的组合得分。 |
| $P(Elim|i)$ | Elimination Prob | Real $\in [0, 1]$ | Probability of being eliminated under Judges' Save. <br> 在法官拯救机制下被淘汰的概率。 |
| $Bias_{Method}$ | Fan Bias Score | Real $\in [0, 1]$ | GNN-derived contribution of fan votes to the outcome. <br> 基于GNN计算的粉丝投票对结果的贡献度。 |

---

## 2. Parallel World Simulation Logic (平行世界仿真逻辑)

We construct "Parallel Worlds" to simulate the fate of each contestant under three different rule sets, keeping their performance data ($S_{J, i}, V_{F, i}$) constant.
我们构建“平行世界”，在保持表现数据（$S_{J, i}, V_{F, i}$）不变的情况下，模拟每位选手在三种不同规则下的命运。

### 2.1. World A: The Rank Method (排名法)
*Used in Seasons 1-2, 28+*

*   **Mechanism (机制)**:
    $$ C_{Rank, i} = R_{J, i} + R_{V, i} $$
*   **Elimination Rule (淘汰规则)**:
    The contestant with the **highest** $C_{Rank, i}$ (worst combined rank) is eliminated.
    组合得分 $C_{Rank, i}$ **最高**（排名总和最差）的选手被淘汰。
*   **Tie-Breaker (平局规则)**:
    If tied, the one with worse Fan Rank $R_{V, i}$ is eliminated.
    若平局，粉丝排名 $R_{V, i}$ 较差者被淘汰。

### 2.2. World B: The Percentage Method (百分比法)
*Used in Seasons 3-27*

*   **Mechanism (机制)**:
    $$ C_{Pct, i} = \frac{S_{J, i}}{\sum_k S_{J, k}} + \frac{V_{F, i}}{\sum_k V_{F, k}} $$
*   **Elimination Rule (淘汰规则)**:
    The contestant with the **lowest** $C_{Pct, i}$ (smallest total share) is eliminated.
    组合得分 $C_{Pct, i}$ **最低**（总占比最小）的选手被淘汰。

### 2.3. World C: The Judges' Save (法官拯救机制)
*Refined Rule for Season 28+*

*   **Step 1: Bottom Two Identification (确定倒数两名)**
    Identify two contestants $\{c_1, c_2\}$ with the highest $C_{Rank}$.
    根据 $C_{Rank}$ 选出得分最差的两位选手。
*   **Step 2: Probabilistic Save (概率性拯救)**
    We model the judges' decision using a Logit model based on judge scores. Let $c_1$ be the contestant with the lower judge score.
    我们利用基于评委分数的 Logit 模型模拟法官决策。设 $c_1$ 为评委分较低者。
    
    $$ P(Elim\_c_1) = \frac{S_{J, c_2}}{S_{J, c_1} + S_{J, c_2}} $$
    
    *   *Interpretation*: If the opponent $c_2$ has a much higher judge score, $c_1$ is almost certainly eliminated.
    *   *解释*：如果对手 $c_2$ 的评委分显著更高，则 $c_1$ 几乎肯定被淘汰。

---

## 3. Bias Quantification via GNN Proxy (基于GNN代理的偏向性量化)

To quantify "Which method favors fan votes more?", we train a Graph Neural Network (GNN) to learn the mapping from features to elimination outcomes.
为了量化“哪种方法更偏向粉丝投票”，我们训练一个图神经网络（GNN）来学习从特征到淘汰结果的映射。

### 3.1. Model Architecture (模型架构)

*   **Graph Structure (图结构)**: Fully connected graph of contestants in week $w$. (第 $w$ 周选手的全连接图)。
*   **Node Features (节点特征)**: $X_i = [Norm(S_{J, i}), Norm(V_{F, i}), Industry_i]$.
*   **Layer (层)**: GraphSAGE Convolution (captures relative competitiveness). (GraphSAGE 卷积，捕捉相对竞争力)。

### 3.2. Feature Perturbation Analysis (特征扰动分析)

We define the **Fan Bias Score** by measuring the prediction drop when features are shuffled.
我们通过测量特征打乱后的预测下降幅度来定义**粉丝偏向性得分**。

$$ \text{Bias Score} = \frac{\Delta_{Fan}}{\Delta_{Fan} + \Delta_{Judge}} $$

*   $\Delta_{Fan}$: Model error increase when Fan Votes are shuffled. (打乱粉丝票时的模型误差增量)。
*   $\Delta_{Judge}$: Model error increase when Judge Scores are shuffled. (打乱评委分时的模型误差增量)。

---

## 4. Key Results (关键结果)

### 4.1. Bias Score Comparison (偏向性得分对比)

| Method (方法) | Bias Score | Conclusion (结论) |
| :--- | :--- | :--- |
| **Percentage** | **0.77** | **Winner-Takes-All (赢家通吃)**. Highly sensitive to fan popularity spikes. <br> 对粉丝人气爆发极度敏感。 |
| **Rank** | **0.64** | **Linearized (线性化)**. Dampens the effect of "superstars" but still fan-heavy. <br> 抑制了“超级巨星”的效应，但仍偏重粉丝。 |
| **Judges' Save** | **0.53** | **Balanced (平衡)**. The "Veto Power" of judges restores professional standards. <br> 法官的“否决权”恢复了专业标准。 |

### 4.2. Counterfactual Case Study: Bobby Bones (反事实案例：Bobby Bones)

*   **Actual History (Percentage)**: Won Season 27 despite low scores.
    **真实历史（百分比法）**：尽管分数低，仍赢得第27季冠军。
*   **Simulation (Judges' Save)**: **Eliminated in Week 8 (Semi-Finals)**.
    **仿真结果（法官拯救）**：**在第8周（半决赛）被淘汰**。
    *   *Reason*: He fell into the Bottom Two, and judges saved his opponent who had superior technical scores.
    *   *原因*：他跌入倒数两名，法官拯救了技术分更高的对手。
