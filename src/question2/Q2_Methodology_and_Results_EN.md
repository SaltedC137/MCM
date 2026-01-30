# Causal Inference Enhanced Simulation Model for Voting Method Comparison

## 1. Nomenclature and Parameter Definitions

To ensure mathematical rigor, we define the set of symbols and parameters used in the Question 2 model.

| Symbol | Definition | Type/Dimension | Description |
|:--- |:--- |:--- |:--- |
| $S_{J, i}$ | Judge Score | Real $\in [0, 40]$ | The raw judge score for contestant $i$. |
| $V_{F, i}$ | Fan Vote Estimate | Real $\in [0, 1]$ | The estimated fan vote share (from Q1 Model). |
| $R_{J, i}$ | Judge Rank | Integer $\in [1, N_w]$ | Rank based on judge score (1 = Highest Score). |
| $R_{V, i}$ | Fan Rank | Integer $\in [1, N_w]$ | Rank based on fan vote (1 = Highest Vote). |
| $C_{Rank, i}$ | Rank Combined Score | Integer | Combined score under Rank Method ($R_{J, i} + R_{V, i}$). |
| $C_{Pct, i}$ | Pct Combined Score | Real $\in [0, 2]$ | Combined score under Percentage Method. |
| $P(Elim|i)$ | Elimination Prob | Real $\in [0, 1]$ | Probability of being eliminated under Judges' Save. |
| $Bias_{Method}$ | Fan Bias Score | Real $\in [0, 1]$ | GNN-derived contribution of fan votes to the outcome. |

---

## 2. Parallel World Simulation Logic

We construct "Parallel Worlds" to simulate the fate of each contestant under three different rule sets, keeping their performance data ($S_{J, i}, V_{F, i}$) constant.

### 2.1. World A: The Rank Method
*Used in Seasons 1-2, 28+*

*   **Mechanism**:
    $$ C_{Rank, i} = R_{J, i} + R_{V, i} $$
*   **Elimination Rule**:
    The contestant with the **highest** $C_{Rank, i}$ (worst combined rank) is eliminated.
*   **Tie-Breaker**:
    If tied, the one with worse Fan Rank $R_{V, i}$ is eliminated.

### 2.2. World B: The Percentage Method
*Used in Seasons 3-27*

*   **Mechanism**:
    $$ C_{Pct, i} = \frac{S_{J, i}}{\sum_k S_{J, k}} + \frac{V_{F, i}}{\sum_k V_{F, k}} $$
*   **Elimination Rule**:
    The contestant with the **lowest** $C_{Pct, i}$ (smallest total share) is eliminated.

### 2.3. World C: The Judges' Save
*Refined Rule for Season 28+*

*   **Step 1: Bottom Two Identification**
    Identify two contestants $\{c_1, c_2\}$ with the highest $C_{Rank}$.
*   **Step 2: Probabilistic Save**
    We model the judges' decision using a Logit model based on judge scores. Let $c_1$ be the contestant with the lower judge score.
    
    $$ P(Elim\_c_1) = \frac{S_{J, c_2}}{S_{J, c_1} + S_{J, c_2}} $$
    
    *   *Interpretation*: If the opponent $c_2$ has a much higher judge score, $c_1$ is almost certainly eliminated.

---

## 3. Bias Quantification via GNN Proxy

To quantify "Which method favors fan votes more?", we train a Graph Neural Network (GNN) to learn the mapping from features to elimination outcomes.

### 3.1. Model Architecture

*   **Graph Structure**: Fully connected graph of contestants in week $w$.
*   **Node Features**: $X_i = [Norm(S_{J, i}), Norm(V_{F, i}), Industry_i]$.
*   **Layer**: GraphSAGE Convolution (captures relative competitiveness).

### 3.2. Feature Perturbation Analysis

We define the **Fan Bias Score** by measuring the prediction drop when features are shuffled.

$$ \text{Bias Score} = \frac{\Delta_{Fan}}{\Delta_{Fan} + \Delta_{Judge}} $$

*   $\Delta_{Fan}$: Model error increase when Fan Votes are shuffled.
*   $\Delta_{Judge}$: Model error increase when Judge Scores are shuffled.

---

## 4. Key Results

### 4.1. Bias Score Comparison

| Method | Bias Score | Conclusion |
| :--- | :--- | :--- |
| **Percentage** | **0.77** | **Winner-Takes-All**. Highly sensitive to fan popularity spikes. |
| **Rank** | **0.64** | **Linearized**. Dampens the effect of "superstars" but still fan-heavy. |
| **Judges' Save** | **0.53** | **Balanced**. The "Veto Power" of judges restores professional standards. |

### 4.2. Counterfactual Case Study: Bobby Bones

*   **Actual History (Percentage)**: Won Season 27 despite low scores.
*   **Simulation (Judges' Save)**: **Eliminated in Week 8 (Semi-Finals)**.
    *   *Reason*: He fell into the Bottom Two, and judges saved his opponent who had superior technical scores.
