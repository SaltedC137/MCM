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
| $\mathcal{G}_w$ | Competition Graph | Graph Structure | Graph representing the competition in week $w$. |
| $\Delta_{Fan}$ | Perturbation Error | Real | Model error increase when fan votes are shuffled. |

---

## 2. Parallel World Simulation Logic

We construct "Parallel Worlds" to simulate the fate of each contestant under three different rule sets, keeping their performance data ($S_{J, i}, V_{F, i}$) constant. This counterfactual approach allows us to isolate the effect of the voting mechanism itself.

### 2.1. World A: The Rank Method
*Used in Seasons 1-2, 28+*

In this world, we simulate the "Rank Sum" logic. Lower sums are better.

*   **Data Transformation**:
    *   $R_{J, i} = \text{Rank}(S_{J, i})$ descending (Higher score $\to$ Rank 1).
    *   $R_{V, i} = \text{Rank}(V_{F, i})$ descending (Higher votes $\to$ Rank 1).
*   **Mechanism**:
    $$ C_{Rank, i} = R_{J, i} + R_{V, i} $$
*   **Elimination Rule**:
    The contestant with the **highest** $C_{Rank, i}$ (worst combined rank) is eliminated.
    $$ \text{Eliminated} = \arg\max_{i} (C_{Rank, i}) $$
*   **Tie-Breaker**:
    If $C_{Rank, i} = C_{Rank, j}$, the contestant with the worse Fan Rank ($R_{V}$) is eliminated.

### 2.2. World B: The Percentage Method
*Used in Seasons 3-27*

In this world, we simulate the "Proportional Share" logic. Higher shares are better.

*   **Data Transformation**:
    *   $P_{J, i} = \frac{S_{J, i}}{\sum_{k \in C_w} S_{J, k}}$
    *   $P_{V, i} = \frac{V_{F, i}}{\sum_{k \in C_w} V_{F, k}}$
*   **Mechanism**:
    $$ C_{Pct, i} = P_{J, i} + P_{V, i} $$
*   **Elimination Rule**:
    The contestant with the **lowest** $C_{Pct, i}$ (smallest total share) is eliminated.
    $$ \text{Eliminated} = \arg\min_{i} (C_{Pct, i}) $$

### 2.3. World C: The Judges' Save
*Refined Rule for Season 28+*

This world introduces a stochastic component to model human decision-making (the judges' intervention).

*   **Step 1: Bottom Two Identification**
    We first calculate the Rank Combined Score $C_{Rank}$ as in World A. The two contestants with the highest $C_{Rank}$ are identified as the "Bottom Two" set $\mathcal{B} = \{c_1, c_2\}$.
*   **Step 2: Probabilistic Save Modeling**
    We model the judges' decision using a **Logit Model** based on the judge score differential. We assume judges are biased towards saving the contestant with higher technical merit ($S_J$), but the decision is probabilistic.
    
    Let $c_1$ be the contestant with the lower judge score in the bottom two. The probability of $c_1$ being eliminated is:
    
    $$ P(Elim\_c_1) = \text{Sigmoid}(\beta \cdot (S_{J, c_2} - S_{J, c_1})) \approx \frac{S_{J, c_2}}{S_{J, c_1} + S_{J, c_2}} $$
    
    *   *Interpretation*: If the opponent $c_2$ has a significantly higher judge score, the probability of $c_1$ being eliminated approaches 1.
    *   *Implementation*: We use Monte Carlo sampling (`np.random.random()`) to determine the outcome based on this probability.

---

## 3. Bias Quantification via GNN Proxy

To quantify "Which method favors fan votes more?" without relying on subjective observation, we train a **Graph Neural Network (GNN)** to learn the implicit mapping from features to elimination outcomes.

### 3.1. Model Architecture

We model each week of competition as a fully connected graph $\mathcal{G}_w = (\mathcal{V}_w, \mathcal{E}_w)$, where nodes represent contestants.

*   **Node Features**: $X_i = [\text{Norm}(S_{J, i}), \text{Norm}(V_{F, i}), \text{IndustryCode}_i]$.
*   **Edge Structure**: Fully connected (every contestant competes with every other contestant).
*   **GNN Layer**: We use **GraphSAGE** (SAGEConv) layers. GraphSAGE aggregates information from neighbors, effectively capturing the *relative* standing of a contestant compared to their peers (which is crucial for ranking systems).
    $$ h_i^{(l+1)} = \sigma \left( W \cdot \text{CONCAT}(h_i^{(l)}, \text{AGG}(\{h_j^{(l)}, \forall j \in \mathcal{N}(i)\})) \right) $$
*   **Target**: Binary classification (1 = Eliminated, 0 = Safe) based on the simulation ground truth.

### 3.2. Feature Perturbation Analysis (Proxy for SHAP)

Since direct SHAP calculation is computationally expensive for GNNs, we implement a **Permutation Feature Importance** algorithm to quantify bias.

1.  **Baseline**: Calculate the model's prediction error (MSE) on the original dataset.
2.  **Perturb Fan Votes**: Randomly shuffle the $V_{F}$ column across the dataset and measure the increase in error ($\Delta_{Fan}$). This destroys the information in the fan vote feature.
3.  **Perturb Judge Scores**: Randomly shuffle the $S_{J}$ column and measure the increase in error ($\Delta_{Judge}$).
4.  **Bias Score Calculation**:
    $$ \text{Bias Score} = \frac{\Delta_{Fan}}{\Delta_{Fan} + \Delta_{Judge}} $$
    *   Score $\to 1$: The method is dominated by Fan Votes.
    *   Score $\to 0$: The method is dominated by Judge Scores.
    *   Score $\approx 0.5$: The method is balanced.

---

## 4. Key Results

### 4.1. Bias Score Comparison

| Method | Bias Score | Conclusion |
| :--- | :--- | :--- |
| **Percentage** | **0.77** | **Winner-Takes-All**. The bias score is significantly > 0.5, indicating extreme sensitivity to fan popularity. Since fan votes follow a Zipfian distribution, top stars accumulate massive vote shares that override judge scores. |
| **Rank** | **0.64** | **Linearized**. The score drops but remains > 0.5. Ranking linearizes the vote distribution, dampening the "superstar effect," but fan influence is still primary. |
| **Judges' Save** | **0.53** | **Balanced**. The score is nearly 0.5. The "Veto Power" of judges in the final step effectively neutralizes the fan bias, restoring professional standards. |

### 4.2. Counterfactual Case Study: Bobby Bones (S27)

*   **Actual History (Percentage Method)**: Bobby Bones won Season 27. Our simulation confirms he had **0%** elimination risk under this rule, despite low judge scores.
*   **Simulation (Judges' Save)**: In our parallel world simulation, Bobby Bones faces a **critical elimination risk in Week 8**.
    *   *Mechanism*: In Week 8, his low judge scores would have placed him in the Bottom Two. Our Logit model predicts that judges would have saved his opponent (who had higher technical scores) with high probability.
    *   *Conclusion*: The Judges' Save mechanism would likely have prevented the "Bobby Bones Controversy."
