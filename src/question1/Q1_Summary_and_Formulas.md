# Comprehensive Mathematical Framework and Algorithm Specification

## 1. Nomenclature and Parameter Definitions

To ensure mathematical rigor, we first define the set of symbols and parameters used throughout the model.

| Symbol            | Definition             | Type/Dimension                | Description                                                                    |
|:----------------- |:---------------------- |:----------------------------- |:------------------------------------------------------------------------------ |
| $s$               | Season Index           | Integer $\in [1, 34]$         | The season number.                                                             |
| $w$               | Week Index             | Integer $\in [1, 12]$         | The week number within a season.                                               |
| $N_w$             | Number of Contestants  | Integer                       | The number of active contestants in week $w$.                                  |
| $C_w$             | Set of Contestants     | Set $\{c_1, \dots, c_{N_w}\}$ | The specific contestants participating in week $w$.                            |
| $S_j^{(i)}$       | Raw Judge Score        | Real $\in [0, 30]$ (or 40)    | The raw score received by contestant $i$ from judges.                          |
| $\bar{S}_j^{(i)}$ | Mean Judge Score       | Real $\in [0, 10]$            | The arithmetic mean of valid judge scores for contestant $i$.                  |
| $J^{(i)}$         | Normalized Judge Score | Real $\in [0, 1]$             | The judge's contribution to the total score (Share or Rank).                   |
| $F^{(i)}$         | Latent Fan Vote        | Real $\in [0, 1]$             | The **unknown** fan vote share or rank (Target variable).                      |
| $T^{(i)}$         | Total Composite Score  | Real                          | The final score used for elimination ($J^{(i)} + F^{(i)}$).                    |
| $e_{obs}$         | Observed Elimination   | Index $\in C_w$               | The contestant actually eliminated in historical data (Ground Truth).          |
| $\mathcal{M}$     | Monte Carlo Iterations | Integer ($5000$)              | The number of simulation trials per week.                                      |
| $\alpha$          | Dirichlet Prior        | Vector $\in \mathbb{R}^{N_w}$ | Hyperparameter for fan vote distribution (set to $\mathbf{1}$ for uniformity). |

---

## 2. Data Preprocessing and Feature Engineering

Before simulation, we transform the raw scorecard into a structured format suitable for the inverse optimization problem.

### 2.1. Variable Standardization (Judge Score Share)

Since the number of judges varies (3 or 4) and the max score changes, we normalize absolute scores into a **relative competitive share**. This eliminates scale bias.

$$
J_{share}^{(i)} = \frac{\bar{S}_j^{(i)}}{\sum_{k=1}^{N_w} \bar{S}_j^{(k)}}
$$

* **Rationale**: In a "share of total points" system, what matters is not the absolute score (e.g., 27/30), but the proportion of the total "pie" of points a contestant secures against their rivals.

### 2.2. Ground Truth Extraction

We parse the textual `Results` column to define the constraints:

* $e_{obs} \neq \emptyset$: If a contestant is eliminated in week $w$, their index defines the **lower bound constraint** for the simulation.
* $W_{withdrew}$: If `is_withdrew = 1`, the contestant is removed from the set $C_w$ for modeling purposes, as their exit was not determined by the score function $T^{(i)}$.

---

## 3. Mathematical Modeling of Voting Systems

We formulate the problem as finding the posterior distribution of $F^{(i)}$ given the observed outcome $e_{obs}$. We model two distinct eras of rules.

### 3.1. Model A: The Percentage-Based System (Seasons 3-27)

**Mechanism**: Scores are additive percentages.
**Constraint**: The person with the **lowest** total score is eliminated.

* **Judge Input**: $J^{(i)} = J_{share}^{(i)}$
* **Fan Unknown**: $\mathbf{F} \sim \text{Dirichlet}(\alpha_1, \dots, \alpha_{N_w})$
  * *Assumption*: Without prior polling data, we assume a flat prior ($\alpha_k = 1, \forall k$), meaning every fan vote combination is initially equally likely.
* **Composite Score**: $T^{(i)} = J^{(i)} + F^{(i)}$
* **Inequality Constraint**:
  For a simulation to be valid, the observed eliminated contestant $e_{obs}$ must satisfy:
  
  $$
  T^{(e_{obs})} < T^{(k)} \quad \forall k \in C_w \setminus \{e_{obs}\}
  $$
  
  *Tie-breaking Rule*: If $T^{(e_{obs})} = T^{(k)}$, the one with the lower fan vote $F$ is eliminated.

### 3.2. Model B: The Rank-Sum System (Seasons 1-2, 28-34)

**Mechanism**: Scores are ordinal ranks (1 = Best).
**Constraint**: The person with the **highest** rank sum (worst performance) is at risk.

* **Judge Input**: $J^{(i)} = \text{Rank}(\bar{S}_j^{(i)})$ (Ascending: Highest Score $\to$ Rank 1)
* **Fan Unknown**: $\mathbf{F} \sim \text{Permutation}(\{1, 2, \dots, N_w\})$
  * *Assumption*: Fan ranks are a random permutation of integers from 1 to $N_w$.
* **Composite Score**: $T^{(i)} = J^{(i)} + F^{(i)}$
* **Inequality Constraint (S1-S2)**:
  
  $$
  T^{(e_{obs})} > T^{(k)} \quad \forall k \in C_w \setminus \{e_{obs}\}
  $$
* **Inequality Constraint (S28+ "Bottom Two")**:
  The judges save one of the bottom two. Thus, the eliminated person must be in the bottom two:
  
  $$
  e_{obs} \in \{ i \mid \text{Rank}(T^{(i)}) \ge N_w - 1 \}
  $$

---

## 4. Algorithm Specification: Monte Carlo Rejection Sampling

Since analytical derivation of the feasible region for $\mathbf{F}$ is complex (especially with high dimensions), we use a stochastic approach.

### 4.1. Algorithm Pseudocode

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

### 4.2. Parameter Justification

* **Simulation Count ($\mathcal{M} = 5000$)**:
  * We chose 5000 iterations as a trade-off between convergence and computational cost.
  * For a typical week with 10 contestants, the solution space is high-dimensional. 5000 samples provide a standard error of mean $< 1\%$, which is sufficient for robust estimation.
* **Dirichlet Prior ($\alpha = \mathbf{1}$)**:
  * This is the "non-informative" prior. It asserts that before observing the elimination, we consider a fan vote split of $(90\%, 10\%)$ to be just as probable as $(50\%, 50\%)$. This prevents the model from introducing bias.

---

## 5. Post-Processing Metrics

### 5.1. Estimation of Fan Votes ($\hat{V}_f$)

The final estimated fan vote for contestant $i$ is the expected value over the valid posterior distribution:

$$
\hat{V}_f^{(i)} = \mathbb{E}[F^{(i)} \mid e_{obs}] \approx \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} v^{(i)}
$$

### 5.2. Certainty Quantification ($\kappa$)

We define a metric $\kappa \in [0, 1]$ to quantify the "tightness" of the constraints. If a contestant *must* have had a low fan vote to be eliminated, the variance of valid samples will be low, and certainty will be high.

$$
\kappa_i = 1 - \frac{\sigma_i}{\mu_i + \epsilon}
$$

* $\sigma_i$: Standard deviation of valid fan vote samples for contestant $i$.
* $\mu_i$: Mean estimated fan vote.
* $\epsilon = 0.1$: A regularization term to prevent division by zero for low-vote contestants.

**Interpretation**:

* $\kappa \approx 1.0$: Deterministic. The outcome can *only* happen if the fan vote is exactly this value.
* $\kappa < 0.3$: High Uncertainty. The observed elimination is consistent with a wide range of fan vote scenarios.
