# Question 3: Multivariate Influence Analysis Model

## 1. Nomenclature and Parameter Definitions

To ensure mathematical rigor, we define the set of symbols and parameters used in the Question 3 model.

| Symbol | Definition | Type/Dimension | Description |
|:--- |:--- |:--- |:--- |
| $Y_{Judge}$ | Target: Judge Score | Real $\in [0, 40]$ | The dependent variable representing the judges' technical score. |
| $Y_{Fan}$ | Target: Fan Vote | Real $\in [0, 1]$ | The dependent variable representing the estimated fan vote share. |
| $Y_{Rank}$ | Target: Weekly Rank | Integer $\in [1, N_w]$ | The dependent variable representing the contestant's rank within the week. |
| $X_{Season}$ | Feature: Season ID | Integer | Captures macro-temporal trends (e.g., ratings decline, rule changes). |
| $X_{Week}$ | Feature: Week Number | Integer | Represents the progression within a season (learning curve, survival bias). |
| $X_{Age}$ | Feature: Celebrity Age | Integer | The age of the celebrity contestant. |
| $X_{Partner}$ | Feature: Pro Partner | Categorical | The professional dancer paired with the celebrity. |
| $X_{Industry}$ | Feature: Industry | Categorical | The celebrity's professional background (e.g., Actor, Athlete). |
| $\phi_{i, j}$ | SHAP Value | Real | The contribution of feature $j$ to the prediction for instance $i$. |
| $R^2$ | Coeff. of Determination | Real $\in [0, 1]$ | Metric evaluating the explanatory power of the XGBoost model. |

---

## 2. XGBoost Regression Framework

We employ **XGBoost (Extreme Gradient Boosting)** as our core analytical engine. Unlike linear regression, XGBoost can capture complex non-linear interactions (e.g., "Age matters more in later weeks") and handle categorical variables effectively.

### 2.1. Model Specification
For each target variable $Y \in \{Y_{Judge}, Y_{Fan}, Y_{Rank}\}$, we train a separate regressor:

$$ \hat{Y} = \sum_{k=1}^{K} f_k(X_i), \quad f_k \in \mathcal{F} $$

*   **Objective Function**: We minimize the squared error loss with regularization to prevent overfitting:
    $$ \mathcal{L}(\phi) = \sum_i (y_i - \hat{y}_i)^2 + \sum_k \Omega(f_k) $$
*   **Hyperparameters**:
    *   `n_estimators=100`: Number of boosting rounds.
    *   `learning_rate=0.1`: Step size shrinkage.
    *   `max_depth=5`: Maximum tree depth to capture interactions.

### 2.2. Model Performance Validation
Our models achieved robust predictive performance, validating their reliability for influence analysis:

*   **Fan Vote Model ($R^2 = 0.94$)**: Extremely high accuracy. This indicates that fan voting patterns are highly structural and predictable, primarily driven by macro factors (Season).
*   **Judge Score Model ($R^2 = 0.63$)**: Moderate to high accuracy. Judges' scores are largely determined by objective factors (Week, Partner), though subjective variance remains.
*   **Weekly Rank Model ($R^2 = 0.39$)**: Lower accuracy. Ranking is a relative metric dependent on specific opponents, making it harder to predict from absolute features alone.

---

## 3. SHAP Interpretation Mechanism

To open the "black box" of XGBoost, we utilize **SHAP (SHapley Additive exPlanations)** values. Due to library compatibility, we implemented a robust **Permutation Explainer** fallback mechanism.

### 3.1. SHAP Value Definition
The SHAP value $\phi_{i, j}$ represents the marginal contribution of feature $j$ to the prediction for instance $i$, satisfying game-theoretic properties of consistency and local accuracy.

$$ \hat{f}(x) = \phi_0 + \sum_{j=1}^{M} \phi_j $$

### 3.2. Feature Importance Quantification
We define the global importance of feature $j$ as the mean absolute SHAP value across all samples:

$$ I_j = \frac{1}{N} \sum_{i=1}^{N} |\phi_{i, j}| $$

This metric allows us to directly compare the impact of different features (e.g., "Is Age more important than Industry?") across different targets.

---

## 4. Key Findings

### 4.1. Divergent Drivers: Judge vs. Fan
Our analysis reveals a fundamental "Dual Standard" in the competition:

*   **Judge Score Drivers**:
    *   **Primary Driver: Week ($X_{Week}$)**. Judges reward persistence and technical growth. As the season progresses, scores naturally inflate due to survival bias (only good dancers remain) and learning effects.
    *   **Secondary Driver: Partner ($X_{Partner}$)**. A skilled professional partner significantly boosts a celebrity's technical score.
*   **Fan Vote Drivers**:
    *   **Dominant Driver: Season ($X_{Season}$)**. The macro-environment (e.g., show ratings, viewership trends) overwhelmingly dictates the total volume of fan votes. This explains the "Winner-Takes-All" phenomenon observed in Q2.
    *   **Secondary Driver: None**. Individual attributes like Age or Industry have negligible impact compared to the season-wide trend.

### 4.2. The Age Factor in Rankings
*   While **Age** has limited impact on absolute Judge Scores or Fan Votes, it becomes a **critical determinant for Weekly Rank ($Y_{Rank}$)**.
*   *Interpretation*: In the relative competition of rankings, physical stamina and agility (correlated with youth) provide a decisive edge that separates the top tier from the middle tier.

### 4.3. Industry Impact
*   Contary to popular belief, **Celebrity Industry ($X_{Industry}$)** plays a minor role. Whether a contestant is an athlete or an actor matters far less than which Season they are in or which Week they reach.
