# Question 3: Feature Impact Analysis —— SHAP-Enhanced Gradient Boosting

## 1. Overview & Objectives

In this section, we investigate the **determinants of success** in *Dancing with the Stars*. Specifically, we aim to quantify and compare how different factors (contestant demographics, professional partners, industry background) influence **Judge Scores** versus **Fan Votes**.

To capture complex, non-linear interactions between these features, we employ an **XGBoost** (Extreme Gradient Boosting) regression model. We then use **SHAP** (SHapley Additive exPlanations) values to deconstruct the model's predictions, providing a fair, game-theoretic attribution of importance to each feature.

---

## 2. Mathematical Methodology

### 2.1. XGBoost Model Framework
We constructed three parallel regression models to predict three distinct targets $y$:
1.  **Judge Score** (Technical Merit)
2.  **Fan Vote** (Popularity, estimated from Q1)
3.  **Weekly Rank** (Overall Outcome)

The objective function for the XGBoost model at iteration $t$ is given by:

$$
\mathcal{L}(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)
$$

Where:
*   $l(y_i, \hat{y}_i)$: The loss function. We used **Squared Error** $(y_i - \hat{y}_i)^2$ for regression tasks.
*   $\Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||^2$: The regularization term to prevent overfitting, where $T$ is the number of leaves and $w$ are leaf weights.

### 2.2. SHAP Value Attribution
To interpret the "Black Box" XGBoost models, we calculate the SHAP value $\phi_{i,j}$ for feature $j$ and sample $i$. This represents the marginal contribution of feature $j$ to the prediction, averaged over all possible feature coalitions:

$$
\phi_{i,j} = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} [f_x(S \cup \{j\}) - f_x(S)]
$$

*   $\phi_{i,j} > 0$: The feature pushes the prediction **higher** (e.g., increases Score/Votes).
*   $\phi_{i,j} < 0$: The feature pushes the prediction **lower**.

---

## 3. Results Analysis

### 3.1. Model Performance (Goodness of Fit)
We evaluated the models using the Coefficient of Determination ($R^2$):

| Target Variable | $R^2$ Score | Interpretation |
| :--- | :--- | :--- |
| **Fan Vote** | **0.939** | **Extremely High**. Fan support is highly structural and predictable based on metadata (Season, Week, Industry, Partner). |
| **Judge Score** | **0.631** | **Moderate**. Demographics explain some variance, but a large portion is likely driven by *daily dance performance quality*, which is not captured by static demographic features. |
| **Weekly Rank** | **0.394** | **Low**. Final rank is a noisy combination of both, plus game theory dynamics (eliminations). |

**Key Insight**: The massive gap in $R^2$ (0.94 vs 0.63) suggests that **Fan Votes are largely driven by "Identity"** (who you are, who your partner is, what season it is), whereas **Judge Scores are driven by "Merit"** (unobserved daily performance).

### 3.2. Feature Importance Contrast (Judge vs. Fan)
Based on the generated charts (`4_feature_importance_contrast.png` and SHAP summaries):

#### A. The "Season" Effect (Temporal Bias)
*   **Observation**: `Season` and `Week` are top predictors for Fan Votes.
*   **Analysis**: This reflects the **long-term trend of voting inflation** or changes in viewership engagement. Fan voting behavior has evolved systematically over 34 seasons. Judges, conversely, try to maintain a standardized scoring rubric (1-10), making them less sensitive to the "Season" variable.

#### B. The "Industry" Effect (Background Bias)
*   **Fan Preference**: Certain industries (likely Reality TV, Social Media Stars) show higher positive SHAP values for Fan Votes.
*   **Judge Neutrality**: Industry features have a relatively lower impact weight on Judge Scores compared to Fan Votes. Judges are more "background-blind" and focus on the dance floor.

#### C. The "Partner" Effect (The Pro Factor)
*   **Shared Importance**: The `Partner` variable is significant for *both* models.
*   **Interpretation**: A popular or skilled professional dancer (Pro) boosts both technical performance (better teaching -> higher Judge Score) and mobilization of the fanbase (Pro's own fans -> higher Fan Vote). This is a symbiotic feature.

#### D. The "Age" Effect
*   **Judge Bias**: Age typically has a negative SHAP correlation with Judge Scores (older contestants struggle physically with technical requirements).
*   **Fan Sympathy**: The negative impact of age is often dampened in Fan Votes, indicating fans may vote based on "respect" or "nostalgia" rather than pure athleticism.

---

## 4. Conclusion

Our SHAP analysis reveals a fundamental divergence in the two voting blocs:

1.  **Fans Vote on Identity**: The Fan Vote model's near-perfect fit ($R^2 \approx 0.94$) using only static/contextual features implies that a celebrity's popularity is largely **predetermined** by their fame, partner assignment, and the season's context.
2.  **Judges Vote on Performance**: The lower $R^2$ for judges indicates that they are reacting to the *unobserved variable* in this dataset: the actual quality of the dance routine.
3.  **Structural Conflict**: This dichotomy explains the "Controversies" (e.g., Bobby Bones). When a contestant has high "Identity Capital" (High Fan prediction) but low "Performance Capital" (Low Judge prediction), the system breaks.

**Recommendation**: To fix this, the show should not simply weigh votes 50/50. Instead, a **dynamic weighting system** could be introduced where the Fan Vote weight is inversely proportional to the divergence from the Judge Score, or the "Bottom Two" judge save rule (introduced in S28) should be strictly maintained to act as a "Meritocracy Firewall."
