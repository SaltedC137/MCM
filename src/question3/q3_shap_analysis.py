
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Data Preparation
# ==========================================
def load_and_prepare_data():
    file_path = os.path.join(base_dir, 'data', 'processed', 'estimated_fan_votes_results.csv')
    if not os.path.exists(file_path):
        print("Error: Processed data not found. Run Q1 first.")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    
    # Filter valid rows
    df = df.dropna(subset=['est_fan_vote', 'score'])
    
    # --- Feature Engineering ---
    
    # 1. Contestant Characteristics
    le_industry = LabelEncoder()
    df['industry_code'] = le_industry.fit_transform(df['celebrity_industry'].fillna('Unknown'))
    
    # Age (Simulate if missing, or use actual if available)
    # The problem description says 'celebrity_age_during_season' is in data.
    if 'celebrity_age_during_season' not in df.columns:
        # Fallback: Random age or mean imputation if column missing (should be there based on problem desc)
        # Check raw data columns? Assuming it might be missing in processed.
        # Let's assume processed has it. If not, fill with mean.
        df['age'] = 35 # Placeholder default
    else:
        df['age'] = df['celebrity_age_during_season'].fillna(df['celebrity_age_during_season'].mean())

    # 2. Performance Metrics
    # Judge Score (Normalized per week)
    # Fan Vote (Normalized per week) -> This is a target for Model 2, but can be feature for Model 3
    
    # 3. Partner Features
    # 'ballroom_partner' might be text. Encode it.
    if 'ballroom_partner' in df.columns:
        le_partner = LabelEncoder()
        df['partner_code'] = le_partner.fit_transform(df['ballroom_partner'].fillna('Unknown'))
    else:
        df['partner_code'] = 0

    # 4. Context Features
    df['is_final_week'] = (df['week'] >= df.groupby('season')['week'].transform('max') - 1).astype(int)
    
    # --- Define Feature Set X ---
    feature_cols = ['season', 'week', 'industry_code', 'age', 'partner_code', 'is_final_week']
    
    X = df[feature_cols].copy()
    
    # --- Define Targets Y ---
    # Model 1: Judge Score (Regression)
    y_judge = df['score']
    
    # Model 2: Fan Vote (Regression)
    y_fan = df['est_fan_vote']
    
    # Model 3: Final Placement (Ranking/Regression)
    # We use 'placement' (1 is best). Lower is better.
    # Note: Placement is constant per season for a person. 
    # Maybe 'Weekly Rank' is better?
    # Let's use 'weekly_rank' if available, else placement.
    if 'weekly_rank' in df.columns:
        y_rank = df['weekly_rank']
    else:
        y_rank = df.groupby(['season', 'week'])['score'].rank(ascending=False)

    return X, y_judge, y_fan, y_rank, feature_cols

# ==========================================
# 2. XGBoost + SHAP Pipeline
# ==========================================
def train_and_explain(X, y, target_name, model_type='regressor'):
    print(f"\n--- Training XGBoost for Target: {target_name} ---")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    if model_type == 'regressor':
        model = xgb.XGBRegressor(
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            n_jobs=-1
        )
    else:
        # Ranker (pairwise) - Requires group info, skip for simplicity, treat rank as regression (lower is better)
        model = xgb.XGBRegressor(objective='reg:squarederror') 

    model.fit(X_train, y_train)
    
    # Evaluation
    score = model.score(X_test, y_test)
    print(f"R^2 Score: {score:.4f}")
    
    # SHAP Explanation
    # Use TreeExplainer for XGBoost which is optimized and robust
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    return shap_values, explainer

# ==========================================
# 3. Visualization & Analysis
# ==========================================
def plot_shap_summary(shap_values, target_name, output_dir):
    plt.figure()
    shap.summary_plot(shap_values, show=False)
    plt.title(f"SHAP Summary: Impact on {target_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_summary_{target_name}.png"))
    plt.close()
    print(f"Saved SHAP summary for {target_name}")

def compare_importance(shap_dict, feature_names, output_dir):
    """
    Compare feature importance across Judge vs Fan models.
    Importance = Mean(|SHAP|)
    """
    importance_data = []
    
    for target, shap_vals in shap_dict.items():
        # shap_vals is an Explanation object. .values is the array.
        mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
        # Normalize to sum to 1 for fair comparison
        mean_abs_shap = mean_abs_shap / mean_abs_shap.sum()
        
        for i, feat in enumerate(feature_names):
            importance_data.append({
                'Feature': feat,
                'Target': target,
                'Importance': mean_abs_shap[i]
            })
            
    df_imp = pd.DataFrame(importance_data)
    
    # Plot Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_imp, x='Feature', y='Importance', hue='Target', palette='viridis')
    plt.title("Feature Importance Contrast: Judge vs. Fan vs. Rank", fontsize=16)
    plt.ylabel("Relative Normalized Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_contrast.png"))
    plt.close()
    print("Saved Feature Importance Contrast Plot.")

def main():
    # Setup Output
    output_dir = os.path.join(base_dir, 'results', 'plots', 'question3')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Load
    X, y_judge, y_fan, y_rank, feature_names = load_and_prepare_data()
    
    # 2. Train & Explain Parallel Models
    shap_dict = {}
    
    # Model A: Judge Score
    sv_judge, _ = train_and_explain(X, y_judge, "Judge Score")
    shap_dict['Judge Score'] = sv_judge
    plot_shap_summary(sv_judge, "Judge_Score", output_dir)
    
    # Model B: Fan Vote
    sv_fan, _ = train_and_explain(X, y_fan, "Fan Vote")
    shap_dict['Fan Vote'] = sv_fan
    plot_shap_summary(sv_fan, "Fan_Vote", output_dir)
    
    # Model C: Weekly Rank (Outcome)
    sv_rank, _ = train_and_explain(X, y_rank, "Weekly Rank")
    shap_dict['Weekly Rank'] = sv_rank
    plot_shap_summary(sv_rank, "Weekly_Rank", output_dir)
    
    # 3. Comparative Analysis
    compare_importance(shap_dict, feature_names, output_dir)
    
    print("\nAnalysis Complete. Check results in src/question3/")

if __name__ == "__main__":
    main()
