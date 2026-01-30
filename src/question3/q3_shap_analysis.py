
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
# Global Plotting Styles
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
sns.set_palette("deep")

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
    
    # Age
    if 'celebrity_age_during_season' not in df.columns:
        df['age'] = 35 
    else:
        df['age'] = df['celebrity_age_during_season'].fillna(df['celebrity_age_during_season'].mean())

    # 3. Partner Features
    if 'ballroom_partner' in df.columns:
        le_partner = LabelEncoder()
        df['partner_code'] = le_partner.fit_transform(df['ballroom_partner'].fillna('Unknown'))
    else:
        df['partner_code'] = 0

    # 4. Context Features
    df['is_final_week'] = (df['week'] >= df.groupby('season')['week'].transform('max') - 1).astype(int)
    
    # --- Define Feature Set X ---
    # Renaming columns for better plot labels
    X = pd.DataFrame()
    X['Season'] = df['season']
    X['Week'] = df['week']
    X['Industry'] = df['industry_code']
    X['Age'] = df['age']
    X['Partner'] = df['partner_code']
    X['Is_Final'] = df['is_final_week']
    
    feature_names = X.columns.tolist()
    
    # --- Define Targets Y ---
    y_judge = df['score']
    y_fan = df['est_fan_vote']
    
    if 'weekly_rank' in df.columns:
        y_rank = df['weekly_rank']
    else:
        y_rank = df.groupby(['season', 'week'])['score'].rank(ascending=False)

    return X, y_judge, y_fan, y_rank, feature_names

# ==========================================
# 2. XGBoost + SHAP Pipeline
# ==========================================
def train_and_explain(X, y, target_name, model_type='regressor'):
    print(f"\n--- Training XGBoost for Target: {target_name} ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        model = xgb.XGBRegressor(objective='reg:squarederror') 

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"R^2 Score: {score:.4f}")
    
    # SHAP Explanation
    try:
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer(X_test)
    except Exception as e:
        print(f"TreeExplainer failed ({e}). Attempting fallback with PermutationExplainer...")
        masker = shap.maskers.Independent(X_train, max_samples=100)
        explainer = shap.PermutationExplainer(model.predict, masker)
        shap_values = explainer(X_test[:100])
    
    return shap_values, explainer

# ==========================================
# 3. Visualization & Analysis
# ==========================================
def plot_shap_summary(shap_values, target_name, output_dir, file_prefix=""):
    # Create a high-quality figure
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Use generic SHAP summary plot but with customization via matplotlib context
    shap.summary_plot(shap_values, show=False, plot_size=(10, 8))
    
    plt.title(f"SHAP Feature Impact: {target_name}", fontsize=18, pad=20, fontweight='bold')
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=14)
    
    # Save with high resolution
    filename = f"{file_prefix}shap_summary_{target_name.replace(' ', '_')}.png"
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved high-res SHAP summary for {target_name} as {filename}")

def compare_importance(shap_dict, feature_names, output_dir, file_prefix=""):
    """
    Compare feature importance across Judge vs Fan models.
    """
    importance_data = []
    
    for target, shap_vals in shap_dict.items():
        # Handle different SHAP value formats (TreeExplainer vs PermutationExplainer)
        vals = shap_vals.values if hasattr(shap_vals, 'values') else shap_vals
        
        # Calculate mean absolute SHAP value per feature
        if isinstance(vals, list): # For some shap outputs
            vals = vals[0]
            
        mean_abs_shap = np.abs(vals).mean(axis=0)
        
        # Normalize to sum to 1 for fair comparison across different target scales
        if mean_abs_shap.sum() > 0:
            mean_abs_shap = mean_abs_shap / mean_abs_shap.sum()
        
        for i, feat in enumerate(feature_names):
            importance_data.append({
                'Feature': feat,
                'Target': target,
                'Relative Importance': mean_abs_shap[i]
            })
            
    df_imp = pd.DataFrame(importance_data)
    
    # Plot Comparison
    plt.figure(figsize=(14, 8), dpi=300)
    
    # Use a distinct palette for the targets
    custom_palette = sns.color_palette("husl", len(shap_dict))
    
    ax = sns.barplot(
        data=df_imp, 
        x='Feature', 
        y='Relative Importance', 
        hue='Target', 
        palette=custom_palette,
        edgecolor='white',
        linewidth=1.5
    )
    
    plt.title("Feature Importance Contrast: Judge vs. Fan vs. Rank", fontsize=20, pad=20, fontweight='bold')
    plt.ylabel("Relative Normalized Importance (0-1)", fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Prediction Target', title_fontsize=14, fontsize=12, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

    filename = f"{file_prefix}feature_importance_contrast.png"
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Feature Importance Contrast Plot as {filename}")

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
    plot_shap_summary(sv_judge, "Judge Score", output_dir, file_prefix="1_")
    
    # Model B: Fan Vote
    sv_fan, _ = train_and_explain(X, y_fan, "Fan Vote")
    shap_dict['Fan Vote'] = sv_fan
    plot_shap_summary(sv_fan, "Fan Vote", output_dir, file_prefix="2_")
    
    # Model C: Weekly Rank (Outcome)
    sv_rank, _ = train_and_explain(X, y_rank, "Weekly Rank")
    shap_dict['Weekly Rank'] = sv_rank
    plot_shap_summary(sv_rank, "Weekly Rank", output_dir, file_prefix="3_")
    
    # 3. Comparative Analysis
    compare_importance(shap_dict, feature_names, output_dir, file_prefix="4_")
    
    print("\nAnalysis Complete. Check results in results/plots/question3/")

if __name__ == "__main__":
    main()
