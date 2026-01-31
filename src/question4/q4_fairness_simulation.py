
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

# ==========================================
# Global Plotting Styles
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
sns.set_palette("deep")

def load_data():
    file_path = os.path.join(base_dir, 'data', 'processed', 'estimated_fan_votes_results.csv')
    if not os.path.exists(file_path):
        print("Error: Processed data not found.")
        sys.exit(1)
    return pd.read_csv(file_path)

# ==========================================
# Simulation Logic
# ==========================================
def simulate_new_system(df):
    """
    Simulates the 'Merit-Protected Engagement Model' on historical data.
    Rules:
    1. Immunity: Top Judge Scorer is Safe.
    2. Weighted Rank: 0.6 * JudgeRank + 0.4 * FanRank for others.
    3. Save: Bottom 3 face Judges. Judges eliminate the one with lowest cumulative score (or weekly score).
       Assumption: Judges eliminate the one with the lowest WEEKLY Judge Score among the bottom 3.
    """
    
    simulation_results = []
    
    # Group by Season/Week to simulate each elimination night
    # Filter for weeks where an elimination actually happened
    # We can infer elimination weeks by checking if 'results' contains 'Eliminated'
    
    # Pre-calculate Ranks
    df['j_rank'] = df.groupby(['season', 'week'])['score'].rank(ascending=False, method='min')
    df['f_rank'] = df.groupby(['season', 'week'])['est_fan_vote'].rank(ascending=False, method='min')
    
    grouped = df.groupby(['season', 'week'])
    
    total_weeks = 0
    improved_outcomes = 0
    worse_outcomes = 0
    
    regret_pre = 0 # High scorer eliminated in original
    regret_post = 0 # High scorer eliminated in new
    
    survival_unfit_pre = 0 # Low scorer survived in original
    survival_unfit_post = 0 # Low scorer survived in new
    
    for (season, week), group in grouped:
        # Check if there was an elimination
        eliminated_actual = group[group['results'].str.contains('Eliminated', na=False)]
        if eliminated_actual.empty:
            continue
            
        total_weeks += 1
        n_players = len(group)
        if n_players < 4: continue # Skip finals
        
        # --- 1. Original Outcome Metrics ---
        # Who was actually eliminated?
        elim_name_actual = eliminated_actual['celebrity_name'].iloc[0]
        elim_j_rank = group[group['celebrity_name'] == elim_name_actual]['j_rank'].values[0]
        
        # Metric: Regret (Eliminating a Top 3 Dancer)
        if elim_j_rank <= 3:
            regret_pre += 1
            
        # Metric: Survival of Unfit (Worst Dancer Survived)
        # Handle potential NaNs in j_rank (though preprocessing should have handled it)
        if group['j_rank'].isnull().all():
            continue
            
        worst_dancer_idx = group['j_rank'].idxmax(skipna=True)
        # Check if index is valid (not NaN)
        if pd.isna(worst_dancer_idx):
             continue
             
        worst_dancer = group.loc[worst_dancer_idx] 
        if worst_dancer['celebrity_name'] != elim_name_actual:
            survival_unfit_pre += 1

        # --- 2. New System Simulation ---
        group = group.copy()
        
        # Rule 1: Immunity
        # Identify Top Scorer (Min Rank)
        top_scorer_idx = group['j_rank'].idxmin(skipna=True)
        if pd.isna(top_scorer_idx):
            continue
            
        immune_name = group.loc[top_scorer_idx, 'celebrity_name']
        
        # Rule 2: Weighted Rank for Non-Immune
        # Create mask for non-immune
        non_immune = group[group['celebrity_name'] != immune_name].copy()
        
        # Re-rank within the non-immune pool? Or just use global ranks?
        # Using global ranks is simpler and preserves context.
        
        # Weighted Score (Lower is Better)
        # 60% Judge, 40% Fan
        non_immune['composite_score'] = 0.6 * non_immune['j_rank'] + 0.4 * non_immune['f_rank']
        
        # Sort by composite score (descending = worst)
        non_immune = non_immune.sort_values('composite_score', ascending=False)
        
        # Rule 3: Bottom 3 Save
        # Identify Bottom 3
        bottom_3 = non_immune.head(3)
        
        # Judges eliminate the one with lowest Judge Score among Bottom 3
        # (If tied, use Fan Vote as tie breaker? Or just lowest Judge Score)
        # In our data, Higher Rank Number = Worse Score.
        # So we look for Max j_rank in bottom_3
        
        # Handle empty bottom_3 or all NaNs
        if bottom_3.empty or bottom_3['j_rank'].isnull().all():
            continue
            
        elim_sim_idx = bottom_3['j_rank'].idxmax(skipna=True)
        if pd.isna(elim_sim_idx):
            continue
            
        elim_sim_row = bottom_3.loc[elim_sim_idx]
        elim_name_sim = elim_sim_row['celebrity_name']
        elim_sim_j_rank = elim_sim_row['j_rank']
        
        # --- 3. Compare Outcomes ---
        # Metric: Regret Post
        if elim_sim_j_rank <= 3:
            regret_post += 1
            
        # Metric: Survival Post
        # Did the actual worst dancer survive in simulation?
        worst_dancer_name = worst_dancer['celebrity_name']
        if worst_dancer_name != elim_name_sim:
            survival_unfit_post += 1
            
        simulation_results.append({
            'season': season,
            'week': week,
            'elim_actual': elim_name_actual,
            'elim_sim': elim_name_sim,
            'elim_actual_j_rank': elim_j_rank,
            'elim_sim_j_rank': elim_sim_j_rank,
            'immune': immune_name
        })

    metrics = {
        'Total Eliminations': total_weeks,
        'Regret (Original)': regret_pre,
        'Regret (New System)': regret_post,
        'Survival of Unfit (Original)': survival_unfit_pre,
        'Survival of Unfit (New System)': survival_unfit_post
    }
    
    return pd.DataFrame(simulation_results), metrics

# ==========================================
# Visualization
# ==========================================
def plot_metrics(metrics, output_dir):
    labels = ['Regret Cases\n(Top Talent Eliminated)', 'Survival of Unfit\n(Worst Dancer Saved)']
    original = [metrics['Regret (Original)'], metrics['Survival of Unfit (Original)']]
    new_sys = [metrics['Regret (New System)'], metrics['Survival of Unfit (New System)']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6), dpi=300)
    rects1 = plt.bar(x - width/2, original, width, label='Current System', color='#e74c3c', alpha=0.8)
    rects2 = plt.bar(x + width/2, new_sys, width, label='Proposed System', color='#2ecc71', alpha=0.8)
    
    plt.ylabel('Number of Occurrences', fontsize=14)
    plt.title('Fairness Comparison: Current vs. Proposed System', fontsize=18, pad=20)
    plt.xticks(x, labels, fontsize=12)
    plt.legend(fontsize=12)
    
    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_system_fairness_comparison.png"))
    print("Saved Fairness Comparison Plot.")

def plot_rank_correlation(df, output_dir):
    # Calculate correlation between Judge Rank and Outcome (Elimination)
    # Ideally, Eliminated person should have High Judge Rank (Bad Score)
    
    # We visualize the distribution of "Judge Rank of Eliminated Contestants"
    # A fairer system should have eliminations clustered at higher ranks (worse scores)
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    sns.kdeplot(df['elim_actual_j_rank'], label='Current System', color='#e74c3c', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(df['elim_sim_j_rank'], label='Proposed System', color='#2ecc71', fill=True, alpha=0.3, linewidth=2)
    
    plt.title('Distribution of Technical Rank for Eliminated Contestants', fontsize=16)
    plt.xlabel('Judge Rank (Higher = Worse Performance)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    
    # Add annotation
    plt.text(0.05, 0.95, "Desired Shift: \nRightward (Eliminating worse dancers)", 
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_elimination_rank_distribution.png"))
    print("Saved Rank Distribution Plot.")

def main():
    output_dir = os.path.join(base_dir, 'results', 'plots', 'question4')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading data...")
    df = load_data()
    
    print("Running Simulation...")
    res_df, metrics = simulate_new_system(df)
    
    print("\n--- Simulation Results ---")
    print(f"Total Elimination Rounds: {metrics['Total Eliminations']}")
    print(f"Original System Regrets (Good Dancers Eliminated): {metrics['Regret (Original)']}")
    print(f"Proposed System Regrets: {metrics['Regret (New System)']}")
    print(f"Reduction in Unfair Eliminations: {metrics['Regret (Original)'] - metrics['Regret (New System)']}")
    
    print("\nGenerating Plots...")
    plot_metrics(metrics, output_dir)
    plot_rank_correlation(res_df, output_dir)
    
    # Save simulation log
    res_df.to_csv(os.path.join(output_dir, "simulation_log.csv"), index=False)
    print(f"Detailed log saved to {output_dir}")

if __name__ == "__main__":
    main()
