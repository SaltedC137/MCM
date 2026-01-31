
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import entropy

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

def calculate_entropy(series):
    """
    Calculate normalized entropy (0-1) for a series of values.
    Uses min-max normalization to ensure non-negative values if needed,
    but here inputs (scores, votes) are positive.
    """
    # Drop NaNs
    series = series.dropna()
    
    # Handle zeros or negatives just in case
    if series.min() < 0:
        # Shift to positive if needed, though scores shouldn't be negative
        series = series - series.min() + 1e-6
        
    total = series.sum()
    if total == 0:
        return 0
        
    probs = series / total
    # Shannon entropy with base e
    # Normalize by ln(n) to get 0-1 range
    n = len(series)
    if n <= 1:
        return 0
        
    ent = entropy(probs)
    max_ent = np.log(n)
    
    return ent / max_ent

def calculate_z_score(series):
    if series.std() == 0:
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / series.std()

def simulate_entropy_system(df):
    """
    Simulates the Entropy-based Dynamic Weighted Model.
    """
    df = df.copy()
    
    # 1. Feature Engineering: Momentum
    # Sort to calculate diff
    df = df.sort_values(['season', 'celebrity_name', 'week'])
    df['prev_score'] = df.groupby(['season', 'celebrity_name'])['score'].shift(1)
    df['momentum'] = df['score'] - df['prev_score']
    df['momentum'] = df['momentum'].fillna(0) # First week has 0 momentum
    
    # Storage for simulation results and weights
    simulation_log = []
    weight_log = []
    
    # Metrics
    regret_pre = 0
    regret_post = 0
    survival_unfit_pre = 0
    survival_unfit_post = 0
    total_eliminations = 0
    
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), group in grouped:
        # Only process elimination weeks
        eliminated_actual = group[group['results'].str.contains('Eliminated', na=False)]
        if eliminated_actual.empty:
            continue
            
        n_players = len(group)
        if n_players < 3: continue # Skip finals with too few people
        
        # --- Stage 1: Standardization ---
        # Z-Scores
        z_j = calculate_z_score(group['score'])
        z_v = calculate_z_score(group['est_fan_vote'])
        
        # Momentum (already calc, but could Z-score it too? Let's keep it raw or scaled)
        # Scaling momentum to be comparable to Z-score (roughly -3 to 3)
        # Max score jump is usually ~3-4 points. 
        mom = group['momentum']
        
        # Calculate entropy
        h_j = calculate_entropy(group['score'])
        h_v = calculate_entropy(group['est_fan_vote'])
        
        # Calculate initial weights based on entropy
        denom = (1 - h_j) + (1 - h_v)
        if denom == 0:
            w_j_raw = 0.5
        else:
            w_j_raw = (1 - h_j) / denom
            
        # LOGIC CHANGE: 
        # If H_V is very low (Fans are biased), w_V becomes high.
        # We want to LIMIT the Fan Weight when it's too high.
        # So we apply a stricter bounds on w_j (Judge Weight).
        # We ensure Judges always have at least 0.35 weight, and max 0.65.
        w_j = np.clip(w_j_raw, 0.35, 0.65)
        w_v = 1 - w_j
        
        # Record weights
        weight_log.append({
            'season': season,
            'week': week,
            'w_j': w_j,
            'w_v': w_v,
            'h_j': h_j,
            'h_v': h_v
        })
        
        # --- Stage 3: Composite Score ---
        # Formula: C = w_J * Z_J + w_V * Z_V + alpha * Momentum
        alpha = 0.05
        composite_score = w_j * z_j + w_v * z_v + alpha * mom
        
        # --- Simulation Logic ---
        # Identify actual elimination info
        elim_name_actual = eliminated_actual['celebrity_name'].iloc[0]
        # Rank by judge score for metrics (Higher score = Rank 1)
        # Handle ties: method='min' means if 3 people have top score, they are all rank 1
        group_j_rank = group['score'].rank(ascending=False, method='min')
        elim_actual_j_rank = group_j_rank[group['celebrity_name'] == elim_name_actual].iloc[0]
        
        # Identify Worst Dancer (Technically)
        if group['score'].dropna().empty:
            continue
            
        min_score = group['score'].min()
        # Worst dancer is ANYONE who has the minimum score
        worst_dancers = group[group['score'] == min_score]['celebrity_name'].values
        
        # --- New System Logic ---
        # 1. Rank by Composite Score (Descending)
        sim_indices = composite_score.sort_values(ascending=True).index # Ascending: Lowest score first
        
        # 2. Bottom 2
        bottom_2_indices = sim_indices[:2]
        bottom_2 = group.loc[bottom_2_indices].copy()
        bottom_2['z_j'] = z_j[bottom_2_indices]
        
        # 3. Judges' Save Mechanism
        # Relaxed threshold to 0.5 std dev
        save_threshold = 0.5
        
        # Check diff
        p1 = bottom_2.iloc[0] # Lowest composite
        p2 = bottom_2.iloc[1] # Second lowest composite
        
        elim_sim_name = None
        
        if p1['z_j'] - p2['z_j'] > save_threshold:
            # p1 is technically better, judges save p1. p2 goes home.
            elim_sim_name = p2['celebrity_name']
            save_triggered = True
        elif p2['z_j'] - p1['z_j'] > save_threshold:
            # p2 is technically better. p2 is saved. p1 goes home.
            elim_sim_name = p1['celebrity_name']
            save_triggered = True
        else:
            # No save triggered, lowest composite (p1) goes home
            elim_sim_name = p1['celebrity_name']
            save_triggered = False
            
        # --- Metrics Update ---
        total_eliminations += 1
        
        # Regret: Did we eliminate a Top 3 dancer (by Judge Score)?
        if elim_actual_j_rank <= 3:
            regret_pre += 1
            
        # New System Regret
        elim_sim_j_rank = group_j_rank[group['celebrity_name'] == elim_sim_name].iloc[0]
        if elim_sim_j_rank <= 3:
            regret_post += 1
            
        # Survival of Unfit: Did the worst technical dancer survive?
        # If the eliminated person was NOT one of the worst dancers, then a worst dancer survived.
        if elim_name_actual not in worst_dancers:
            survival_unfit_pre += 1
            
        if elim_sim_name not in worst_dancers:
            survival_unfit_post += 1
            
        simulation_log.append({
            'season': season,
            'week': week,
            'elim_actual': elim_name_actual,
            'elim_sim': elim_sim_name,
            'elim_actual_j_rank': elim_actual_j_rank,
            'elim_sim_j_rank': elim_sim_j_rank,
            'save_triggered': save_triggered,
            'w_j': w_j,
            'w_v': w_v
        })
        
    metrics = {
        'Total Eliminations': total_eliminations,
        'Regret (Original)': regret_pre,
        'Regret (New System)': regret_post,
        'Survival of Unfit (Original)': survival_unfit_pre,
        'Survival of Unfit (New System)': survival_unfit_post
    }
    
    return pd.DataFrame(simulation_log), pd.DataFrame(weight_log), metrics

# ==========================================
# Visualization
# ==========================================
def plot_weight_evolution(weight_df, output_dir):
    # Plot average weights per season or week to show dynamic nature
    # Let's plot the distribution of weights
    
    plt.figure(figsize=(12, 6), dpi=300)
    
    # Create a long format for plotting
    w_melt = weight_df.melt(id_vars=['season', 'week'], value_vars=['w_j', 'w_v'], 
                           var_name='Weight Type', value_name='Value')
    w_melt['Weight Type'] = w_melt['Weight Type'].map({'w_j': 'Judge Weight', 'w_v': 'Fan Weight'})
    
    sns.histplot(data=w_melt, x='Value', hue='Weight Type', kde=True, bins=20, alpha=0.6, palette=['#e74c3c', '#3498db'])
    
    plt.title('Distribution of Dynamic Weights (Entropy-Based)', fontsize=16)
    plt.xlabel('Weight Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.51, plt.ylim()[1]*0.9, '50/50 Baseline', color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_entropy_weight_distribution.png"))
    print("Saved Weight Distribution Plot.")

def plot_regret_comparison(metrics, output_dir):
    labels = ['Unfair Eliminations\n(Top Talent Lost)', 'Survival of Unfit\n(Worst Dancer Saved)']
    original = [metrics['Regret (Original)'], metrics['Survival of Unfit (Original)']]
    new_sys = [metrics['Regret (New System)'], metrics['Survival of Unfit (New System)']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6), dpi=300)
    rects1 = plt.bar(x - width/2, original, width, label='Original System', color='#95a5a6', alpha=0.8)
    rects2 = plt.bar(x + width/2, new_sys, width, label='Entropy-Based Model', color='#8e44ad', alpha=0.9)
    
    plt.ylabel('Number of Occurrences', fontsize=14)
    plt.title('Fairness Improvement: Entropy Model vs Original', fontsize=18, pad=20)
    plt.xticks(x, labels, fontsize=12)
    plt.legend(fontsize=12)
    
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
    plt.savefig(os.path.join(output_dir, "4_entropy_fairness_comparison.png"))
    print("Saved Fairness Comparison Plot.")

def plot_weight_time_series(weight_df, output_dir, season=27):
    """
    Plots the evolution of weights for a specific season (e.g. Season 27 with Bobby Bones)
    """
    season_data = weight_df[weight_df['season'] == season]
    if season_data.empty:
        print(f"No data for season {season}")
        return
        
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(season_data['week'], season_data['w_j'], marker='o', label='Judge Weight', color='#e74c3c', linewidth=2.5)
    plt.plot(season_data['week'], season_data['w_v'], marker='s', label='Fan Weight', color='#3498db', linewidth=2.5, linestyle='--')
    
    plt.title(f'Dynamic Weight Evolution in Season {season}', fontsize=16)
    plt.xlabel('Week', fontsize=14)
    plt.ylabel('Weight Value', fontsize=14)
    plt.ylim(0, 1.0)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"4_weight_evolution_season_{season}.png"))
    print(f"Saved Weight Evolution Plot for Season {season}.")

def simulate_fixed_weight(df, fixed_wj):
    """
    Simulates the system with a FIXED weight for Judges (e.g. 0.5, 0.7).
    Used for Pareto Analysis to show the trade-off curve.
    """
    regret = 0
    survival_unfit = 0
    
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), group in grouped:
        eliminated_actual = group[group['results'].str.contains('Eliminated', na=False)]
        if eliminated_actual.empty:
            continue
            
        n_players = len(group)
        if n_players < 3: continue 
        
        # Standardization
        z_j = calculate_z_score(group['score'])
        z_v = calculate_z_score(group['est_fan_vote'])
        mom = group['momentum'] # Already calc in main loop, but here we assume passed df has it
        
        # Fixed Weights
        w_j = fixed_wj
        w_v = 1 - w_j
        
        # Composite Score
        alpha = 0.05
        composite_score = w_j * z_j + w_v * z_v + alpha * mom
        
        # Identify Targets
        elim_name_actual = eliminated_actual['celebrity_name'].iloc[0]
        
        min_score = group['score'].min()
        worst_dancers = group[group['score'] == min_score]['celebrity_name'].values
        
        # Simulation
        sim_indices = composite_score.sort_values(ascending=True).index
        bottom_2_indices = sim_indices[:2]
        bottom_2 = group.loc[bottom_2_indices].copy()
        bottom_2['z_j'] = z_j[bottom_2_indices]
        
        # Save Mechanism (Same as proposed system to be fair)
        save_threshold = 0.5
        p1 = bottom_2.iloc[0]
        p2 = bottom_2.iloc[1]
        
        elim_sim_name = None
        if p1['z_j'] - p2['z_j'] > save_threshold:
            elim_sim_name = p2['celebrity_name']
        elif p2['z_j'] - p1['z_j'] > save_threshold:
            elim_sim_name = p1['celebrity_name']
        else:
            elim_sim_name = p1['celebrity_name']
            
        # Metrics
        # Regret (Eliminating Top 3)
        group_j_rank = group['score'].rank(ascending=False, method='min')
        elim_sim_j_rank = group_j_rank[group['celebrity_name'] == elim_sim_name].iloc[0]
        if elim_sim_j_rank <= 3:
            regret += 1
            
        # Survival of Unfit
        if elim_sim_name not in worst_dancers:
            survival_unfit += 1
            
    return regret, survival_unfit

def perform_pareto_analysis(df, output_dir, entropy_metrics):
    """
    Runs simulation for fixed weights from 0.0 to 1.0 to generate Pareto Front.
    Plots the trade-off and overlays the Entropy Model result.
    """
    print("Running Pareto Analysis (Grid Search)...")
    
    # Ensure momentum is present
    df = df.sort_values(['season', 'celebrity_name', 'week'])
    df['prev_score'] = df.groupby(['season', 'celebrity_name'])['score'].shift(1)
    df['momentum'] = df['score'] - df['prev_score']
    df['momentum'] = df['momentum'].fillna(0)
    
    weights = np.arange(0.0, 1.05, 0.05)
    results = []
    
    for w in weights:
        regret, unfit = simulate_fixed_weight(df, w)
        results.append({
            'Judge Weight': w,
            'Regret (Fairness Loss)': regret,
            'Survival of Unfit (Merit Loss)': unfit
        })
        
    pareto_df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Plot curve
    sns.lineplot(data=pareto_df, x='Regret (Fairness Loss)', y='Survival of Unfit (Merit Loss)', 
                 marker='o', color='gray', label='Static Weight Trade-off', alpha=0.6)
    
    # Scatter points color-coded by weight
    sc = plt.scatter(pareto_df['Regret (Fairness Loss)'], pareto_df['Survival of Unfit (Merit Loss)'], 
                     c=pareto_df['Judge Weight'], cmap='viridis', s=100, zorder=3)
    plt.colorbar(sc, label='Fixed Judge Weight ($w_J$)')
    
    # Plot Entropy Model Point
    ent_regret = entropy_metrics['Regret (New System)']
    ent_unfit = entropy_metrics['Survival of Unfit (New System)']
    
    plt.scatter(ent_regret, ent_unfit, color='#e74c3c', s=250, marker='*', 
                label='Proposed Entropy Model', zorder=5, edgecolors='black')
    
    # Annotate
    plt.text(ent_regret+2, ent_unfit+2, "Proposed Model\n(Dynamic)", fontsize=12, fontweight='bold', color='#c0392b')
    
    # Ideal Point
    plt.scatter(0, 0, color='gold', s=100, marker='X', label='Ideal Utopia (0,0)', zorder=5)
    
    plt.title('Pareto Efficiency Analysis: Static vs Dynamic Weights', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', frameon=True, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_pareto_efficiency_analysis.png"))
    print("Saved Pareto Analysis Plot.")
    
    return pareto_df

def main():
    output_dir = os.path.join(base_dir, 'results', 'plots', 'question4')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading data...")
    df = load_data()
    
    print("Running Entropy Model Simulation...")
    res_df, weight_df, metrics = simulate_entropy_system(df)
    
    print("\n--- Entropy Model Results ---")
    print(f"Total Elimination Rounds: {metrics['Total Eliminations']}")
    print(f"Original System Regrets (Top Talent Eliminated): {metrics['Regret (Original)']}")
    print(f"New System Regrets: {metrics['Regret (New System)']}")
    
    print(f"Original System Survival of Unfit (Worst Dancer Saved): {metrics['Survival of Unfit (Original)']}")
    print(f"New System Survival of Unfit: {metrics['Survival of Unfit (New System)']}")
    
    improvement_regret = metrics['Regret (Original)'] - metrics['Regret (New System)']
    improvement_survival = metrics['Survival of Unfit (Original)'] - metrics['Survival of Unfit (New System)']
    
    print(f"Improvement in Fairness (Fewer Regrets): {improvement_regret}")
    print(f"Improvement in Meritocracy (Fewer Unfit Survivors): {improvement_survival}")
    
    print("\nGenerating Plots...")
    plot_regret_comparison(metrics, output_dir)
    plot_weight_evolution(weight_df, output_dir)
    plot_weight_time_series(weight_df, output_dir, season=27) 
    
    # Run Pareto Analysis
    perform_pareto_analysis(df, output_dir, metrics)
    
    # Save logs
    res_df.to_csv(os.path.join(output_dir, "entropy_simulation_results.csv"), index=False)
    weight_df.to_csv(os.path.join(output_dir, "entropy_weights_log.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    main()
