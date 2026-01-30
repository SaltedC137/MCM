import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# Configuration: Academic / High-End Aesthetic
# ==========================================
# Using a cleaner, more professional style suitable for LaTeX/Academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.color'] = '#dddddd'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.alpha'] = 0.8

# Professional Color Palette (Colorblind friendly + Academic)
# Palette: "Muted" from Seaborn or custom
COLOR_RANK = "#4A90E2"      # Soft Blue
COLOR_PCT = "#E74C3C"       # Soft Red
COLOR_SAVE = "#2ECC71"      # Soft Green
COLOR_NEUTRAL = "#95A5A6"
BG_COLOR = "#FFFFFF"        # Pure White for paper

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sim_path = os.path.join(base_dir, 'data', 'processed', 'causal_simulation_data.csv')
    bias_path = os.path.join(base_dir, 'data', 'processed', 'method_bias_analysis.csv')
    return pd.read_csv(sim_path), pd.read_csv(bias_path)

def plot_bias_comparison(bias_df):
    """
    Plot 1: Bias Score Comparison Bar Chart
    Refined: Thinner bars, error bars style (implied), clear threshold.
    """
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Sort
    bias_df = bias_df.sort_values('Fan_Bias_Score', ascending=False)
    
    # Map method names to display names
    name_map = {
        'Percentage': 'Percentage Method\n(Winner-Takes-All)',
        'Rank': 'Rank Method\n(Linear)',
        'Save': 'Judges\' Save\n(Hybrid)'
    }
    bias_df['Display_Name'] = bias_df['Method'].map(name_map).fillna(bias_df['Method'])

    # Colors
    colors = [COLOR_PCT, COLOR_RANK, COLOR_SAVE]
    
    # Bar Chart
    bars = ax.bar(bias_df['Display_Name'], bias_df['Fan_Bias_Score'], color=colors, alpha=0.85, width=0.5, zorder=3)
    
    # Add value labels with background box for readability
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')
        
    # Threshold Line (0.5 = Balanced)
    ax.axhline(y=0.5, color='#555555', linestyle='--', linewidth=1.5, zorder=2)
    ax.text(2.6, 0.51, "Balanced Zone (0.5)", color='#555555', fontsize=9, style='italic', ha='right', va='bottom')

    # Styling
    ax.set_title('Fan Bias Quantification (GNN-Derived)', fontsize=14, fontweight='bold', pad=15, color='#222222')
    ax.set_ylabel('Bias Score (0=Judge, 1=Fan)', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', labelsize=10)
    
    # Remove grid on x
    ax.grid(False, axis='x')
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "plots", "question2")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/1_bias_score_comparison.png", dpi=300, bbox_inches='tight')
    print("Plot 1 (Bias Comparison) saved.")

def plot_controversy_heatmap(sim_df):
    """
    Plot 2: Controversy Heatmap
    Refined: Better aspect ratio, clearer labels, discrete color map for risk levels.
    """
    targets = ['Jerry Rice', 'Billy Ray Cyrus', 'Bobby Bones', 'Bristol Palin']
    target_data = sim_df[sim_df['celebrity_name'].isin(targets)].copy()
    
    if target_data.empty: return

    risk_summary = target_data.groupby('celebrity_name')[['is_elim_rank', 'is_elim_pct', 'is_elim_save']].mean()
    
    # Rename columns
    risk_summary.columns = ['Rank System', 'Percentage System', 'Judges\' Save']
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Custom Diverging Colormap (White -> Yellow -> Red)
    cmap = sns.light_palette("#C0392B", as_cmap=True)
    
    sns.heatmap(risk_summary, annot=True, fmt=".1%", cmap=cmap, 
                linewidths=1.5, linecolor='white', cbar_kws={'label': 'Elimination Risk'},
                square=True, ax=ax)
    
    ax.set_title('Controversy Stress Test: Elimination Risk', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.xticks(rotation=0)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "plots", "question2")
    plt.savefig(f"{output_dir}/2_controversy_risk_heatmap.png", dpi=300, bbox_inches='tight')
    print("Plot 2 (Controversy Heatmap) saved.")

def plot_parallel_world_timeline(sim_df):
    """
    Plot 3: Bobby Bones Timeline
    Refined: Gantt-chart style or categorical strip plot.
    """
    bobby = sim_df[(sim_df['season'] == 27) & (sim_df['celebrity_name'] == 'Bobby Bones')].sort_values('week')
    
    if bobby.empty: return
        
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    weeks = bobby['week'].values
    heatmap_data = np.array([
        bobby['is_elim_pct'].values,   # Actual
        bobby['is_elim_rank'].values,  # Rank
        bobby['is_elim_save'].values   # Save
    ])
    
    # Custom cmap: Green (Safe), Red (Eliminated)
    # Using specific hex codes for "Good" and "Bad"
    cmap = LinearSegmentedColormap.from_list("Safety", ["#2ECC71", "#E74C3C"], N=2)
    
    sns.heatmap(heatmap_data, cmap=cmap, cbar=False, linewidths=2, linecolor='white',
                ax=ax, annot=True, fmt="d", annot_kws={"weight": "bold", "size": 10, "color": "white"})
    
    # Labels
    ax.set_xticks(np.arange(len(weeks)) + 0.5)
    ax.set_xticklabels([f"Week {w}" for w in weeks], fontsize=10)
    
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['Percentage (Actual)', 'Rank System', 'Judges\' Save'], fontsize=11, rotation=0)
    
    ax.set_title('Bobby Bones (S27): Counterfactual Timeline', fontsize=14, fontweight='bold', pad=15, color='#222222')
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ECC71', label='Safe', edgecolor='white'),
                       Patch(facecolor='#E74C3C', label='Eliminated', edgecolor='white')]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=10)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "plots", "question2")
    plt.savefig(f"{output_dir}/3_bobby_bones_timeline.png", dpi=300, bbox_inches='tight')
    print("Plot 3 (Bobby Bones Timeline) saved.")

if __name__ == "__main__":
    sim_df, bias_df = load_data()
    
    plot_bias_comparison(bias_df)
    plot_controversy_heatmap(sim_df)
    plot_parallel_world_timeline(sim_df)
