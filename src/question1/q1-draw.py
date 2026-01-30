import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "plots", "question1")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.6) # Further increase font size


PALETTE = ["#34495e", "#e67e22", "#27ae60", "#c0392b", "#2980b9"]

def load_and_normalize(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    # Strategy A: Keep NaNs for visualization to allow line breaks
    # df = df.dropna(subset=['est_fan_vote']) 

    processed_frames = []
    
    # Iterate through each week of each season for normalization
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), sub_df in grouped:
        sub_df = sub_df.copy()
        
        # --- Judge Score Normalization ---
        s_min, s_max = sub_df['score'].min(), sub_df['score'].max()
        # Visual Correction: Map to 0.05 - 1.0 to prevent lowest score (0.0) from disappearing on the plot
        if s_max == s_min:
            sub_df['j_strength'] = 0.5 
        else:
            raw_norm = (sub_df['score'] - s_min) / (s_max - s_min)
            sub_df['j_strength'] = 0.05 + 0.95 * raw_norm
        
        # --- Fan Vote Normalization ---
        v = sub_df['est_fan_vote']
        v_min, v_max = v.min(), v.max()
        
        raw_fan_norm = 0
        if v_max == v_min:
            raw_fan_norm = 0.5
        else:
            if season <= 2 or season >= 28:
                # Rank System: Smaller value is better -> Invert
                raw_fan_norm = (v_max - v) / (v_max - v_min)
            else:
                # Percentage System: Larger value is better
                raw_fan_norm = (v - v_min) / (v_max - v_min)
        
        # Apply visual correction as well, minimum is 0.05
        if isinstance(raw_fan_norm, float):
            sub_df['f_strength'] = 0.05 + 0.95 * raw_fan_norm
        else:
            sub_df['f_strength'] = 0.05 + 0.95 * raw_fan_norm

        processed_frames.append(sub_df)

    df_normalized = pd.concat(processed_frames, ignore_index=True)
    return df_normalized

def plot_correlation(df):
    """Plot 1: Optimized Version - Add Jitter to prevent data overlap"""
    plt.figure(figsize=(12, 9)) # Increase canvas size
    

    sns.kdeplot(data=df, x='j_strength', y='f_strength', 
                fill=True, cmap="Blues", alpha=0.2, levels=10, thresh=0.05)
    
    sns.regplot(data=df, x='j_strength', y='f_strength', 
                x_jitter=0.03, y_jitter=0.03, 
                scatter_kws={'s': 25, 'alpha': 0.4, 'color': PALETTE[0], 'edgecolor': 'w'}, 
                line_kws={'color': PALETTE[3], 'linewidth': 3, 'label': 'Trend Line'})
    
    plt.title('Correlation Analysis: Technical Merit vs. Fan Support', fontsize=18, pad=20)
    plt.xlabel('Normalized Judge Score (Low → High)', fontsize=15)
    plt.ylabel('Normalized Fan Support (Low → High)', fontsize=15)
    
    # Limit axis range, leave some margin
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper left', fontsize=12,
            frameon=True,           
            facecolor='white',      
            edgecolor='#b0b0b0',    
            framealpha=0.9,         
            fancybox=True,          
            shadow=True,            
            borderpad=0.8)         
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_judge_fan_correlation_v2.png", dpi=300)
    print("Plot 1 (Jittered) saved.")

def plot_industry_performance_heatmap(df):
    """
    Plot 2 (New): Industry-Performance Quadrant Heatmap
    Uses a 'Confusion Matrix' style approach to classify contestant performance.
    Categories:
    1. Powerhouse (High Judge, High Fan)
    2. Fan Favorite (Low Judge, High Fan)
    3. Underrated (High Judge, Low Fan)
    4. Early Exit (Low Judge, Low Fan)
    """
    plt.figure(figsize=(12, 8))
    
    # 1. Filter Data & Define Quadrants
    top_inds = df['celebrity_industry'].value_counts().nlargest(6).index
    df_sub = df[df['celebrity_industry'].isin(top_inds)].copy()
    
    # Define thresholds (Using median as split point, more robust)
    j_median = df_sub['j_strength'].median()
    f_median = df_sub['f_strength'].median()
    
    def classify(row):
        high_j = row['j_strength'] >= j_median
        high_f = row['f_strength'] >= f_median
        
        if high_j and high_f: return "Powerhouse\n(High J, High F)"
        if not high_j and high_f: return "Fan Favorite\n(Low J, High F)"
        if high_j and not high_f: return "Underrated\n(High J, Low F)"
        return "Early Exit\n(Low J, Low F)"

    df_sub['category'] = df_sub.apply(classify, axis=1)
    
    # 2. Build Matrix (Crosstab) -> Row Normalization (Calculate Percentage)
    matrix = pd.crosstab(df_sub['celebrity_industry'], df_sub['category'], normalize='index')
    
    # Reorder columns to follow logical order
    col_order = [
        "Powerhouse\n(High J, High F)", 
        "Fan Favorite\n(Low J, High F)",
        "Underrated\n(High J, Low F)", 
        "Early Exit\n(Low J, Low F)"
    ]
    matrix = matrix[col_order]
    
    # Sort industries by "Powerhouse" ratio
    matrix = matrix.sort_values("Powerhouse\n(High J, High F)", ascending=False)
    
    # 3. Plot Heatmap
    sns.heatmap(matrix, annot=True, fmt=".1%", cmap="YlGnBu", 
                linewidths=1, linecolor='white', cbar_kws={'label': 'Percentage of Weeks'})
    
    plt.title('Industry Performance Archetypes: Judge vs. Fan Preference', fontsize=16)
    plt.xlabel('Performance Archetype', fontsize=14)
    plt.ylabel('Industry', fontsize=14)
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_industry_quadrant_heatmap.png", dpi=300)
    print("Plot 2 (Quadrant Heatmap) saved.")

def plot_certainty_timeline(df):
    """Plot 3: Certainty"""
    plt.figure(figsize=(14, 6))
    
    avg_cert = df.groupby('season')['certainty'].mean()
    std_cert = df.groupby('season')['certainty'].std()
    
    x = avg_cert.index
    y = avg_cert.values
    
    plt.plot(x, y, color=PALETTE[0], linewidth=2.5, marker='o', markersize=6, label='Mean Certainty')
    plt.fill_between(x, y - std_cert, y + std_cert, color=PALETTE[0], alpha=0.15, label='Std Dev')
    
    # Format Change Line
    plt.axvline(x=28, color=PALETTE[3], linestyle='--', linewidth=2.5, label='Format Change (S28)')
    
    plt.title('Model Estimation Certainty (Season 1-34)', fontsize=18)
    plt.xlabel('Season', fontsize=15)
    plt.ylabel('Certainty Score', fontsize=15)
    plt.legend(loc='lower left', frameon=True, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_certainty_timeline.png", dpi=300)
    print("Plot 3 saved.")

def plot_archetypes_improved(df):
    """
    Plot 4: Case Analysis (Logic Improved)
    1. Filter out contestants who survived < 4 weeks.
    2. Sort by Gap (Fan - Judge).
    """
    # 1. Filter Step: Calculate only for contestants who participated in at least 4 weeks
    # IMPORTANT: Filter only rows with valid fan estimates (ignore NaN rows from elimination/withdrawn)
    # Exclude withdrawn contestants from statistical analysis
    # We can identify withdrawn contestants if needed, but the main thing is they won't have valid estimates after withdrawal.
    # However, if someone withdrew early, they shouldn't be in the "Archetype" analysis which looks for long-term trends.
    
    # First, let's identify withdrawn contestants
    if 'results' in df.columns:
        withdrawn_names = df[df['results'].str.contains('Withdrew', na=False, case=False)]['celebrity_name'].unique()
    else:
        withdrawn_names = []

    # Calculate counts based on VALID fan estimates
    counts = df.dropna(subset=['est_fan_vote']).groupby(['season', 'celebrity_name']).size()
    
    # Threshold: Must have at least 8 weeks of VALID data to be considered for long-term bias analysis
    valid_contestants = counts[counts >= 8].index 
    
    # Filter Data
    df_filtered = df.set_index(['season', 'celebrity_name']).loc[valid_contestants].reset_index()
    
    # Exclude withdrawn contestants explicitly if you want to be stricter
    # df_filtered = df_filtered[~df_filtered['celebrity_name'].isin(withdrawn_names)]
    
    # Also ensure we only aggregate valid rows for the stats
    df_filtered = df_filtered.dropna(subset=['est_fan_vote'])

    # 2. Calculate Average Performance
    celeb_stats = df_filtered.groupby(['celebrity_name', 'season']).agg({
        'j_strength': 'mean',
        'f_strength': 'mean'
    }).reset_index()
    
    # Gap > 0: Higher Fan Score (Popularity Bias)
    # Gap < 0: Higher Judge Score (Skill Bias)
    celeb_stats['gap'] = celeb_stats['f_strength'] - celeb_stats['j_strength']

    # 3. Intelligent Case Selection
    # Must include Bobby Bones as mentioned in the problem
    bobby = celeb_stats[celeb_stats['celebrity_name'].str.contains("Bobby Bones")]
    
    # Select largest positive Gap (Fan Favorites)
    top_gap = celeb_stats.nlargest(2, 'gap')
    
    # Select smallest negative Gap (Judge Favorites)
    bottom_gap = celeb_stats.nsmallest(2, 'gap')
    
    # Select Strongest (Judge Strength Max)
    top_skill = celeb_stats.nlargest(1, 'j_strength')
    
    # Merge and Drop Duplicates
    cases = pd.concat([bobby, top_gap, bottom_gap, top_skill]).drop_duplicates()
    
    # 4. Key: Sort by Gap to form a storyline
    # Sort gap ascending (negative at bottom, positive at top) -> Index 0 is at bottom when plotting barh
    cases = cases.sort_values('gap', ascending=True)

    # 5. Plotting
    plt.figure(figsize=(12, 8))
    
    y_pos = np.arange(len(cases))
    height = 0.35
    
    plt.barh(y_pos + height/2, cases['j_strength'], height, label='Judge Score (Technical)', color=PALETTE[0], alpha=0.9)
    plt.barh(y_pos - height/2, cases['f_strength'], height, label='Fan Vote (Popularity)', color=PALETTE[1], alpha=0.9)
    
    # Generate labels with descriptions
    labels = []
    for _, row in cases.iterrows():
        name = row['celebrity_name']
        season = row['season']
        gap = row['gap']
        
        if gap > 0.15: tag = "\n(Popularity Bias)"
        elif gap < -0.15: tag = "\n(Skill Bias)"
        else: tag = "" # Balanced
        
        labels.append(f"{name} (S{season}){tag}")

    plt.yticks(y_pos, labels, fontsize=12)
    plt.xlabel('Normalized Strength Index', fontsize=14)
    plt.title('Divergence Analysis: Technical Merit vs. Popularity', fontsize=18)
    plt.legend(loc='lower right', fontsize=12, frameon=True)
    plt.xlim(0, 1.15) # Leave space for legend
    plt.grid(axis='x', alpha=0.3)
    
    # Add a dashed line in the middle to represent balance point
    # plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_archetype_logic_fixed.png", dpi=300)
    print("Plot 4 (Logic Fixed) saved.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, 'data', 'processed', 'estimated_fan_votes_results.csv')
    if os.path.exists(file_path):
        df = load_and_normalize(file_path)
        
        plot_correlation(df)
        plot_industry_performance_heatmap(df)
        plot_certainty_timeline(df)
        plot_archetypes_improved(df)
        
        print(f"\nOptimization Complete. Check '{output_dir}' for improved plots.")
    else:
        print("Error: csv file not found.")
