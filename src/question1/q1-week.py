import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def check_optimal_weeks(file_path):
    print("Analyzing optimal screening weeks...")
    
    # 1. Load data
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return
    
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['est_fan_vote', 'certainty'])
    
    # 2. Calculate "Total Weeks" for each contestant
    # Group by season and name to count occurrences
    contestant_stats = df.groupby(['season', 'celebrity_name']).agg({
        'week': 'count',           # Weeks participated
        'certainty': 'mean'        # Average model certainty for this contestant
    }).reset_index()
    
    contestant_stats.rename(columns={'week': 'total_weeks_survived'}, inplace=True)
    
    # 3. Group by "Total Weeks" to see how average certainty changes
    week_analysis = contestant_stats.groupby('total_weeks_survived')['certainty'].agg(['mean', 'std', 'count']).reset_index()
    
    # Filter out rare cases (e.g., someone skipping to 15 weeks, statistically insignificant)
    week_analysis = week_analysis[week_analysis['count'] >= 5]
    
    print("\n--- Data Analysis Results ---")
    print(week_analysis.to_string(index=False))
    
    # 4. Plotting: Elbow Method
    plt.figure(figsize=(10, 6))
    
    # Plot main line
    plt.plot(week_analysis['total_weeks_survived'], week_analysis['mean'], 
             marker='o', linestyle='-', linewidth=3, color='#2980b9', label='Average Model Certainty')
    
    # Plot error band (standard deviation)
    plt.fill_between(week_analysis['total_weeks_survived'], 
                     week_analysis['mean'] - week_analysis['std'], 
                     week_analysis['mean'] + week_analysis['std'], 
                     color='#2980b9', alpha=0.15)
    
    # Mark current choice "4 Weeks"
    plt.axvline(x=4, color='#c0392b', linestyle='--', linewidth=2, label='Current Choice (4 Weeks)')
    
    plt.title('Determining Optimal Threshold: The "Elbow Method"', fontsize=15)
    plt.xlabel('Total Weeks Contestant Participated', fontsize=12)
    plt.ylabel('Average Estimation Certainty (0-1)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(min(week_analysis['total_weeks_survived']), 13, 1))
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "plots", "question1")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_img = os.path.join(output_dir, "week_threshold_check.png")
    plt.savefig(output_img, dpi=300)
    print(f"\nAnalysis plot saved to: {output_img}")
    print("Please observe the plot: The point where the curve slope flattens (elbow point) is the optimal threshold.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, 'data', 'processed', 'estimated_fan_votes_results.csv')
    check_optimal_weeks(file_path)
