import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_results():
    # Load the results
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, 'data', 'processed', 'estimated_fan_votes_results.csv')
    df = pd.read_csv(file_path)
    
    print("=== Data Overview ===")
    print(f"Total Rows: {len(df)}")
    print(f"Seasons Covered: {df['season'].unique()}")
    print(f"Average Certainty: {df['certainty'].mean():.4f}")
    
    # 1. Analyze Controversial Figures
    print("\n=== Controversial Figures Analysis ===")
    controversial = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    for name, season in controversial:
        subset = df[(df['celebrity_name'] == name) & (df['season'] == season)].sort_values('week')
        if subset.empty:
            print(f"No data for {name} in Season {season}")
            continue
            
        print(f"\n{name} (Season {season}):")
        print(subset[['week', 'score', 'est_fan_vote', 'certainty']].to_string(index=False))
        
        # Calculate average fan vote rank/percent relative to others in that week
        avg_fan_performance = []
        for w in subset['week']:
            week_data = df[(df['season'] == season) & (df['week'] == w)]
            # Check if rank or percent system
            if season <= 2 or season >= 28: # Rank
                # In Rank system, Lower est_fan_vote is Better (1=Best)
                # But wait, our model outputs MEAN rank.
                my_score = subset[subset['week'] == w]['est_fan_vote'].values[0]
                rank_in_week = (week_data['est_fan_vote'] < my_score).sum() + 1
                total_in_week = len(week_data)
                avg_fan_performance.append(f"{rank_in_week}/{total_in_week}")
            else: # Percent
                # In Percent system, Higher est_fan_vote is Better
                my_score = subset[subset['week'] == w]['est_fan_vote'].values[0]
                rank_in_week = (week_data['est_fan_vote'] > my_score).sum() + 1
                total_in_week = len(week_data)
                avg_fan_performance.append(f"{rank_in_week}/{total_in_week}")
        
        print(f"Fan Vote Rankings per Week: {avg_fan_performance}")

    # 2. Analyze Certainty Distribution
    print("\n=== Certainty Analysis ===")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['certainty'], bins=30, kde=True)
    plt.title('Distribution of Model Certainty')
    plt.xlabel('Certainty Score (0-1)')
    output_img = os.path.join(base_dir, 'results', 'plots', 'question1', 'certainty_distribution.png')
    plt.savefig(output_img)
    print(f"Certainty distribution plot saved to '{output_img}'")
    
    # Check low certainty cases
    low_certainty = df[df['certainty'] < 0.3]
    if not low_certainty.empty:
        print(f"\nFound {len(low_certainty)} rows with low certainty (< 0.3).")
        print("Sample of low certainty cases:")
        print(low_certainty.head())
    else:
        print("\nNo extremely low certainty cases found.")

if __name__ == "__main__":
    analyze_results()
