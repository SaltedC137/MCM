import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Add project root to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from utils.data_preprocessing import preprocess_raw_data

# ==========================================
# 2. Inverse Optimization Engine (Monte Carlo)
# ==========================================
def estimate_fan_votes(group, iterations=5000):
    """
    Estimates fan votes for ALL contestants in a given week based on 
    the observed elimination or final placement.
    """
    season = group['season'].iloc[0]
    week = group['week'].iloc[0]
    n_players = len(group)
    player_names = group['celebrity_name'].values
    judge_scores = group['score'].values
    
    # Identify eliminated contestants for the specific week
    eliminated = group[group['elim_week'] == week]
    
    # Identify if it is a Final Week (usually top 3 or top 4 remain)
    is_final_week = (group['placement'] <= 3).any() and week >= group['elim_week'].max() - 2
    
    # If no one is eliminated and it's not the final, constraints are too weak to estimate
    if eliminated.empty and not is_final_week:
        return None 

    valid_simulations = []
    # Rank system: S1-S2 and S28+ (Assumed S28 based on problem description)
    is_rank_system = (season <= 2) or (season >= 28)

    if not is_rank_system:
        # --- Percentage-based System (Seasons 3-27) ---
        judge_pcts = judge_scores / (np.sum(judge_scores) + 1e-6)
        for _ in range(iterations):
            # Generate random fan vote shares (sum to 1.0)
            fan_pcts = np.random.dirichlet([1.0] * n_players)
            total_scores = judge_pcts + fan_pcts
            
            if not is_final_week:
                # Rule: Eliminated contestant has the LOWEST total score
                # Tie-breaker: Lower Fan Vote (Fan %) is worse.
                # Sort: Primary Key = Total Score (Asc), Secondary Key = Fan % (Asc)
                # The first item in sorted list is the eliminated one.
                
                # Create a structured array for sorting
                dtype = [('total', float), ('fan', float)]
                values = np.array(list(zip(total_scores, fan_pcts)), dtype=dtype)
                sorted_indices = np.argsort(values, order=('total', 'fan'))
                
                worst_player_idx = sorted_indices[0]
                
                if player_names[worst_player_idx] in eliminated['celebrity_name'].values:
                    valid_simulations.append(fan_pcts)
            else:
                # Rule: Total scores match final placement (Higher Score = Better Rank)
                # Note: In Percentage system, Higher % is Better.
                # So Rank 1 (Winner) should have Highest %.
                # pd.rank(ascending=False) gives Rank 1 to Highest.
                
                pred_placement = pd.Series(total_scores).rank(ascending=False, method='min').values
                actual_placement = group['placement'].values
                if np.array_equal(pred_placement, actual_placement):
                    valid_simulations.append(fan_pcts)
    else:
        # --- Rank-based System (Seasons 1-2, 28-34) ---
        # Rank 1 = Best, Rank N = Worst
        # Score = Judge Rank + Fan Rank.
        # Higher Score = Worse.
        judge_ranks = group['score'].rank(ascending=False, method='min').values
        
        for _ in range(iterations):
            # Generate random fan ranks (permutation of 1 to N)
            fan_ranks = np.random.permutation(np.arange(1, n_players + 1))
            rank_sums = judge_ranks + fan_ranks 
            
            if not is_final_week:
                # Sort criteria for elimination:
                # Primary: Rank Sum (Descending) -> Higher is worse
                # Secondary: Fan Rank (Descending) -> Higher (worse fan vote) is worse
                
                dtype = [('sum', float), ('fan', float)]
                values = np.array(list(zip(rank_sums, fan_ranks)), dtype=dtype)
                # argsort sorts Ascending. We want Descending.
                # So we look at the END of the sorted array for the "worst"
                sorted_indices = np.argsort(values, order=('sum', 'fan'))
                
                if season < 28:
                    # S1-S2: The single worst person is eliminated
                    # The last element in sorted_indices is the one with Max Sum (and Max Fan Rank if tied)
                    worst_idx = sorted_indices[-1]
                    if player_names[worst_idx] in eliminated['celebrity_name'].values:
                        valid_simulations.append(fan_ranks)
                else:
                    # S28+: Bottom Two are identified. Judges pick one.
                    # The last 2 elements are the Bottom Two.
                    bottom_two_indices = sorted_indices[-2:]
                    # The eliminated person MUST be in the bottom two
                    if eliminated['celebrity_name'].iloc[0] in player_names[bottom_two_indices]:
                        valid_simulations.append(fan_ranks)
            else:
                # Final Week: Smallest Rank Sum = 1st Place
                pred_placement = pd.Series(rank_sums).rank(method='min').values
                actual_placement = group['placement'].values
                if np.array_equal(pred_placement, actual_placement):
                    valid_simulations.append(fan_ranks)

    if not valid_simulations:
        return None

    # Calculate statistics from valid samples
    sim_array = np.array(valid_simulations)
    mean_estimates = np.mean(sim_array, axis=0) 
    std_votes = np.std(sim_array, axis=0)
    
    # Certainty Metric
    certainty = 1 - (std_votes / (mean_estimates + 0.1))
    certainty = np.clip(certainty, 0, 1)

    return pd.DataFrame({
        'celebrity_name': player_names,
        'season': [season] * n_players,
        'week': [week] * n_players,
        'est_fan_vote': mean_estimates,
        'certainty': certainty
    })

# ==========================================
# 3. Main Execution and Validation
# ==========================================
def main():
    # Use relative paths from the script location
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    output_file = os.path.join(base_dir, 'data', 'processed', 'estimated_fan_votes_results.csv')

    print("Task 1/3: Preprocessing Data...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    data = preprocess_raw_data(input_file)
    
    # Create a subset of active data for processing (ignore rows with NaN scores)
    # This ensures the Monte Carlo simulation only runs on valid contestants for that week
    active_processing_data = data.dropna(subset=['score']).copy()
    
    print("Task 2/3: Estimating Fan Votes for all contestants (this may take a minute)...")
    results_list = []
    grouped = active_processing_data.groupby(['season', 'week'])
    
    for _, group in tqdm(grouped):
        est = estimate_fan_votes(group)
        if est is not None:
            results_list.append(est)
    
    if not results_list:
        print("No valid estimates found.")
        return

    final_estimates = pd.concat(results_list)
    
    # Merge results back to the original active dataframe
    merged_df = pd.merge(data, final_estimates, on=['celebrity_name', 'season', 'week'], how='left')
    
    print("\nTask 3/3: Evaluating Model Consistency...")
    eval_df = merged_df.dropna(subset=['est_fan_vote'])
    
    # Validation logic mirroring the simulation logic
    total_checks = 0
    matches = 0
    
    for (season, week), group in eval_df.groupby(['season', 'week']):
        elim_actual = group[group['elim_week'] == week]
        if elim_actual.empty:
            continue
            
        total_checks += 1
        is_match = False
        
        if (season <= 2) or (season >= 28):
            # Rank Logic
            j_rank = group['score'].rank(ascending=False, method='min').values
            # For validation, we use the MEAN estimated fan vote to compute implied rank
            # Note: Using mean fan vote to reconstruct 'rank' is tricky because mean is float.
            # We approximate by ranking the mean estimates. 
            # Lower Mean Estimate (e.g. 1.5) = Better Rank (1 or 2)
            f_rank = pd.Series(group['est_fan_vote']).rank(ascending=True, method='min').values
            
            rank_sums = j_rank + f_rank
            
            # Sort desc
            dtype = [('sum', float), ('fan', float)]
            values = np.array(list(zip(rank_sums, f_rank)), dtype=dtype)
            sorted_indices = np.argsort(values, order=('sum', 'fan'))
            
            player_names = group['celebrity_name'].values
            
            if season < 28:
                worst_idx = sorted_indices[-1]
                if player_names[worst_idx] in elim_actual['celebrity_name'].values:
                    is_match = True
            else:
                bottom_two_indices = sorted_indices[-2:]
                if elim_actual['celebrity_name'].iloc[0] in player_names[bottom_two_indices]:
                    is_match = True
        else:
            # Percent Logic
            # Higher % is Better. Lower % is Worse.
            j_pct = group['score'] / group['score'].sum()
            # Mean estimated fan pct
            f_pct = group['est_fan_vote']
            total = j_pct + f_pct
            
            # Sort asc (Lowest is eliminated)
            dtype = [('total', float), ('fan', float)]
            values = np.array(list(zip(total, f_pct)), dtype=dtype)
            sorted_indices = np.argsort(values, order=('total', 'fan'))
            
            worst_idx = sorted_indices[0]
            player_names = group['celebrity_name'].values
            if player_names[worst_idx] in elim_actual['celebrity_name'].values:
                is_match = True
                
        if is_match:
            matches += 1

    print(f"Estimation complete.")
    print(f"Contestant-Weeks Processed: {len(eval_df)}")
    print(f"Average System Certainty: {eval_df['certainty'].mean():.4f}")
    if total_checks > 0:
        print(f"Consistency Match Rate (Approx): {matches/total_checks*100:.2f}%")
    
    # Save the final dataset
    merged_df.to_csv(output_file, index=False)
    print(f"\nSUCCESS: Results saved to '{output_file}'")

if __name__ == "__main__":
    main()
