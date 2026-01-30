import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import scipy.special

# ==========================================
# Configuration & Math Formulas
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_logit_save_prob(score_a, score_b, sensitivity=0.5):
    """
    Formula 4: Logit-based Save Probability
    P(Elim_A) = S_B / (S_A + S_B)  <-- This is linear ratio, user provided this.
    
    User Formula:
    P(Eliminate i1) = S_J_i2 / (S_J_i1 + S_J_i2)
    
    Wait, the user text says:
    P(Eliminate i1) = S_J_i2 / (S_J_i1 + S_J_i2)
    This means if i2 has HIGHER score, i1 has HIGHER prob of elimination.
    This makes sense.
    """
    # Ensure scores are positive to avoid division by zero
    # Scores are usually 20-30 or 0-10.
    s_a = max(0.1, score_a)
    s_b = max(0.1, score_b)
    
    prob_elim_a = s_b / (s_a + s_b)
    return prob_elim_a

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, 'data', 'processed', 'estimated_fan_votes_results.csv')
    df = pd.read_csv(file_path)
    # Filter valid data
    df = df.dropna(subset=['score', 'est_fan_vote'])
    return df

# ==========================================
# 2. Parallel World Simulation (The Causal Engine)
# ==========================================

def simulate_rank_logic(sub_df):
    """
    Formula 1: Rank Method
    C = Rank(Judge) + Rank(Fan)
    Lower is Better.
    """
    df = sub_df.copy()
    
    # R_J: Descending Rank (Higher Score -> Rank 1)
    df['R_J'] = df['score'].rank(ascending=False, method='min')
    
    # R_V: Descending Rank (Higher Vote -> Rank 1)
    # Note: 'est_fan_vote' from Q1 might be mixed. 
    # We assume 'est_fan_vote' is PROPORTIONAL to popularity (Higher is better).
    df['R_V'] = df['est_fan_vote'].rank(ascending=False, method='min')
    
    # C_Rank
    df['C_Rank'] = df['R_J'] + df['R_V']
    
    # Elimination: Highest C_Rank (Worst)
    # Sort by C_Rank (Desc), then Fan Rank (Desc) as tie breaker
    df = df.sort_values(by=['C_Rank', 'R_V'], ascending=[False, False])
    
    # Return the elimination candidate (Top 1 after sort)
    return df.iloc[0]['celebrity_name'], df[['celebrity_name', 'C_Rank']]

def simulate_pct_logic(sub_df):
    """
    Formula 2: Percentage Method
    C = S_J / Sum(S_J) + V / Sum(V)
    Higher is Better.
    """
    df = sub_df.copy()
    
    # Judge Pct
    j_sum = df['score'].sum()
    if j_sum == 0: j_sum = 1 # Avoid div/0
    df['Pct_J'] = df['score'] / j_sum
    
    # Fan Pct
    # Assume est_fan_vote is unnormalized likelihood/vote count
    v_sum = df['est_fan_vote'].sum()
    if v_sum == 0: v_sum = 1
    df['Pct_V'] = df['est_fan_vote'] / v_sum
    
    # C_Pct
    df['C_Pct'] = df['Pct_J'] + df['Pct_V']
    
    # Elimination: Lowest C_Pct (Worst)
    # Sort by C_Pct (Asc), then Fan Pct (Asc)
    df = df.sort_values(by=['C_Pct', 'Pct_V'], ascending=[True, True])
    
    return df.iloc[0]['celebrity_name'], df[['celebrity_name', 'C_Pct']]

def simulate_save_logic(sub_df):
    """
    Formula 4: Bottom Two + Probabilistic Save
    Based on Rank Logic for Bottom Two determination (as S28+ uses Rank).
    """
    df = sub_df.copy()
    
    # 1. Determine Bottom Two using Rank Method
    df['R_J'] = df['score'].rank(ascending=False, method='min')
    df['R_V'] = df['est_fan_vote'].rank(ascending=False, method='min')
    df['C_Rank'] = df['R_J'] + df['R_V']
    
    # Sort Worst first
    df = df.sort_values(by=['C_Rank', 'R_V'], ascending=[False, False])
    
    if len(df) < 2:
        return df.iloc[0]['celebrity_name'], "N/A"
        
    # Bottom 2 candidates
    c1 = df.iloc[0] # The worst
    c2 = df.iloc[1] # The second worst
    
    # 2. Probabilistic Save
    s1 = c1['score']
    s2 = c2['score']
    
    # Prob that c1 is eliminated
    # User Formula: P(Elim_1) = S2 / (S1 + S2)
    # If S2 is very high (strong opponent), P(Elim_1) increases.
    prob_elim_c1 = s2 / (s1 + s2)
    
    # Simulation
    rand_val = np.random.random()
    if rand_val < prob_elim_c1:
        eliminated = c1['celebrity_name']
    else:
        eliminated = c2['celebrity_name']
        
    return eliminated, prob_elim_c1

# ==========================================
# 3. Main Loop
# ==========================================

def run_causal_simulation():
    df = load_data()
    results = []
    
    print("Running Parallel World Simulation...")
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), sub_df in tqdm(grouped):
        if len(sub_df) < 3: continue
        
        # 1. Rank Method Simulation
        elim_rank, _ = simulate_rank_logic(sub_df)
        
        # 2. Pct Method Simulation
        elim_pct, _ = simulate_pct_logic(sub_df)
        
        # 3. Save Method Simulation
        elim_save, prob_elim = simulate_save_logic(sub_df)
        
        # 4. Store Counterfactuals for EVERY contestant
        # We want to know: For contestant i, did they get eliminated in World A? World B?
        for idx, row in sub_df.iterrows():
            name = row['celebrity_name']
            
            is_elim_rank = 1 if name == elim_rank else 0
            is_elim_pct = 1 if name == elim_pct else 0
            is_elim_save = 1 if name == elim_save else 0
            
            results.append({
                'season': season,
                'week': week,
                'celebrity_name': name,
                'judge_score': row['score'],
                'fan_vote_est': row['est_fan_vote'],
                'industry': row['celebrity_industry'],
                'is_elim_rank': is_elim_rank,
                'is_elim_pct': is_elim_pct,
                'is_elim_save': is_elim_save,
                'save_prob_context': prob_elim # Context var
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    sim_data = run_causal_simulation()
    
    # Save for GNN Analysis
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_path = os.path.join(base_dir, 'data', 'processed', 'causal_simulation_data.csv')
    sim_data.to_csv(out_path, index=False)
    print(f"Simulation Complete. Data saved to {out_path}")
