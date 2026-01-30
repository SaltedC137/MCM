import pandas as pd
import numpy as np
import re

def preprocess_raw_data(file_path):
    """
    Loads and preprocesses the raw MCM Problem C data.
    
    Args:
        file_path (str): Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: The preprocessed data in long format with elimination weeks parsed.
    """
    # Load raw data, handle N/A strings
    # User requirement: Handle "N/A" and empty values
    df = pd.read_csv(file_path, na_values=['N/A', 'NA', ''])
    
    # Define columns to keep
    # User requirement: Judge Score 1 to 10 (and potentially others if format varies)
    # The columns are typically named weekX_judgeY_score
    score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
    # Added 'ballroom_partner' to id_vars for Gender Inference (Task 5.5)
    id_vars = ['celebrity_name', 'season', 'results', 'placement', 'celebrity_industry', 'ballroom_partner']
    
    # Pivot from Wide to Long format
    df_long = df.melt(id_vars=id_vars, value_vars=score_cols, 
                      var_name='week_judge', value_name='score')
    
    # User requirement: Convert scores to numeric, removing invalid data
    df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')
    
    # Extract numeric week number
    df_long['week'] = df_long['week_judge'].str.extract(r'week(\d+)').astype(int)
    
    # Group by week to get the AVERAGE judge score per person
    # User requirement: Arithmetic mean of valid scores
    # dropna=False is crucial to keep rows where grouping keys might be NaN
    weekly_scores = df_long.groupby(['season', 'week', 'celebrity_name', 'results', 'placement', 'celebrity_industry', 'ballroom_partner'], dropna=False)['score'].mean().reset_index()
    
    # User requirement: Treat 0 average score as NaN (to break lines in plots)
    # Also ensure existing NaNs are kept as NaNs (don't fill them with 0)
    weekly_scores.loc[weekly_scores['score'] == 0, 'score'] = np.nan
    
    # Note: For Q1 model estimation, we typically need active rows. 
    # However, to satisfy the plotting requirement of "breaking lines", 
    # we might want to return everything.
    # But q1.py expects valid scores to estimate fan votes.
    # Let's keep rows with NaN scores but mark them. 
    # The original code did: active_data = weekly_scores[weekly_scores['score'] > 0].dropna()
    # We will relax this here and let the consumer decide, 
    # BUT q1.py logic relies on active_data. 
    # To avoid breaking q1.py immediately, let's stick to returning a DataFrame 
    # that *can* be filtered.
    
    # Parse ground truth: Which week was the person actually eliminated?
    # User requirement: Regex (Eliminated|Withdrew).*Week (\d+)
    def get_elim_week(res):
        res_str = str(res)
        # Match "Eliminated" or "Withdrew" followed by "Week X"
        match = re.search(r'(?:Eliminated|Withdrew).*Week\s*(\d+)', res_str, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 99  # Represents finalists or unknown
    
    weekly_scores['elim_week'] = weekly_scores['results'].apply(get_elim_week)
    
    # User requirement: Is Withdrew
    weekly_scores['is_withdrew'] = weekly_scores['results'].str.contains('Withdrew', case=False, na=False).astype(int)
    
    # User requirement: Industry Encoding
    # 1=Athlete, 2=Performer/Model, 3=TV/Media, 4=Other
    industry_map = {
        'Athlete': 1, 'Racing Driver': 1, 'Olympian': 1,
        'Actor/Actress': 2, 'Singer/Rapper': 2, 'Comedian': 2, 'Magician': 2, 'Model': 2, 'Beauty Pagent': 2,
        'TV Personality': 3, 'News Anchor': 3, 'Radio Personality': 3, 'Reality TV Star': 3,
        'Politician': 4, 'Entrepreneur': 4
    }
    # Map and fill missing with 4 (Other)
    weekly_scores['industry_encoded'] = weekly_scores['celebrity_industry'].map(industry_map).fillna(4).astype(int)
    
    # User requirement: Core Feature Construction
    
    # Variable 2: Judge Score Share (Percentage of total score in that week)
    # Calculate total score per week
    weekly_sums = weekly_scores.groupby(['season', 'week'])['score'].transform('sum')
    weekly_scores['judge_score_share'] = weekly_scores['score'] / weekly_sums
    
    # Variable 3: Weekly Rank (Based on Judge Score)
    # Higher Score = Better Rank (1)
    # Note: This is purely based on judge scores, irrespective of the season's elimination rules
    weekly_scores['weekly_rank'] = weekly_scores.groupby(['season', 'week'])['score'].rank(ascending=False, method='min')
    
    # User Requirement: Gender Inference (Optional)
    # Since we don't have a name database, we just initialize the column.
    # Users can fill this later using partner names if needed.
    weekly_scores['gender'] = np.nan
    
    return weekly_scores
