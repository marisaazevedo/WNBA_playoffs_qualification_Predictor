import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder



# Define regular season score calculation
def calculate_regular_season_score(row, weights):
    MAX = 383.3
    score = 0
    if 'G' in row['pos'] and 'F' in row['pos']:  # G-F or F-G
        score = (
            0.15 * row['points'] * weights['points'] +
            0.275 * row['assists'] * weights['assists'] +
            0.1 * row['steals'] * weights['steals'] +
            0.175 * row['threeMade'] * weights['threeMade'] +
            0.15 * row['rebounds'] * weights['rebounds'] +
            0.075 * row['blocks'] * weights['blocks'] -
            0.1 * row['turnovers'] * weights['turnovers'] +
            row['awards'] * weights['awards']
        )
    elif 'F' in row['pos'] and 'C' in row['pos']:  # F-C or C-F
        score = (
            0.35 * row['rebounds'] * weights['rebounds'] +
            0.35 * row['points'] * weights['points'] +
            0.15 * row['blocks'] * weights['blocks'] +
            0.075 * row['assists'] * weights['assists'] -
            0.05 * row['turnovers'] * weights['turnovers'] +
            row['awards'] * weights['awards']
        )
    elif 'G' in row['pos']:  # Pure Guard
        score = (
            0.3 * row['points'] * weights['points'] +
            0.25 * row['assists'] * weights['assists'] +
            0.2 * row['steals'] * weights['steals'] +
            0.15 * row['threeMade'] * weights['threeMade'] -
            0.1 * row['turnovers'] * weights['turnovers'] +
            row['awards'] * weights['awards']
        )
    elif 'F' in row['pos']:  # Pure Forward
        score = (
            0.35 * row['points'] * weights['points'] +
            0.3 * row['rebounds'] * weights['rebounds'] +
            0.2 * row['assists'] * weights['assists'] +
            0.1 * row['blocks'] * weights['blocks'] -
            0.05 * row['turnovers'] * weights['turnovers'] +
            row['awards'] * weights['awards']
        )
    elif 'C' in row['pos']:  # Pure Center
        score = (
            0.4 * row['rebounds'] * weights['rebounds'] +
            0.35 * row['points'] * weights['points'] +
            0.2 * row['blocks'] * weights['blocks'] -
            0.05 * row['turnovers'] * weights['turnovers'] +
            row['awards'] * weights['awards']
        )

    # Normalize between 0 and 1
    score = score / MAX


    return score


# Define postseason score calculation
def calculate_postseason_score(row, weights):
    max = 101.9
    
    if 'G' in row['pos'] and 'F' in row['pos']:  # G-F or F-G
        score = ((0.15 * row['PostPoints'] * weights['points'] +
                + 0.275 * row['PostAssists'] * weights['assists'] + 
                0.1 * row['PostSteals'] * weights['steals'] +
                0.175 * row['PostthreeMade'] * weights['threeMade'] +
                0.15 * row['PostRebounds'] * weights['rebounds'] +
                0.075 * row['PostBlocks'] * weights['blocks'] -
                0.1 * row['PostTurnovers'] * weights['turnovers']))
    elif 'F' in row['pos'] and 'C' in row['pos']:  # F-C or C-F
        score = ((0.35 * row['PostRebounds'] * weights['rebounds'] +
                0.35 * row['PostPoints'] * weights['points'] +
                0.15 * row['PostBlocks'] * weights['blocks'] +
                0.075 * row['PostAssists'] * weights['assists'] -
                0.05 * row['PostTurnovers'] * weights['turnovers']))
    elif 'G' in row['pos']:  # Pure Guard
        score = ((0.3 * row['PostPoints'] * weights['points'] +
                0.25 * row['PostAssists'] * weights['assists'] + 
                0.2 * row['PostSteals'] * weights['steals'] +
                0.15 * row['PostthreeMade'] * weights['threeMade'] -
                0.1 * row['PostTurnovers'] * weights['turnovers']))
    elif 'F' in row['pos']:  # Pure Forward
        score = ((0.35 * row['PostPoints'] * weights['points'] +
                0.3 * row['PostRebounds'] * weights['rebounds'] +
                0.2 * row['PostAssists'] * weights['assists'] +
                0.1 * row['PostBlocks'] * weights['blocks'] -
                0.05 * row['PostTurnovers'] * weights['turnovers']))
    elif 'C' in row['pos']:  # Pure Center
        score = ((0.4 * row['PostRebounds'] * weights['rebounds'] +
                0.35 * row['PostPoints'] * weights['points'] +
                0.2 * row['PostBlocks'] * weights['blocks'] -
                0.05 * row['PostTurnovers'] * weights['turnovers']))
    return score/max


def calculate_team_score(row, df):
    '''
    Calculate the team score based on the RegularScore of the players.
    Different weights are applied based on the stint value.
    '''
    # Create a copy and apply weights based on the stint values
    df_copy = df.copy()
    df_copy['weight'] = df_copy['stint'].map({0: 1, 1: 0.5, 2: 0.5})
    
    # Filter the DataFrame for the relevant team and year
    team_players = df_copy[(df_copy['tmID'] == row['tmID']) & (df_copy['year'] == row['year'])]
    
    # Calculate the weighted mean of the RegularScore
    weighted_mean = (team_players['RegularScore'] * team_players['weight']).sum() / team_players['weight'].sum()
    
    return weighted_mean


def calculate_team_post_score(row, df):
    '''
    Calculate the team score based on the PostseasonScore of the players.
    Different weights are applied based on the stint value.
    '''
    # Create a copy and apply weights based on the stint values
    df_copy = df.copy()
    df_copy['weight'] = df_copy['stint'].map({0: 1, 1: 0.5, 2: 0.5})
    
    # Filter the DataFrame for the relevant team and year
    team_players = df_copy[(df_copy['tmID'] == row['tmID']) & (df_copy['year'] == row['year'])]
    
    # Calculate the weighted mean of the PostseasonScore
    weighted_mean = (team_players['PostseasonScore'] * team_players['weight']).sum() / team_players['weight'].sum()
    
    return weighted_mean



def predict_missing_regular_scores(df):
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()

    # Identify rows with missing RegularScore
    missing_mask = df['RegularScore'].isnull()

    # Prepare data for modeling
    # Drop rows where RegularScore is missing for training
    train_data = df[~missing_mask]

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for col in ['playerID', 'tmID', 'pos', 'college', 'collegeOther']:
        le = LabelEncoder()
        df[col] = df[col].fillna('Unknown')  # Fill NaN with 'Unknown'
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure all values are strings before encoding
        label_encoders[col] = le

    # Features to use for prediction
    features = ['year', 'stint', 'tmID', 'GP', 'GS', 'minutes', 'points', 'pos',
                'height', 'weight', 'college', 'collegeOther', 'birthDate', 'PostseasonScore']

    # Split data into features (X) and target (y)
    X = train_data[features]
    y = train_data['RegularScore']

        # Split data into features (X) and target (y)
    X = train_data[features]
    y = train_data['RegularScore']

    # Split into train and validation sets

    # Initialize and train the RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict missing RegularScore values
    missing_data = df[missing_mask]
    if not missing_data.empty:
        X_missing = missing_data[features]
        df.loc[missing_mask, 'RegularScore'] = model.predict(X_missing)

    return df


def calculate_predict_team_score(row, df_players):
    team_players = df_players[(df_players['tmID'] == row['tmID']) & (df_players['year'] == row['year'])]
    return team_players['RegularScore_next_year'].mean()

def calculate_teamScore_post(row, df):
    team_players = df[(df['tmID'] == row['tmID']) & (df['year'] == row['year'])]
    return team_players['PostseasonScore'].mean()


def merge_players_with_awards(players_df, awards_df):
    # Create a column 'awards' in awards_df to indicate if a player has gained an award
    players_df['awards'] = 0

    # compare the playerID in players_df and awards_df and merge it the count of awards in the same year to the players_df
    for index, row in awards_df.iterrows():
        playerID = row['playerID']
        year = row['year']
        players_df.loc[(players_df['playerID'] == playerID) & (players_df['year'] == year), 'awards'] += 1
    
    return players_df



def merge_awards(players_df, coaches_df, awards_df):
    # Step 1: Count awards for each player and year
    player_awards_count = awards_df.groupby(['playerID', 'year']).size().reset_index(name='awards')

    # Step 2: Count awards for each coach and year
    coach_awards_count = awards_df.groupby(['playerID', 'year']).size().reset_index(name='awards')
    coach_awards_count.rename(columns={'playerID': 'coachID'}, inplace=True)  # Rename playerID to coachID for coaches

    # Step 3: Merge the player awards count with players dataframe
    players_merged_df = players_df.merge(player_awards_count, on=['playerID', 'year'], how='left')

    # Step 4: Merge the coach awards count with coaches dataframe
    coaches_merged_df = coaches_df.merge(coach_awards_count, on=['coachID', 'year'], how='left')

    # Step 5: Fill NaN values in the 'awards' column with 0 for both players and coaches
    players_merged_df['awards'] = players_merged_df['awards'].fillna(0).astype(int)
    coaches_merged_df['awards'] = coaches_merged_df['awards'].fillna(0).astype(int)

    return players_merged_df, coaches_merged_df






def merge_coaches(df_teams, df_coaches):
    # Create a copy and apply weights
    df_coachesOP = df_coaches.copy()
    df_coachesOP['weight'] = df_coachesOP['stint'].map({0: 1, 1: 0.5, 2: 0.5})

    # Group by year and tmID, calculate weighted mean for each field separately
    weighted_means = (
        df_coachesOP
        .groupby(['year', 'tmID'])
        .apply(lambda group: pd.Series({
            'coaches_win_rate': (group['win_rate'] * group['weight']).sum() / group['weight'].sum(),
            'coaches_post': (group['post_win_rate'] * group['weight']).sum() / group['weight'].sum(),
            'coaches_history': (group['mean_win_rate'] * group['weight']).sum() / group['weight'].sum(),
            'coaches_history_pos': (group['mean_post_win_rate'] * group['weight']).sum() / group['weight'].sum(),
            #'awards_coach': (group['awards'] * group['weight']).sum(),
        }))
        .reset_index()
    )

    # Merge the result back into df_teams
    df_final = df_teams.copy()
    df_final = df_final.merge(weighted_means, on=['year', 'tmID'], how='left')

    return df_final

def merge_players(df_teams, df_players):
    # Create a copy and apply weights
    df_players_teamPredictScoreOP = df_players.copy()
    df_players_teamPredictScoreOP['weight'] = df_players_teamPredictScoreOP['stint'].map({0: 1, 1: 0.5, 2: 0.5})

    # Group by year and tmID, calculate weighted mean for each field separately
    weighted_means = (
        df_players_teamPredictScoreOP
        .groupby(['year', 'tmID'])
        .apply(lambda group: pd.Series({
            'squad_post_performance': (group['TeamPostScore'] * group['weight']).sum() / group['weight'].sum(),
            'squad_performance': (group['TeamScore'] * group['weight']).sum() / group['weight'].sum(),
            'awards_players': (group['awards'] * group['weight']).sum(),
        }))
        .reset_index()
    )

    # Merge the result back into df_teams
    df_teams = df_teams.copy()
    df_teams = df_teams.merge(weighted_means, on=['year', 'tmID'], how='left')

    return df_teams


def calculate_last_3_years_history(df):
    df['last_3_years_history'] = None  # Initialize the column
    for tmID, group in df.groupby('tmID'):
        group = group.sort_values('year')  # Sort by year
        scores = []
        
        for i in range(len(group)):
            # Calculate the average score for the current and previous years
            num_years = min(i + 1, 3)  # Max years is 3
            last_years = group.iloc[max(0, i - num_years + 1):i + 1]
            avg_score = last_years.apply(
                lambda row: (row['squad_performance']), axis=1
            ).mean()
            scores.append(avg_score)
        
        # Assign scores back to the DataFrame
        df.loc[group.index, 'last_3_years_history'] = scores

    return df


def calculate_history_excluding_last_3_years(df):
    df['history_excluding_last_3_years'] = None  # Initialize the column
    for tmID, group in df.groupby('tmID'):
        group = group.sort_values('year')  # Sort by year
        scores = []
        
        for i in range(len(group)):
            # Calculate the average score for all years except the last 3
            if i < 3:
                scores.append(0)  # Not enough data to exclude the last 3 years
            else:
                history_years = group.iloc[:i - 2]  # Exclude the last 3 years
                avg_score = history_years.apply(
                    lambda row: row['squad_performance'], axis=1
                ).mean()
                scores.append(avg_score)
        
        # Assign scores back to the DataFrame
        df.loc[group.index, 'history_until_3_years_left'] = scores

    return df


def calculate_team_lore(df):
    # Initialize the team_lore column
    df['team_lore'] = None

    # Group by tmID
    for tmID, group in df.groupby('tmID'):
        group = group.sort_values('year')  # Sort by year
        previous_team_lore = 0  # Initialize the previous year's team_lore

        for i, row in group.iterrows():
            # Calculate team_lore for the current year
            current_team_lore = (
                row['attend'] * 0.01 +
                row['last_3_years_history'] * 0.9 +
                row['awards_players'] * 0.2 
                # +
                # row['awards_coach'] * 0.1
            )

            # Add the previous year's team_lore * 0.2
            team_lore = current_team_lore + previous_team_lore * 0.2

            # Assign the calculated team_lore back to the DataFrame
            df.at[i, 'team_lore'] = team_lore

            # Update the previous_team_lore for the next iteration
            previous_team_lore = team_lore

    return df