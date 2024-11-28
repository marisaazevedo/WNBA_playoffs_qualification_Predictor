import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error


# Define regular season score calculation
def calculate_regular_season_score(row, weights):
    if 'G' in row['pos'] and 'F' in row['pos']:  # G-F or F-G
        return (
            0.15 * row['points'] * weights['points'] +
            0.275 * row['assists'] * weights['assists'] +
            0.1 * row['steals'] * weights['steals'] +
            0.175 * row['threeMade'] * weights['threeMade'] +
            0.15 * row['rebounds'] * weights['rebounds'] +
            0.075 * row['blocks'] * weights['blocks'] -
            0.1 * row['turnovers'] * weights['turnovers']
        )
    elif 'F' in row['pos'] and 'C' in row['pos']:  # F-C or C-F
        return (
            0.35 * row['rebounds'] * weights['rebounds'] +
            0.35 * row['points'] * weights['points'] +
            0.15 * row['blocks'] * weights['blocks'] +
            0.075 * row['assists'] * weights['assists'] -
            0.05 * row['turnovers'] * weights['turnovers']
        )
    elif 'G' in row['pos']:  # Pure Guard
        return (
            0.3 * row['points'] * weights['points'] +
            0.25 * row['assists'] * weights['assists'] +
            0.2 * row['steals'] * weights['steals'] +
            0.15 * row['threeMade'] * weights['threeMade'] -
            0.1 * row['turnovers'] * weights['turnovers']
        )
    elif 'F' in row['pos']:  # Pure Forward
        return (
            0.35 * row['points'] * weights['points'] +
            0.3 * row['rebounds'] * weights['rebounds'] +
            0.2 * row['assists'] * weights['assists'] +
            0.1 * row['blocks'] * weights['blocks'] -
            0.05 * row['turnovers'] * weights['turnovers']
        )
    elif 'C' in row['pos']:  # Pure Center
        return (
            0.4 * row['rebounds'] * weights['rebounds'] +
            0.35 * row['points'] * weights['points'] +
            0.2 * row['blocks'] * weights['blocks'] -
            0.05 * row['turnovers'] * weights['turnovers']
        )
    return 0


# Define postseason score calculation
def calculate_postseason_score(row):
    if row['PostMinutes'] == 0:  # Avoid division by zero
        return 0
    PostPointsPerMinute = row['PostPoints'] / row['PostMinutes']
    PostReboundsPerMinute = row['PostRebounds'] / row['PostMinutes']
    PostAssistsPerMinute = row['PostAssists'] / row['PostMinutes']
    PostStealsPerMinute = row['PostSteals'] / row['PostMinutes']
    PostBlocksPerMinute = row['PostBlocks'] / row['PostMinutes']
    PostTurnoversPerMinute = row['PostTurnovers'] / row['PostMinutes']
    PostThreeMadePerMinute = row['PostthreeMade'] / row['PostMinutes']
    
    if 'G' in row['pos'] and 'F' in row['pos']:  # G-F or F-G
        return (0.15 * PostPointsPerMinute + 0.275 * PostAssistsPerMinute + 
                0.1 * PostStealsPerMinute + 0.175 * PostThreeMadePerMinute + 
                0.15 * PostReboundsPerMinute + 0.075 * PostBlocksPerMinute - 
                0.1 * PostTurnoversPerMinute)
    elif 'F' in row['pos'] and 'C' in row['pos']:  # F-C or C-F
        return (0.35 * PostReboundsPerMinute + 0.35 * PostPointsPerMinute + 
                0.15 * PostBlocksPerMinute + 0.075 * PostAssistsPerMinute - 
                0.05 * PostTurnoversPerMinute)
    elif 'G' in row['pos']:  # Pure Guard
        return (0.3 * PostPointsPerMinute + 0.25 * PostAssistsPerMinute + 
                0.2 * PostStealsPerMinute + 0.15 * PostThreeMadePerMinute - 
                0.1 * PostTurnoversPerMinute)
    elif 'F' in row['pos']:  # Pure Forward
        return (0.35 * PostPointsPerMinute + 0.3 * PostReboundsPerMinute + 
                0.2 * PostAssistsPerMinute + 0.1 * PostBlocksPerMinute - 
                0.05 * PostTurnoversPerMinute)
    elif 'C' in row['pos']:  # Pure Center
        return (0.4 * PostReboundsPerMinute + 0.35 * PostPointsPerMinute + 
                0.2 * PostBlocksPerMinute - 0.05 * PostTurnoversPerMinute)
    return 0


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


def predict_heuristic_next(df):
    # === Feature Engineering ===

    # Rolling averages for past performance
    df['avg_points_last_3_years'] = df.groupby('playerID')['points'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['avg_minutes_last_3_years'] = df.groupby('playerID')['minutes'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['team_avg_score'] = df.groupby('tmID')['TeamScore'].transform('mean')

    # Fill missing or invalid values
    df = df.fillna(0)

    # === Encode 'pos' for the Model ===
    # Use one-hot encoding for the position field
    pos_encoder = OneHotEncoder(sparse_output=False)
    pos_encoded = pos_encoder.fit_transform(df[['pos']])
    pos_encoded_df = pd.DataFrame(pos_encoded, columns=pos_encoder.get_feature_names_out(['pos']))
    df = pd.concat([df, pos_encoded_df], axis=1)

    # === Shift RegularScore to Next Year ===
    df['RegularScore_next_year'] = df.groupby('playerID')['RegularScore'].shift(-1)

    # Handle null values in RegularScore_next_year
    def handle_nulls(row):
        if pd.isna(row['RegularScore_next_year']):
            next_year_data = df[(df['playerID'] == row['playerID']) & (df['year'] == row['year'] + 1)]
            if next_year_data.empty:
                return None  # Predict this value
            else:
                return 0  # Assume the player retired
        return row['RegularScore_next_year']

    df['RegularScore_next_year'] = df.apply(handle_nulls, axis=1)

    # Filter out rows with null target values
    df_non_null = df.dropna(subset=['RegularScore_next_year'])

    # === Model Training ===

    # Define features and target
    features = ['avg_points_last_3_years', 'avg_minutes_last_3_years', 'team_avg_score',
                'height', 'weight', 'GP', 'GS', 'stint', 'year'] + list(pos_encoded_df.columns)
    X = df_non_null[features]
    y = df_non_null['RegularScore_next_year']

    # Train model on the filtered dataset
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict on the entire dataset to create 'predict_score'
    df['predict_score'] = model.predict(df[features])

    # Evaluate
    mse = mean_squared_error(y, df.loc[df_non_null.index, 'predict_score'])
    print(f'Mean Squared Error: {mse}')

        # Fill the remaining null values in RegularScore_next_year with the predicted scores
    df['RegularScore_next_year'] = df.apply(
        lambda row: (
            row['predict_score'] if pd.isna(row['RegularScore_next_year']) and 
            df[(df['playerID'] == row['playerID']) & (df['year'] == row['year'] + 1)].empty and 
            df[(df['playerID'] == row['playerID']) & (df['year'] > row['year'] + 1)].empty else
            0 if pd.isna(row['RegularScore_next_year']) and 
            not df[(df['playerID'] == row['playerID']) & (df['year'] > row['year'] + 1)].empty else
            row['RegularScore_next_year']
        ),
        axis=1
    )

    # Drop the temporary 'predict_score' column
    #df = df.drop(columns=['predict_score, avg_points_last_3_years, avg_minutes_last_3_years, team_avg_score'])

    return df


def calculate_predict_team_score(row, df_players):
    team_players = df_players[(df_players['tmID'] == row['tmID']) & (df_players['year'] == row['year'])]
    return team_players['RegularScore_next_year'].mean()

def calculate_teamScore_post(row, df):
    team_players = df[(df['tmID'] == row['tmID']) & (df['year'] == row['year'])]
    return team_players['PostseasonScore'].mean()


def merge_players_with_awards(players_df, awards_df):
    # Create a column 'awards' in awards_df to indicate if a player has gained an award
    awards_df['awards'] = 1

    # Group the awards data by playerID and year, and aggregate the awards count
    awards_agg = awards_df.groupby(['playerID', 'year'])['awards'].sum().reset_index()

    # Merge the aggregated awards data with the players DataFrame
    merged_df = players_df.merge(awards_agg, on=['playerID', 'year'], how='left')

    # Fill NaN values in the awards column with 0
    merged_df['awards'] = merged_df['awards'].fillna(0).astype(int)

    return merged_df


def merge_awards(players_df, coaches_df, awards_df):
    # Create a column 'awards' in awards_df to indicate if a player or coach has gained an award
    awards_df['awards'] = 1

    # Separate player awards and coach awards
    player_awards = awards_df[~awards_df['award'].str.contains("Coach")]
    coach_awards = awards_df[awards_df['award'].str.contains("Coach")]

    # Group the player awards data by playerID and year, and aggregate the awards count
    player_awards_agg = player_awards.groupby(['playerID', 'year'])['awards'].sum().reset_index()

    # Group the coach awards data by coachID and year, and aggregate the awards count
    coach_awards_agg = coach_awards.groupby(['playerID', 'year'])['awards'].sum().reset_index()
    coach_awards_agg.rename(columns={'playerID': 'coachID'}, inplace=True)

    # Merge the aggregated player awards data with the players DataFrame
    players_merged_df = players_df.merge(player_awards_agg, on=['playerID', 'year'], how='left')

    # Merge the aggregated coach awards data with the coaches DataFrame
    coaches_merged_df = coaches_df.merge(coach_awards_agg, on=['coachID', 'year'], how='left')

    # Fill NaN values in the awards column with 0
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
            'awards_coach': (group['awards'] * group['weight']).sum(),
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
            'team_predict_mean': (group['predict_team_score'] * group['weight']).sum() / group['weight'].sum(),
            'team_post_mean': (group['teamScore_post'] * group['weight']).sum() / group['weight'].sum(),
            'teamScore_mean': (group['TeamScore'] * group['weight']).sum() / group['weight'].sum(),
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
                lambda row: (row['team_predict_mean']), axis=1
            ).mean()
            scores.append(avg_score)
        
        # Assign scores back to the DataFrame
        df.loc[group.index, 'last_3_years_history'] = scores

    return df