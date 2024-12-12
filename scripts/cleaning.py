import pandas as pd


def clean_teams(df):
    '''
    This function cleans the teams table by dropping columns that are not needed,
    changing the values in the columns to numerical and agregating columns.
    '''
    # Drop column: 'lgID', 'franchID', 'divID', 'seeded'
    df = df.drop(columns=['lgID', 'franchID', 'divID', 'seeded'])
    df = df.sort_values(['year'])
    # Change the Y to 1 and N to 0 in the playoff column
    df['playoff'] = df['playoff'].replace({'Y': 1, 'N': 0}).infer_objects(copy=False)
    # change the conference column to 1 for East and 0 for West
    df['confID'] = df['confID'].replace({'EA': 1, 'WE': 0}).infer_objects(copy=False)
    # In the firstRound semis and finals columns, replace W with 1, L with 0, and null with -1
    columns_to_replace = ['firstRound', 'semis', 'finals']
    df[columns_to_replace] = df[columns_to_replace].replace({'W': 1, 'L': 0}).fillna(-1).infer_objects(copy=False)
    # Drop column: 'name'
    df = df.drop(columns=['name'])
    # Drop columns: 'opptmORB', 'tmTRB', 'tmDRB', 'tmORB' , 'opptmDRB', 'opptmTRB'
    df = df.drop(columns=['opptmORB', 'tmTRB', 'tmDRB', 'tmORB' , 'opptmDRB', 'opptmTRB'])
    # Drop column: 'arena'
    df = df.drop(columns=['arena'])
    # Win Rates
    df['win_rate'] = (df['won'] / df['GP'] * 100).round(2)
    df['homeWin_rate'] = (df['homeW'] / (df['homeW'] + df['homeL']) * 100).round(2)
    df['awayWin_rate'] = (df['awayW'] / (df['awayW'] + df['awayL']) * 100).round(2)
    df['confW_rate'] = (df['confW'] / (df['confW'] + df['confL']) * 100).round(2)
    # Drop columns: 'won', 'lost' and 7 other columns
    df = df.drop(columns=['won', 'lost', 'homeW', 'homeL', 'awayW', 'awayL', 'confW', 'confL', 'GP'])

    # Offensive metrics
    df['FGE'] = df['o_fgm'] / df['o_fga']
    df['FTE'] = df['o_ftm'] / df['o_fta']
    df['3PE'] = df['o_3pm'] / df['o_3pa']
    df['OREB%'] = df['o_oreb'] / (df['o_oreb'] + df['d_dreb'])
    df['AST/TO'] = df['o_asts'] / (df['o_to'] + 1e-9)  # To avoid division by zero

    # Defensive metrics (inverted for opponent metrics where lower values are better)
    df['Opp_FGE'] = 1 - (df['d_fgm'] / df['d_fga'])
    df['Opp_FTE'] = 1 - (df['d_ftm'] / df['d_fta'])
    df['Opp_3PE'] = 1 - (df['d_3pm'] / df['d_3pa'])
    df['DREB%'] = df['d_dreb'] / (df['d_dreb'] + df['o_oreb'])
    df['TO_Forced'] = df['d_to'] / (df['o_to'] + df['d_to'] + 1e-9)

    # Offensive weighted score
    df['Offensive_Score'] = (
        0.20 * df['FGE'] +
        0.10 * df['FTE'] +
        0.10 * df['3PE'] +
        0.15 * df['OREB%'] +
        0.15 * df['AST/TO'] +
        0.30 * df['o_pts'] / df['o_pts'].max()  # Normalizing points
    )

    # Defensive weighted score
    df['Defensive_Score'] = (
        0.20 * df['Opp_FGE'] +
        0.10 * df['Opp_FTE'] +
        0.10 * df['Opp_3PE'] +
        0.15 * df['DREB%'] +
        0.15 * df['TO_Forced'] +
        0.30 * (1 - df['d_pts'] / df['d_pts'].max())  # Normalizing points allowed
    )


    columns_to_drop = [
        'FGE', 'FTE', '3PE', 'OREB%', 'AST/TO',  # Offensive metrics
        'Opp_FGE', 'Opp_FTE', 'Opp_3PE', 'DREB%', 'TO_Forced',  # Defensive metrics
        'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 
        'o_oreb', 'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts',
        'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb', 'd_dreb',
        'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts',
    ]

    df = df.drop(columns=columns_to_drop)
    return df


def clean_players_table(df):
    '''
    This function cleans the players table by dropping columns that are not needed,
    changing the values in the columns to numerical and agregating columns.
    '''
    # Drop column: 'firstseason'
    df = df.drop(columns=['firstseason', 'lastseason', 'deathDate'])
    # Extract year from birthDate
    df['birthDate'] = df['birthDate'].str[:4].astype(int)
    return df


def clean_pteams_data(df):
    '''
    This function cleans the players teams table by dropping columns that are not needed,
    '''
    # Sort by year, then aggregate by playerID
    df = df.sort_values(by=['year', 'playerID', 'stint'])
    return df

def clean_merge_players(df_players, df_pteams):
    '''
    This function clean merges the players and players teams 
    tables and cleans the merged table.
    '''
    df_players = clean_players_table(df_players.copy())
    df_pteams = clean_pteams_data(df_pteams.copy())

    # Merge the two tables
    df_players_merged = df_pteams.merge(df_players, left_on='playerID', right_on='bioID', how='left')
    df_players_merged = df_players_merged.drop(columns=['bioID', 'lgID'])
    df_players_merged = df_players_merged[df_players_merged['stint'] != 3]

    return df_players_merged

def clean_coaches(df):
    '''
    This function cleans the coaches table by dropping columns that are not needed,
    agregating columns by calculating win_rate and post_win_rate.
    adding other columns like mean_win_rate and mean_post_win_rate from the past 5 years.
    '''
    # Drop column: 'lgID'
    df = df.drop(columns=['lgID'])
    # Sort by column: 'year' (ascending)
    df = df.sort_values(['year', 'stint', 'tmID'])
    # Calculate win_rate and post_win_rate
    df['win_rate'] = (df['won'] / (df['won'] + df['lost'])) * 100
    df['post_win_rate'] = 0.0
    df.loc[(df['post_wins'] != 0) | (df['post_losses'] != 0), 'post_win_rate'] = (df['post_wins'] / (df['post_wins'] + df['post_losses'])) * 100
    # Calculate mean win_rate and post_win_rate for the past 5 years
    df['mean_win_rate'] = df.groupby('coachID')['win_rate'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['mean_post_win_rate'] = df.groupby('coachID')['post_win_rate'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    return df