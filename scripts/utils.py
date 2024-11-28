import pandas as pd
import os

folder_path = './data/'

def load_data():
    '''
    return a dictionary of dataframes
    '''
    teams = pd.read_csv(os.path.join(folder_path, 'teams.csv'))
    players = pd.read_csv(os.path.join(folder_path, 'players.csv'))
    players_teams = pd.read_csv(os.path.join(folder_path,'players_teams.csv'))
    coaches = pd.read_csv(os.path.join(folder_path, 'coaches.csv'))
    awards = pd.read_csv(os.path.join(folder_path, 'awards_players.csv'))
    series_post = pd.read_csv(os.path.join(folder_path, 'series_post.csv'))
    teams_post = pd.read_csv(os.path.join(folder_path, 'teams_post.csv'))

    return {'teams': teams, 'players': players, 'players_teams': players_teams, 'coaches': coaches, 'awards': awards, 'series_post': series_post, 'teams_post': teams_post}