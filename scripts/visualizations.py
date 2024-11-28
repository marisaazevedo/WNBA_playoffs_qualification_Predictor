from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# visualizations.py

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def plot_pca(df_clean):
    # Columns to use for PCA
    columns_for_pca = [
        'confID', 'win_rate', 'homeWin_rate', 'awayWin_rate', 'confW_rate',
        'Offensive_Score', 'Defensive_Score'
    ]

    # Standardizing the data
    scaler = StandardScaler()
    df_clean_scaled = scaler.fit_transform(df_clean[columns_for_pca])

    # Applying PCA once on the entire dataset
    pca = PCA(n_components=2)
    df_clean_pca = pca.fit_transform(df_clean_scaled)

    # Adding PCA components to the original DataFrame
    df_clean['PCA1'] = df_clean_pca[:, 0]
    df_clean['PCA2'] = df_clean_pca[:, 1]

    # Get unique years for plotting
    unique_years = df_clean['year'].unique()

    # Define a color mapping based on playoff and finals status
    def get_color(row):
        # First, determine playoff color: red for playoff, blue for non-playoff
        if row['playoff'] == 1:
            color = 'red'  # Playoff teams will be red
        else:
            color = 'blue'  # Non-playoff teams will be blue
        
        # Then, determine if they are in the finals (gold if in finals, silver if not)
        if row['finals'] == 1:
            return 'gold'  # Gold for teams in finals
        elif row['finals'] == 0:
            return 'silver'  # Silver for teams not in finals
        return color

    # Apply the color function to each row
    df_clean['color'] = df_clean.apply(get_color, axis=1)

    # Plotting PCA for each year separately with custom color coding
    for year in unique_years:
        df_year = df_clean[df_clean['year'] == year]
        
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(
            df_year['PCA1'], df_year['PCA2'],
            c=df_year['color'],  # Use the custom color array
            alpha=0.7
        )
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA of NBA Teams - Year {year}')
        
        # Adding tmID labels for each point
        for i, tm_id in enumerate(df_year['tmID']):
            plt.text(df_year['PCA1'].iloc[i], df_year['PCA2'].iloc[i], str(tm_id), fontsize=8, ha='right')
        
        # Adding a legend for each condition
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label='Winner', markerfacecolor='gold', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Finalist', markerfacecolor='silver', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Playoffs', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Non-Playoff', markerfacecolor='blue', markersize=10)
        ]
        plt.legend(handles=handles, title='Team Status')

        plt.show()

    # Get the principal component loadings
    loadings = pca.components_
    importances = ""
    # Display each principal component with the feature names and their contributions
    for i, component in enumerate(loadings):
        importances += f"\nPrincipal Component {i+1}\n"
        for feature, loading in zip(columns_for_pca, component):
            importances += f"{feature}: {loading:.4f}\n"

    return importances


def plot_player_regular_score(player_id, df_players_teamScore):
    # Filter the data for the given player
    df_player = df_players_teamScore[df_players_teamScore['playerID'] == player_id]

    # Plot the regular score for each year
    plt.figure(figsize=(10, 6))
    plt.plot(df_player['year'], df_player['RegularScore'], marker='o')
    plt.xlabel('Year')
    plt.ylabel('Regular Season Score')
    plt.title(f'Regular Season Score for Player {player_id}')
    plt.grid(True)
    plt.show()