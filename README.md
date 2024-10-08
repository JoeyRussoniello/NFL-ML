# NFL Sports Betting Model

## Overview
This project leverages machine learning to predict NFL game outcomes and Super Bowl winners based on NFL team statistics sourced from https://www.nfl.com/stats/team-stats/ and similar websites. The model scrapes historical NFL data and trains two RandomForest classifiers using **scikit-learn**. One which forecasts the likelihood of a given team winning the Super Bowl in its year, and a second to predict which of two teams will win a particular game based on their team statistics last year.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Usage](#usage)
- [Additional Features](#additional-features)

## Installation
To run this project, you will need Python 3.x and the following Python libraries:

```bash
pip install lxml requests beautifulsoup4 pandas numpy matplotlib seaborn scikit-learn scipy
```
Ensure you have the following libraries installed:

requests, BeautifulSoup for web scraping
lxml as a dependency for the requests module
pandas, numpy for data processing
matplotlib, seaborn for visualizations
scikit-learn, scipy for machine learning and statistical analysis

Dataset
The project scrapes NFL team statistics from the official NFL website, covering offense, defense, and special teams for various seasons. The results of the webscraping are stored in 3 CSV files located in the DataSets folder to avoid repeatedly scraping the web; however, they will be automatically updated when more data is publically available. The dataset contains information such as:

Team (NFL Team)
Year (Season Year)
Offense: Passing, Rushing, Receiving stats, etc.
Defense: Tackles, Interceptions, etc.
Special Teams: Field goals, Kick returns, etc.

## Data Gathering
The data is collected through custom scraper functions. Here's a snippet of how the historical data is gathered:

``` python
# NFL Historical Data Scraper
class NFLData:
    # Initializes the scraper
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.dflist = []

    def pull_all(self):
        while log <= self.end:
            dfs = single_season(log)
            self.dflist.append(dfs)
            log += 1

    def to_one_df(self):
        combined_df = pd.merge(self.dflist[0], self.dflist[1], on=['Team', 'Year'])
        for df in self.dflist[2:]:
            combined_df = merged(combined_df, df, on=['Team', 'Year'])

    def save(self, name):
        self.df.to_csv(name)
```

The NFLData class handles collecting data for multiple seasons and combining them into a single DataFrame.

# Model Training
The model uses a RandomForestClassifier to predict NFL game outcomes. Here’s an example of how the model is trained with different settings using cross-validation:

``` python 
def try_different_settings(data_train, label_train):
    best_crossval = 0
    bestsettings = []
    estimators = [50, 100, 200]
    features = ["sqrt", "log2"]
    max_depths = [10, 20, None]

    for estimator in estimators:
        for feature in features:
            for depth in max_depths:
                forest = RandomForestClassifier(n_estimators=estimator, max_features=feature, max_depth=depth)
                forest.fit(data_train, label_train)
                score = np.mean(cross_val_score(forest, data_train, label_train))
                if score > best_crossval:
                    best_crossval = score
                    bestsettings = [estimator, feature, depth]
                    best_model = forest
    print(f"Best settings: {bestsettings}")

```

Example usage of the model:

``` python
all_games_forest = RandomForestClassifier(n_estimators=100, max_depth=10, max_features="sqrt", random_state=110)
all_games_forest.fit(data_train, label_train)
effectiveness = all_games_forest.score(data_test, label_test)
print(f"Model effectiveness: {round(effectiveness, 4)}")
```

# Usage
To use the model:

Scrape and update data using the provided scraper in main.ipynb
Follow the syntax of the example command provided ex: predict_winner("Patriots", "Eagles", 2024)

# Additional Features
Pre-scraped data: The DataSets folder contains 3 CSV files with pre-scraped NFL statistics, avoiding the need to rerun the scraper each time.
Visualizations: Statistical visualizations of the model's results are available in the Visualizations folder.
Statistical Testing: Further analysis of the model’s predictions can be done through the statistical_exploration.ipynb notebook, where statistical tests and data exploration are carried out.
