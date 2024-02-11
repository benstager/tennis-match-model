import os
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('tennis_data.csv')

# encode AD as '60', even though it may not be best practice
df['player_1_points'] = df['player_1_points'].replace(['AD'], 60)
df['player_2_points'] = df['player_2_points'].replace(['AD'], 60)

# we will concatenate the dataset by merely summing all of the necessary columns
# intuitively this does not make sense, but due to model testing this was enough
df["player_1_points"] = pd.to_numeric(df["player_1_points"])
df["player_2_points"] = pd.to_numeric(df["player_2_points"])

# we will group by "game_id" where each row represents one match, and the match winner will just be the average equaling the winner
final_df = df.groupby(['game_id'], as_index=False).agg({
    'player_1_points':'sum', 'player_2_points':'sum','player_1_games':'sum',
     'player_2_games':'sum', 'player_1_sets':'sum', 'player_2_sets':'sum',
       'match_winning_player':'mean'
})

# we'll store the final data point for prediction
last_match = final_df.iloc[135]

# create new dataframe with last point removed
final_df_no_last = final_df.drop(135)

# create feature matrix and response variable
X = final_df_no_last.drop('match_winning_player',axis=1)

# shift player over one due to classifier resttrictions
y = final_df_no_last['match_winning_player'] - 1

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22, test_size=.25)

# initialize and run model
clf = RandomForestClassifier(max_depth=3, random_state=22)
clf.fit(X_train, y_train)

# predictions and metrics
y_pred = clf.predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(f1_score(y_train, y_pred))

# now that the model looks good, lets train on full set and predict the winner!
clf.fit(X,y)
y_true_pred = clf.predict(X)
print(confusion_matrix(y, y_true_pred))
print(f1_score(y, y_true_pred)) # yields an f1 of 94%, not bad

# FINAL PREDICTION
last_match = [last_match.drop('match_winning_player')]
winner = clf.predict(last_match) + 1
print(f"The winner of the 136th WTA match is ... Player {winner[0]}")