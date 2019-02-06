# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:15:57 2018

@author: Chinmay
"""


# Import all the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pyspark.sql.types import StringType
from pyspark import SQLContext , SparkContext
from pyspark.sql.session import SparkSession

# Create a Spark Context

sc = SparkContext('local','fisrt_SPARK')  # If using locally
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# sc.stop()

# Load all the data into Spark Data Frame

df = (spark.read.format("csv").options(header="true" , inferSchema = True ).load("D:/Dbda/Project/Final Draft/Chinmay/Final/final_all_3.0.csv"))

batsman_all_data= (spark.read.format("csv").options(header="true" , inferSchema = True).load("D:/Dbda/Project/Final Draft/Chinmay/Final/all_batsmans_data_2.0.csv"))

# List of the teams

teams = ['Australia' , 'New Zealand' , 'India' , 'Zimbabwe' , 'Bangladesh' , 'South Africa' , 'England' 
         , 'Sri Lanka' , 'Pakistan' , 'West Indies' , 'Ireland']


match_type = 'ODI'

team1 = 'India'
team2 = 'Australia'



team_1 = [ 'AM Rahane' , 'RG Sharma' , 'V Kohli' , 'MK Pandey', 'KD Jadhav', 'MS Dhoni' , 'HH Pandya'  , 'B Kumar' , 'JJ Bumrah' , 'Kuldeep Yadav' , 'YZ Chahal']

team_2 = ['H Cartwright' , 'DA Warner' , 'SPD Smith' , 'T Head' , 'GJ Maxwell' , 'M Stoinis' , 'MS Wade' , 'A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'K Richardson']


bowlers_1 = ['B Kumar' , 'JJ Bumrah' , 'Kuldeep Yadav' , 'HH Pandya' , 'YZ Chahal']

bowlers_2 = ['A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'KW Richardson', 'M Stoinis', 'T Head']

#bowlers_2 = ['CR Woakes' , 'DJ Willey' , 'JT Ball' , 'BA Stokes' , 'MM Ali' , 'AU Rashid' , 'JE Root']

#players = df.filter(df['`team`'].rlike('England')).select('batsman').distinct()

#df_batsman = pd.DataFrame(df.filter(df['`team`'].rlike('England')).select('batsman').distinct().collect())
#df_bolwer = pd.DataFrame(df.filter(df['`team`'].rlike('England')).select('bowler').distinct().collect())
import math
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan,isnull,when,count
from sklearn.ensemble import RandomForestRegressor

# Function to predict Player Score 

def score_predict(batsman , bowlers , team1 , team2 , match_type):
    
    #batsman = 'V Kohli'    
    #bowlers = ['A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'KW Richardson', 'M Stoinis', 'T Head']
    ############ Prediction Based On Bowlers ##############################
    
    batsman_data = batsman_all_data[(batsman_all_data['name'] == batsman)  & (batsman_all_data['match_type'] == match_type)]
    
    batsman_data_team = batsman_data[(batsman_data['against'] == team2)]
    
    if batsman_data_team.select('match_id').count() == 0:
        print("No Data Found against" , team2)
        return 0,0,0,0,0,0,0,0,0
    
    batsman_data_team = batsman_data_team.toPandas()
    
    batsman_data = batsman_all_data[(batsman_all_data['name'] == batsman)  & (batsman_all_data['match_type'] == match_type)]
    
    batsman_data_team = batsman_data[(batsman_data['against'] == team2)]
    
    if batsman_data_team.select('match_id').count() == 0:
        print("No Data Found against" , team2)
        return 0,0,0,0,0,0,
    
    batsman_data_team = batsman_data_team.toPandas()
    
    batsman_data_team['bowler_encoded'] = batsman_data_team['bowler'].astype('category').cat.codes
    
    A = batsman_data_team.loc[:, ['balls' , 'bowler_encoded' , 'home_away']].values
    B = batsman_data_team.loc[:, 'runs_scored'].values
    
    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    
    score_pred = []
    score_act_pred = []
    score_act = []
    for bowler in bowlers:
        bowl = []
        test = []
        if batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].empty:
            continue
        
        encoded_bowl = batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].index
        
        
        encoded_bowler = batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].iloc[0]
        if not encoded_bowler:
            continue
        
        X_train = A.loc[A.index != list(encoded_bowl)[-1]].values
        y_train = B.loc[B.index != list(encoded_bowl)[-1]].values
        X_test = A.loc[A.index == list(encoded_bowl)[-1]].values
        y_test = B.loc[B.index == list(encoded_bowl)[-1]].values
        
        y_test = y_test.tolist()
        
        if batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].count() == 0:
            continue
        
        avg = batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].sum()/batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].count()
        bowl.append(avg)
        bowl.append(encoded_bowler)
        test.append(bowl)
        
        model = GaussianNB()
        
        model.fit(X_train, y_train)
        regressor = RandomForestRegressor(max_depth = 2, min_samples_split=2, n_estimators = 100, random_state = 1)
        regressor.fit(X_train,y_train)
        if not test:
            predicted2 = 0
        else:
            #predicted2= model.predict(test)
            #predicted = model.predict(X_test)    
            predicted = regressor.predict(X_test)
            
        #score_pred.append(sum(predicted2))
        score_act_pred.append(sum(predicted))
        score_act.append(y_test[0][0])
    #predicted = sum(score_pred)
    predicted = math.ceil(sum(score_act_pred))
    predicted2 = math.ceil(sum(score_act))
    
    #print(batsman , "will score against bowlers predicted" , sum(score_pred))
    print(batsman , "will score against bowlers predicted" , predicted)
    print(batsman , "will score against bowlers " , predicted2)


    ############### Predicition Based On Against Team Record #########################
    from pyspark.sql.functions import col
    df_team = df.filter(df['`info.teams`'].rlike('India'))
    df_team = df_team.toPandas()
    
    df_team['info.teams'] = df_team['info.teams'].str.strip().apply(ast.literal_eval)
    
    match_index = df_team['index_all'].unique()
    batsman_data = pd.DataFrame()
    for mindex in match_index:
        
        df_match = df_team[(df_team['index_all']==mindex)].copy()
        
        if df_match[(df_team['batsman']==batsman)].empty:
            continue
        
        runs = 0
        balls_faced = 0
        bats_data = []
        for indexs,match_details in df_match.iterrows():
            
            ls = list(df_match.loc[indexs , 'info.teams'])
            if ls[0] == team1:
                home = 0
                opposition = ls[1]
            else:
                home = 1
                opposition = ls[0]
                
            if (df_match.loc[indexs,'batsman'] == batsman):
                runs = runs + df_match.loc[indexs,'runs.batsman']
                balls_faced = balls_faced + 1
                
        bats_data.append(df_match.loc[indexs,'info.match_type'])
        bats_data.append(df_match.loc[indexs,'info.dates'])
        bats_data.append(df_match.loc[indexs,'info.neutral_venue'])
        bats_data.append(home)
        bats_data.append(opposition)
        bats_data.append(runs)
        bats_data.append(balls_faced)
        bats_data.append(df_match.loc[indexs,'info.venue'])
        batsman_data = batsman_data.append(pd.Series(bats_data), ignore_index=True)
        
    batsman_data.columns = ['match_type' , 'date' , 'neutral' ,'home_away', 'against' , 'runs' , 'balls_faced' , 'venue']
    
    pd.options.mode.chained_assignment = None
    #batsman_data2 = batsman_data[(batsman_data['match_type'] == match_type) & (batsman_data['home_away'] == home)]
    batsman_data2 = batsman_data[(batsman_data['match_type'] == match_type)]
    encoded = batsman_data2.loc[:,'against'].astype('category').cat.codes
    encoded = list(encoded)
    encoded = [x+1 for x in encoded]
    batsman_data2['team_encoded'] = encoded
    
    encoded_t = batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].index#.iloc[0]
    
    A = batsman_data2.loc[:, ['balls_faced' , 'team_encoded']]#.values
    B = batsman_data2.loc[:, 'runs']#.values
    
    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    
    if not list(encoded_t):
        predicted3= 0
        predicted4 = 0
    #X_train = A.loc[list(encoded_team)[-1],:].values
    else:
        X_train = A.loc[A.index != list(encoded_t)[-1]].values
        y_train = B.loc[B.index != list(encoded_t)[-1]].values
        X_test = A.loc[A.index == list(encoded_t)[-1]].values
        y_test = B.loc[B.index == list(encoded_t)[-1]].values
        
        y_test = y_test.tolist()
        #X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.05, random_state = 0)
        
        if batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].empty:
            print("No Data Found against" , team2)
            return predicted,predicted2,0,0,0,0
            
        
        encoded_team = batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].iloc[0]
        avg = batsman_data2[(batsman_data2['team_encoded'] == encoded_team) & (batsman_data2['balls_faced'] != 0)]['balls_faced'].sum()/batsman_data2[(batsman_data2['team_encoded'] == encoded_team) & (batsman_data2['balls_faced'] != 0)]['balls_faced'].count()
        
        model = GaussianNB()
        
        model.fit(X_train, y_train)
        regressor = RandomForestRegressor(max_depth = 5, min_samples_split=2, n_estimators = 100, random_state = 1)
        regressor.fit(X_train,y_train)
        
        test = []
        team = []
        team.append(13)
        team.append(encoded_team)
        test.append(team)
        #predicted4= model.predict(test)
        #predicted3= sum(model.predict(X_test))
        predicted3 = math.ceil(sum(regressor.predict(X_test)))
        predicted4 = y_test[0][0]
    
############### Predicition Based On Against Team Record On Home / Away #########################
    
    from pyspark.sql.functions import col

    batsman_data2 = batsman_data[(batsman_data['match_type'] == match_type) & (batsman_data['home_away'] == home)]
    batsman_data2['team_encoded'] = batsman_data2['against'].astype('category').cat.codes
    
    encoded_t = batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].index#.iloc[0]
    
    A = batsman_data2.loc[:, ['balls_faced' , 'team_encoded' , 'home_away']]#.values
    B = batsman_data2.loc[:, 'runs']#.values
    
    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    
    if not list(encoded_t):
        predicted5 = 0
        predicted6 = 0
    #X_train = A.loc[list(encoded_team)[-1],:].values
    else:
        X_train = A.loc[A.index != list(encoded_t)[-1]].values
        y_train = B.loc[B.index != list(encoded_t)[-1]].values
        X_test = A.loc[A.index == list(encoded_t)[-1]].values
        y_test = B.loc[B.index == list(encoded_t)[-1]].values
        
        y_test = y_test.tolist()
        #X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.05, random_state = 0)
        
        if batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].empty:
            print("No Data Found against" , team2)
            return predicted,predicted2,predicted3,predicted4,0,0
            
        
        encoded_team = batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].iloc[0]
        avg = batsman_data2[(batsman_data2['team_encoded'] == encoded_team) & (batsman_data2['balls_faced'] != 0)]['balls_faced'].sum()/batsman_data2[(batsman_data2['team_encoded'] == encoded_team) & (batsman_data2['balls_faced'] != 0)]['balls_faced'].count()
        
        model = GaussianNB()
        
        model.fit(X_train, y_train)
        regressor = RandomForestRegressor(max_depth = 5, min_samples_split=2, n_estimators = 100, random_state = 1)
        regressor.fit(X_train,y_train)
        
        test = []
        team = []
        team.append(avg)
        team.append(encoded_team)
        test.append(team)
        #predicted7= sum(model.predict(test))
        #predicted5= sum(model.predict(X_test))
        predicted5 = math.ceil(sum(regressor.predict(X_test)))
        predicted6=y_test[0][0]
    
    return predicted,predicted2,predicted3,predicted4,predicted5,predicted6
        


against_bowlers_pred,against_bowlers_act,overall_pred,overall_act,home_away_pred,home_away_act = score_predict('MS Dhoni' , bowlers_2 , team1 , team2 , match_type)
print(against_bowlers_pred,against_bowlers_act,overall_pred,overall_act,home_away_pred,home_away_act)


import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 3
predicted_runs = (against_bowlers_pred, overall_pred, home_away_pred)
actual_runs = (against_bowlers_act, overall_act, home_away_act)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, predicted_runs, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Predicted Runs')
 
rects2 = plt.bar(index + bar_width, actual_runs, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Actual Runs')
 
plt.xlabel('Prediction Type')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index + bar_width, ('Bowlers', 'Team', 'Team with Home/Away'))
plt.legend()

plt.tight_layout()
plt.savefig("Ajinkya Rahane.png")
#plt.show()