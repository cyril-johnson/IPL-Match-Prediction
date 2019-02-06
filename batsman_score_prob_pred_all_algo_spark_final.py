# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:07:29 2018

@author: Chinmay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pyspark.sql.types import StringType
from pyspark import SQLContext , SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext('local','fisrt_SPARK')  # If using locally
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# sc.stop()

df = (spark.read.format("csv").options(header="true" , inferSchema = True ).load("D:/Dbda/Project/Final Draft/Chinmay/Final/final_all_3.0.csv"))

batsman_all_data= (spark.read.format("csv").options(header="true" , inferSchema = True).load("D:/Dbda/Project/Final Draft/Chinmay/Final/all_batsmans_data_2.0.csv"))

teams = ['Australia' , 'New Zealand' , 'India' , 'Zimbabwe' , 'Bangladesh' , 'South Africa' , 'England' 
         , 'Sri Lanka' , 'Pakistan' , 'West Indies' , 'Ireland']

match_type = 'ODI'

team1 = 'India'
team2 = 'Australia'


#df_batsman = pd.DataFrame(df.filter(df['`team`'].rlike('England')).select('batsman').distinct().collect())
#df_bolwer = pd.DataFrame(df.filter(df['`team`'].rlike('England')).select('bowler').distinct().collect())


from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan,isnull,when,count


def score_predict(batsman , team1 , team2 , match_type):
    #batsman = 'V Kohli'
    ############### Predicition Based On Against Team Record #########################
    df_pred = pd.DataFrame()
    
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
    
    A = batsman_data2.loc[:, ['balls_faced' , 'team_encoded' , 'home_away']]#.values
    B = batsman_data2.loc[:, 'runs']#.values
    
    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    
    if not list(encoded_t):
        predicted = 0
        predicted2 = 0
        #return None
    #X_train = A.loc[list(encoded_team)[-1],:].values
    else:
        
        X_train = A.loc[A.index != list(encoded_t)[0]].values
        y_train = B.loc[B.index != list(encoded_t)[0]].values
        X_test = A.loc[A.index == list(encoded_t)[0]].values
        y_test = B.loc[B.index == list(encoded_t)[0]].values
        
        y_test = y_test.tolist()
        #X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.05, random_state = 0)
        
#        from sklearn.preprocessing import StandardScaler
#        sc = StandardScaler()
#        X_train = sc.fit_transform(X_train)
        
        if batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].empty:
            print("No Data Found against" , team2)
            #return None
            
        
    # Gaussian Naive Bayes
        
        model = GaussianNB()
        
        model.fit(X_train, y_train)
        predicted= sum(model.predict(X_test))
        predicted2 = y_test[0][0]
        
    df_pred = df_pred.append(pd.Series(['Naive Bayes' , predicted, predicted2]), ignore_index=True)
     
    
    # Decision Tree
    
    from sklearn.tree import DecisionTreeRegressor
    classifier = DecisionTreeRegressor(random_state = 1)
    classifier.fit(X_train,y_train)
    
    predicted= sum(classifier.predict(X_test))
    
    df_pred = df_pred.append(pd.Series(['Decision Tree' , predicted, predicted2]) , ignore_index=True)
    
    
    # SVM
    
    from sklearn.svm import SVR
    classifier = SVR(kernel = 'linear')
    classifier.fit(X_train, y_train)
    predicted = sum(classifier.predict(X_test))
    
    df_pred = df_pred.append(pd.Series(['SVM' , predicted, predicted2]) , ignore_index=True)
    
    # Random Forest
    
    from sklearn.ensemble import RandomForestRegressor
    classifier = RandomForestRegressor(max_depth = 10, min_samples_split=2, n_estimators = 30, random_state = 1)
    classifier.fit(X_train,y_train)
    predicted = sum(classifier.predict(X_test))
    
    df_pred = df_pred.append(pd.Series(['Random Forest' , predicted, predicted2]), ignore_index=True)
    
#    from sklearn.linear_model import LogisticRegression
#    classifier = LogisticRegression()
#    classifier.fit(X_train,y_train)
#    predicted = sum(classifier.predict(X_test))
#    
#    df_pred = df_pred.append(pd.Series(['Logistic Regression' , predicted, predicted2]) , ignore_index=True)
#    
#    df_pred.columns =  ['Algorithm' , 'Predicted Runs' , 'Actual Runs']
    
    return df_pred
    
df_preds = score_predict('V Kohli' , team1 , team2 , match_type)
print(df_preds)


import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 4
predicted_runs = (list(df_preds.iloc[:,1]))
actual_runs = (list(df_preds.iloc[:,2]))
 
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
plt.xticks(index + bar_width, (list(df_preds.iloc[:,0])))
plt.legend()

plt.tight_layout()
plt.savefig("Algorith Comparisons.png")
plt.show()




#import matplotlib.pyplot as plt
#import pandas as pd
#from pandas.tools.plotting import table
#
#ax = plt.subplot(111, frame_on=False) # no visible frame
#ax.xaxis.set_visible(False)  # hide the x axis
#ax.yaxis.set_visible(False)  # hide the y axis
#
#table(ax, df_preds)  # where df is your data frame
#
#plt.savefig('mytable.png')