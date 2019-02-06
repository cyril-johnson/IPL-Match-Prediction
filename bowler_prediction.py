# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:14:45 2018

@author: Sameer
"""

import pandas as pd
import numpy as np

bowler = 'R Ashwin'
df_stats=pd.read_csv("final_all_3.0.csv")

team_1= 'India'
team_2 = 'Australia'
match_type='ODI'

def bowler_stats(team_1,team_2,bowler,match_type):
    algo_compare = pd.DataFrame()    
    bowlers_data = pd.DataFrame()    
    team_12='['+"'"+team_1+"'"+', '+"'"+team_2+"'"+']'
    team_21='['+"'"+team_2+"'"+', '+"'"+team_1+"'"+']'
    df_match_type=df_stats[df_stats['info.match_type']==match_type]
    df_req=df_match_type[((df_match_type['info.teams']==team_12) | (df_match_type['info.teams']==team_21))]
    match_index = df_req['index_all'].unique()
    for mindex in match_index:
        bowler_game = []
        df_match = df_req[(df_req['index_all']==mindex)].copy()
        
        if df_match[(df_match['bowler']==bowler)].empty:
            continue
        

        runs = df_match[(df_match['bowler']== bowler)]['runs.total'].sum()
        wickets = df_match[(df_match['wicket.kind']!= 'run out') & (df_match['bowler']== bowler)]['wicket.kind'].count()
        no_of_balls = df_match[df_match['bowler']==bowler]['over_no'].count()
        overs = len(df_match[(df_match['bowler']==bowler)]['over_no'].unique())
        print("match index =",mindex)
        bowler_game.append(mindex)
        bowler_game.append(runs)
        bowler_game.append(wickets)
        bowler_game.append(overs)
    
        bowlers_data = bowlers_data.append(pd.Series(bowler_game), ignore_index=True)
    bowlers_data.columns = ['Index','Runs','Wickets','Overs']



    from sklearn.naive_bayes import GaussianNB

    A = bowlers_data.loc[:, ['Runs' , 'Overs']]#.values
    B = bowlers_data.loc[:, 'Wickets']#.values

    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.25,random_state = 2)
    y_train=np.ravel(y_train)
    model = GaussianNB()
    
    model.fit(X_train, y_train)
    predicted= model.predict(X_test)
    test = list(y_test['Wickets'])
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(test,predicted)
    
    algo_compare = algo_compare.append(pd.Series(['Naive Bayes' , predicted, test, accuracy]), ignore_index=True)
 

    
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(random_state = 1)
    classifier.fit(X_train,y_train)
        
    predicted= classifier.predict(X_test)
   
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(test,predicted)
    
    algo_compare = algo_compare.append(pd.Series(['Decision Tree' , predicted, test, accuracy]), ignore_index=True)
     
    

    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear')
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(test,predicted)
    
    algo_compare = algo_compare.append(pd.Series(['SVM' , predicted, test, accuracy]), ignore_index=True)
     

    
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 30, random_state = 1)
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(test,predicted)
    
    algo_compare = algo_compare.append(pd.Series(['Random Forest' , predicted, test, accuracy]), ignore_index=True)
 

    return algo_compare
    
    
try1=bowler_stats('India','Australia',bowler, match_type)
import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 4
accuracy = (list(try1.iloc[:,3]))

 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, accuracy, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Accuracy in algo\'s')
 
plt.xlabel('Prediction Type')
plt.ylabel('Accuracy')
plt.title('Accuracy in algorithm')
plt.xticks(index + bar_width, (list(try1.iloc[:,0])))
plt.legend()

plt.tight_layout()
plt.savefig("Algorith Comparisons.png")
plt.show()
