# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:03:37 2018

@author: dbda
"""

import pandas as pd
import numpy as np

df_stats=pd.read_csv("real_final_stats_2.csv")
#df_stats.head()
df_score=pd.read_csv("real_final_scores_2.csv")
#df_score.head()

def df_func(team_1,team_2,match_type='ODI'):
    team_12='['+"'"+team_1+"'"+', '+"'"+team_2+"'"+']'
    team_21='['+"'"+team_2+"'"+', '+"'"+team_1+"'"+']'
    df_match_type=df_stats[df_stats['info.match_type']==match_type]
    df_req=df_match_type[((df_match_type['info.teams']==team_12) | (df_match_type['info.teams']==team_21))]
    homeGround=[]
    #df_exp=pd.DataFrame()
    for i,j in df_req.iterrows(): 
        if j['info.neutral_venue']==1:
            homeGround.append(0)
        elif j['info.teams']==team_12:
            homeGround.append(1)
        else:
            homeGround.append(-1) 
    #print(homeGround)
    firstBat=[]
    for i,j in df_req.iterrows():
        if j['info.toss.winner']==team_1:
            if j['info.toss.decision']=='bat':  
                firstBat.append(1)
            elif j['info.toss.decision']=='field':
                firstBat.append(0)
        elif j['info.toss.winner']==team_2:
            if j['info.toss.decision']=='bat':  
                firstBat.append(0)
            elif j['info.toss.decision']=='field':
                firstBat.append(1)
        else :
            pass
    #print(firstBat)
    toss=[]
    for i,j in df_req.iterrows():
        if j['info.toss.winner']==team_1:
            toss.append(1)
        else:
            toss.append(0)
    target=[]
    for i,j in df_req.iterrows():
        if j['info.outcome.winner']==team_1:
            target.append(1)
        else:
            target.append(0)
    #print(target)
    df_res=pd.DataFrame(list(zip(target,firstBat,homeGround)),columns=['target','firstBat','homeGround'])
    df_res
    df_res['tossWin']=toss
    df_match_type=df_score[df_score['info.match_type']==match_type]
    df_req=df_match_type[((df_match_type['info.teams']==team_12) | (df_match_type['info.teams']==team_21))]
    ind=[]
    ind=list(df_req['index_all'].unique())
    #print(ind)
    #print(firstBat)
    totRuns=[]
    for i in ind:
        df_temp=df_req[df_req['index_all']==i]
        for f in firstBat:
            if f==1:
                df_temp1=df_temp[df_temp['team']==team_1]                
            else:
                df_temp1=df_temp[df_temp['team']==team_2]
        totRuns.append(df_temp1['runs.total'].sum())
    df_res['runs']=totRuns
    return df_res


#for desion tree
def df_tester_dectree(df):
    #select particular columns from the dataframe
    X = df.drop(['target'],axis=1)
    Y = df[['target']]
    #feature_name=X.columns
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15,random_state=1)
    #normalizing data
#    from sklearn.preprocessing import StandardScaler
#    sc = StandardScaler()
#    X_train = sc.fit_transform(X_train)
#    X_test = sc.transform(X_test)
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 1)
    classifier.fit(X_train,y_train)
    #predicting with test dataset
    y_pred = classifier.predict(X_test)
    #Accuracy Score
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    #print(accuracy)
    return accuracy
    

#random forest        
def df_tester_rf(df):
    #select particular columns from the dataframe
    X = df.drop(['target'],axis=1)
    Y = np.ravel(df[['target']])
    #feature_name=X.columns
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15,random_state=1)
    #normalizing data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) 
     #random forest
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(max_depth = 10, min_samples_split=3, n_estimators = 200, random_state = 1)
    classifier.fit(X_train,y_train)
    #predicting with test dataset
    y_pred = classifier.predict(X_test)
    #Accuracy Score
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    #print(accuracy)
    return accuracy



#naive bayes
def df_tester_naive(df):
    #select particular columns from the dataframe
    X = df.drop(['target'],axis=1)
    Y = np.ravel(df[['target']])
    #feature_name=X.columns
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15,random_state=1)
    #normalizing data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) 
     #random forest
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    #predicting with test dataset
    y_pred = classifier.predict(X_test)
    #Accuracy Score
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    #print(accuracy)
    return accuracy


#SVM
def df_tester_svm(df):
    #select particular columns from the dataframe
    X = df.drop(['target'],axis=1)
    Y = np.ravel(df[['target']])
    #feature_name=X.columns
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15,random_state=1)
    #normalizing data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) 
     #random forest
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    #predicting with test dataset
    y_pred = classifier.predict(X_test)
    #Accuracy Score
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    #print(accuracy)
    return accuracy

#logistic regression
def df_tester_logReg(df):
    #select particular columns from the dataframe
    X = df.drop(['target'],axis=1)
    Y = np.ravel(df[['target']])
    #feature_name=X.columns
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15,random_state=1)
    #normalizing data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) 
     #random forest
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train,y_train)
    #predicting with test dataset
    y_pred = classifier.predict(X_test)
    #Accuracy Score
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    #print(accuracy)
    return accuracy




def test_caller():
    teams=['Australia','New Zealand','Bangladesh','Sri Lanka','West Indies','South Africa','England','Pakistan','Zimbabwe']
    #team_1='India'
    df_acc=pd.DataFrame()
    df_acc['teams']=teams
    lst=[]
    for team in teams:
        df=df_func('India',team,'ODI')
        accuracy=df_tester_dectree(df)
        lst.append(accuracy)
    df_acc['Decision tree']=lst
    lst=[]
    for team in teams:
        df=df_func('India',team,'ODI')
        accuracy=df_tester_rf(df)
        lst.append(accuracy)
    df_acc['Random Forest']=lst
    lst=[]
    for team in teams:
        df=df_func('India',team,'ODI')
        accuracy=df_tester_logReg(df)
        lst.append(accuracy)
    df_acc['Logistic Regression']=lst
    lst=[]
    for team in teams:
        df=df_func('India',team,'ODI')
        accuracy=df_tester_naive(df)
        lst.append(accuracy)
    df_acc['Naive Bayes']=lst
    lst=[]
    for team in teams:
        df=df_func('India',team,'ODI')
        accuracy=df_tester_svm(df)
        lst.append(accuracy)
    df_acc['SVM']=lst
    lst=[]
    return df_acc
        
df_acc_temp_3=test_caller()


#df_acc_random_1.to_csv('Accuracy_Matrix.csv')


















