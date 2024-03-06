


from flask import Flask,render_template,request
import xgboost as xgb
import warnings
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt




def posix_time(dt):
    return (dt-datetime(1970,1,1))/timedelta(seconds=1)





data=pd.read_csv("Train.csv")

data.loc[data["is_holiday"]!='None','is_holiday']=1
data.loc[data["is_holiday"]=='None','is_holiday']=0
data["is_holiday"]=data["is_holiday"].astype(int)
    
data["date_time"]=pd.to_datetime(data["date_time"])
data["hour"]=data["date_time"].map(lambda x: int (x.strftime("%H")))
data["month_day"]=data["date_time"].map(lambda x: int (x.strftime("%d")))
data["weekday"]=data["date_time"].map(lambda x: x.weekday()+1)
data["month"]=data["date_time"].map(lambda x: int (x.strftime("%m")))
data["year"]=data["date_time"].map(lambda x: int (x.strftime("%Y")))


data.loc[ data["weather_type"]=='Clear','weather_type']=1
data.loc[data["weather_type"]=='Clouds','weather_type']=2
data.loc[ data["weather_type"]=='Snow','weather_type']=3
data.loc[data["weather_type"]=='Haze','weather_type']=4
data.loc[ data["weather_type"]=='Mist','weather_type']=5
data.loc[data["weather_type"]=='Fog','weather_type']=6
data.loc[ data["weather_type"]=='Drizzle','weather_type']=7
data.loc[data["weather_type"]=='Rain','weather_type']=8
data.loc[ data["weather_type"]=='Thunderstorm','weather_type']=9
data.loc[data["weather_type"]=='Squall','weather_type']=10
data.loc[ data["weather_type"]=='Smoke','weather_type']=11
data["weather_type"]=data["weather_type"].astype(int)


data.loc[ data["weather_description"]=='sky is clear','weather_description']=1
data.loc[ data["weather_description"]=='few clouds','weather_description']=2
data.loc[ data["weather_description"]=='scattered clouds','weather_description']=3
data.loc[ data["weather_description"]=='heavy snow','weather_description']=4
data.loc[ data["weather_description"]=='overcast clouds','weather_description']=5
data.loc[ data["weather_description"]=='broken clouds','weather_description']=6
data.loc[ data["weather_description"]=='haze','weather_description']=7
data.loc[ data["weather_description"]=='light snow','weather_description']=8
data.loc[ data["weather_description"]=='mist','weather_description']=9
data.loc[ data["weather_description"]=='fog','weather_description']=10
data.loc[ data["weather_description"]=='snow','weather_description']=11
data.loc[ data["weather_description"]=='drizzle','weather_description']=12
data.loc[ data["weather_description"]=='light intensity drizzle','weather_description']=13
data.loc[ data["weather_description"]=='moderate rain','weather_description']=14
data.loc[ data["weather_description"]=='light rain','weather_description']=15
data.loc[ data["weather_description"]=='heavy intensity drizzle','weather_description']=16
data.loc[ data["weather_description"]=='Sky is Clear','weather_description']=17
data.loc[ data["weather_description"]=='light rain and snow','weather_description']=18
data.loc[ data["weather_description"]=='proximity shower rain','weather_description']=19
data.loc[ data["weather_description"]=='proximity thunderstorm','weather_description']=20
data.loc[ data["weather_description"]=='heavy intensity rain','weather_description']=21
data.loc[ data["weather_description"]=='thunderstorm','weather_description']=22
data.loc[ data["weather_description"]=='thunderstorm with light rain','weather_description']=23
data.loc[ data["weather_description"]=='proximity thunderstorm with rain','weather_description']=24
data.loc[ data["weather_description"]=='proximity thunderstorm with drizzle','weather_description']=25
data.loc[ data["weather_description"]=='thunderstorm with rain','weather_description']=26
data.loc[ data["weather_description"]=='thunderstorm with heavy rain','weather_description']=27
data.loc[ data["weather_description"]=='thunderstorm with drizzle','weather_description']=28
data.loc[ data["weather_description"]=='freezing rain','weather_description']=29
data.loc[ data["weather_description"]=='shower drizzle','weather_description']=30
data.loc[ data["weather_description"]=='very heavy rain','weather_description']=31
data.loc[ data["weather_description"]=='sleet','weather_description']=32
data.loc[ data["weather_description"]=='light shower snow','weather_description']=33
data.loc[ data["weather_description"]=='shower snow','weather_description']=34
data.loc[ data["weather_description"]=='thunderstorm with light drizzle','weather_description']=35
data.loc[ data["weather_description"]=='light intensity shower rain','weather_description']=36
data.loc[ data["weather_description"]=='SQUALLS','weather_description']=37
data.loc[ data["weather_description"]=='smoke','weather_description']=38
data["weather_description"]=data["weather_description"].astype(int)



y=data["traffic_volume"]
y




print(data)

label_columns=["weather_type","weather_description"]
numeric_columns=["month_day","month","year","hour","temperature","weekday","is_holiday"]

features=numeric_columns+label_columns
x=data[features]
x.head()

type(x['weather_type'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
trainX,testX,trainY,testY=train_test_split(x,y,test_size=0.2)
xgbr = xgb.XGBRegressor(verbosity=0)


xgbr.fit(trainX, trainY)
 
score = xgbr.score(trainX, trainY)   
print("Training score: ", score) 

# - cross validataion 
scores = cross_val_score(xgbr, trainX, trainY, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, trainX, trainY, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
 

ypred = xgbr.predict(testX)
print(ypred)
from numpy import asarray
ip=[0,287.952,2,13,18,2016,10,1,17]
new_data=asarray([ip])
out=xgbr.predict(new_data)
out


import pickle
#filename='savemodel.sav'
#pickle.dump(xgbr,open(filename,'wb'))

#load_model=pickle.load(open(filename,'rb'))


    

pickle.dump(xgbr,open('traffic.pkl','wb'))
