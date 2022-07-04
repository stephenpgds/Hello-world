import numpy as np
import pandas as pd
from  sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB

#create some data

weather =pd.Series(['sunny','rainy','snowy','cloudy','cloudy','snowy','sunny','rainy','sunny','cloudy','snowy','sunny','rainy','cloudy','snowy'])
attend = pd.Series(['yes','no','no','yes','no','yes','yes','yes','yes','yes','no','no','no','yes','no'])

# create pandas dataframe and rename columns
globo_df_long = pd.concat([weather, attend], axis=1)
globo_df_long.columns = ['weather','attended']

print(globo_df_long.head())

#create frequency table
print(pd.crosstab(index=globo_df_long['weather'],columns =globo_df_long['attended']))

# encode the feature

oe_weather = preprocessing.OrdinalEncoder()
le_attend = preprocessing.LabelEncoder()

#ordinal encoder for weather feature

oe_weather.fit(np.array(globo_df_long['weather']).reshape(-1,1))
weather = oe_weather.transform(np.array(globo_df_long['weather']).reshape(-1,1))

# label encoder for target

le_attend.fit(globo_df_long['attended'])
attend_y = le_attend.fit_transform(globo_df_long['attended'])

# view encoding
print(oe_weather.categories_)

print(le_attend.classes_)

# set and fit classifier

clf = CategoricalNB()
clf.fit(weather,attend_y)

# predict and view any given weather

weather_classes = np.unique(weather)

for i in weather_classes:
    print("weather",i,"_","attendence probability:",np.round(clf.predict_proba([[i]]),2),"predicted attendence:",clf.predict([[i]])[0])