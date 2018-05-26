#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 22:28:24 2018

@author: monisia
"""

# Zaimportuj wg Ciebie potrzebne biblioteki

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj odpowiednie dane (advertising.csv) w formacie tablicy danych z modulu Pandas
# Nazwij tablicę danych ad_data

ad_data = pd.read_csv('advertising.csv')

# Sprawdź jak wygląda pierwsze kilka linijek danych

ad_data.head( )

# Użyj metod info() i describe() na swoich danych

ad_data.info()
ad_data.describe()

# Stwórz histogram wieku internauty

sns.distplot(ad_data['Age'], kde = False, bins = 30)
ad_data['Age'].plot.hist(bins = 30)

# Stwórz jointplot (z pakietu seaborn) kolumn 'Area Income' i 'Age'

sns.jointplot(x= "Age", y ="Area Income", data = ad_data)

# Stwórz jointplot (z pakietu seaborn) kolumn 'Age' i 'Daily Time Spent on Site' użyj opcji kind = 'kde'

sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, kind = 'kde', color = 'red')

# Stwórz jointplot (z pakietu seaborn) kolumn 'Daily Time Spent on Site ' i 'Daily Internet Usage' użyj opcji kind = 'kde'

sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data)

# Stwórz pairplot (z pakietu seaborn) z hue = 'Clicked on Ad'

sns.pairplot(ad_data, hue = 'Clicked on Ad')

# Podziel dane na część testową i treningową

from sklearn.model_selection import train_test_split

X_raw_data = ad_data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis = 1 )

X = X_raw_data.drop(['Clicked on Ad'], axis = 1 )
y = X_raw_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# Stwórz estymator, dopasuj dane i sprawdz przewidywania na zbiorze testowym

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

# Stwórz confusion_matrix na podstawie której oszacuj dokładność modelu

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

dokladnosc = 1.0 * (cm[0][0] + cm[1][1])/(np.sum(cm))
