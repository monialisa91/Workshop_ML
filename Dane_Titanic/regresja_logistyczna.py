#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# PRZEWIDYWANIE CZY OSOBA UMARŁA CZY NIE PODCZAS KATASTROFY TITANICA

# DANE z kaggle.com

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# 1. Zapoznanie się z danymi -- wizualizacja

train = pd.read_csv('Dane_Titanic/train.csv')

train.head()
train.tail()

# Pclass - klasa pasażerska
# SibSp - rodzeństwo + małżonek
# Parch - rodzice i dzieci
# cabin - numer kabiny 
# Embarked - miejsce wejścia na pokład:
#   S- Southampton
#   C-Cherbourg
#   Q-Queenstown

# brakujące dane

train.isnull()

sns.heatmap(train.isnull(), cmap = 'viridis', cbar = False, yticklabels = False)

sns.set_style('whitegrid')

# ilość osób które przetrwały i osób które poniosły śmierć
sns.countplot(x='Survived', data = train)

# Rozróżnienie ze względu na płeć
sns.countplot(x='Survived', hue = 'Sex', data = train, palette = 'RdBu_r')

# Rozróżnienie ze względu na wybraną klasę
sns.countplot(x='Survived', hue = 'Pclass', data = train)

# Rozróżnienie ze względu na SibSp
sns.countplot(x="SibSp", data = train)

# Rozkład wieku pasażerów
sns.distplot(train['Age'].dropna(), kde = False, bins = 30)

# Rozkład opłaty za podróż
sns.distplot(train['Fare'], kde = False)

# 2. Czyszczenie danych

# a) Uzupełnienie danych wiekowych
# srednia wieku osob w danej klasie

sns.boxplot(x = 'Pclass', y = 'Age', data = train)

def imput_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if(pd.isnull(Age)):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else: 
            return 24
    else:
        return Age
            
#means = train.groupby('Pclass')['Age'].mean() # verification

train['Age'] = train[['Age','Pclass']].apply(imput_age, axis = 1)

# sprawdzenie nowych danych
sns.heatmap(train.isnull(), cmap = 'viridis', cbar = False, yticklabels = False)

train.drop("Cabin", axis = 1, inplace = True)

sns.heatmap(train.isnull(), cmap = 'viridis', cbar = False, yticklabels = False)

# usuwanie brakujacych danych

train.dropna(inplace = True)

# 3. Przerabianie danych

sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)

train = pd.concat([train, sex, embark], axis = 1)

train.drop([ 'PassengerId', 'Sex','Embarked', 'Name', 'Ticket',  ], axis = 1, inplace = True)


# 4. Modelowanie

X = train.drop('Survived', axis = 1) # cechy
y = train['Survived'] # przewidywana wartosc

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

dokladnosc = 1.0 * (cm[0][0] + cm[1][1])/(np.sum(cm))











