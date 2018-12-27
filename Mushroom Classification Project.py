import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('mushrooms.csv')
# print(df.head())
# print(df.info(max_cols=120))

for i in df.index:
    outcome = df.get_value(i, 'class')
    if outcome == 'p':
        df.set_value(i, 'class', 1)
    else:
        df.set_value(i, 'class', 0)

df['class'] = pd.to_numeric(df['class'])

df = pd.get_dummies(df, columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat'])

X_train, X_test, y_train, y_test = train_test_split(df.drop('class',axis=1),
                                                    df['class'], test_size=0.30,
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))
