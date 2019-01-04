# Mushroom-Classification
My first machine learning exercise to classify if a mushroom is poisonous or edible (data from Kaggle.com)

It uses a Linear Regresion model as a binary classifer.

My approach was very simple - I converted the existing class values to 1 or 0, and then used the Pandas get_dummies method to convert categorical variables to numerical.

Please do not hesitate to let me know if something can be improved. 

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1274
           1       1.00      1.00      1.00      1164

   micro avg       1.00      1.00      1.00      2438
   
   macro avg       1.00      1.00      1.00      2438
   
weighted avg       1.00      1.00      1.00      2438

