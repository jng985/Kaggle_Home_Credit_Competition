![](logo.png)

# Home Credit Default Risk

Home Credit Group is an international consumer finance provider that focuses on
responsible lending to people with little or no credit history. 

The necessary data for this project can be found at:
https://www.kaggle.com/c/home-credit-default-risk/data

# Problem 

The problem at hand is a binary classification task to answer the question of whether an applicant is capable of repaying a given loan based
on application, demographic, and historical credit behavior data.

# Data

![](dat.png)

# Final Model

After trying out various machine learning models, the best algorithm was LightGBM.

The final model contains 1252 features. A 5-fold cross validation training is performed to
obtain optimal hyperparameters. The average AUC score during this hyperparameter search was
0.788. The model is robust as there is little fluctuation in score as parameters are searched. The
hyperparameters for the Light GBM model are detailed below.

![](auc.png)


# Important Features

- The feature importances of the best 20 features used
by the final model.

![](imp.png)

---

# Project Report

![](pngs2/01.png)
![](pngs2/02.png)
![](pngs2/03.png)
![](pngs2/04.png)
![](pngs2/05.png)
![](pngs2/06.png)
![](pngs2/07.png)
![](pngs2/08.png)
![](pngs2/09.png)
![](pngs2/10.png)
![](pngs2/11.png)
![](pngs2/12.png)
![](pngs2/13.png)
![](pngs2/14.png)
![](pngs2/15.png)
![](pngs2/16.png)
![](pngs2/17.png)
![](pngs2/18.png)

