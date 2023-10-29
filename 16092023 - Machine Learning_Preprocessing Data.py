print("DATA PREPROCESSING FOR MACHINE LEARNING")
#### Or
print("Data Cleaning for Machine Learning")
'''
1. Aim - what is the business objective: What kind of data you need as input and
  what is the expected outout.
2. Getting the data
3. Data cleaning (Preprocessing steps): we will focus on now
'''
'''
1. Aim - what is the business objective: What kind of data you need as input and
  what is the expected out.
2. Getting the data
3. Data cleaning (Preprocessing steps): we will focus on now
'''

'''
Library name: scikit-learn
pip install scikit-learn
This library will be used for all our Machine Learning work

Steps for data preprocessing:
1. Handle missing values: delete rows and columns which has many missing values. 
    when less missing values are there then we replace with mean and mode
    Outliers to be treated as missing values
2. Handling categorical value: 1) Encode  2) Column Transform   
    3) Delete any one column from column transform.(avoid dummy variable)
It means text data to be handled this way only.
3. Dividing the dataset into Training and Test set: 
4. Feature Scaling: Min-Max Normalization or Z-Score Normalization

'''

import pandas as pd
link = "D:/TOP MENTOR/MachineLearning-master/MachineLearning-master/1_Data_PreProcessing.csv"
df = pd.read_csv(link)
print(df)

X = df.iloc[:,:-1].values # Integer value is used iloc function and all rows and all columnns except the last one.
Y = df.iloc[:,-1].values # All rows and last column only.
print("Printing X =",X)
print("Printing Y =",Y)

# Step No. 1 is handling missing values:-
# 0 means region don't have missing values, so we should not worry about that only column 1 and 2 have missing values.
from sklearn.impute import SimpleImputer # SimpleImputer is for replace the missing value.
import numpy as np # for array and numbers
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# find np.nan (nan values) and will replace by mean. # for categorical we use mode.
imputer = imputer.fit(X[:,1:3]) #after creating the object, use fit function else use fit-trasform # Data is ready to transform
#  # all rows and 1 and 2 not 3 column, Left side is always included, and right side is always upto while indexing.
X[:,1:3] = imputer.transform(X[:,1:3])
print("New X with the replaced value for column 1 and 2 is ", X)

#2 Handling categorical value: -
print("Before Handling: \n",X[:,0]) # for text only
# To encode we call label encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# OnehotEncode will use to convert into 4 different columns just after encoding.
lc = LabelEncoder()
X[:,0] = lc.fit_transform(X[:,0])
print("After encoding: \n", X[:,0])

# Now encoding is done, column transform to be performed.
from sklearn.compose import ColumnTransformer
trans = ColumnTransformer([("one_hot_encoder", OneHotEncoder(),[0])], remainder="passthrough")
X = trans.fit_transform(X)
print("After Encoding region column will be divided into 4 columns and we used 0 as doing for Region column", X)

# Now we need to delete one column:
X = X[:,1:] #Except 0, everything I need, deleting one column out of 4
print(X)

## Similarly we need to handle Y values for classification problems:
# 3rd Step is :- Break it into train and test set
from sklearn.model_selection import train_test_split
# And this will divide one dataset into 4 parts X = Xtrain and Xtest
# And Y =  Y train and Ytest
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)
# It means 25% of the data or rows for test and 75% of the data or rows given for training.
# Suppose we have 100 rows, so x-test will have 25 rows and 75 will have train test.

# 4th Step is to handle Scaling or normalization:-
from sklearn.preprocessing import StandardScaler  # do scaling for us
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print("X_train: \n", X_train)
print("X_test: \n",X_test)
print("y_train: \n",y_train)
print("y_test: \n",y_test)


# We have not done encoding for Y that is why showing yes, no, yes, no.
# When we have 2 columns and have values of different range (sales person and Quotation)
# then only we do scaling since X has 2 columns
# And y only has one column which is Yes, No value, categorical x.

print("Now Understanding Machine Learning Algoritham")
'''
Linear regression is all about the straight line. and the line we draw, how do we know that is the best fit line or not?
and dont know even which is the best regression line. we end up drawing multiple lines, and we calculate the average 
error or zero error (means the distance from all the points to the line should be in a way where we add them up altogether and it should
be minumum or zero, then only it is called the best fit line or Regression line. 
The line which give least amount of error is called the regression line.

Once we find the regression line then easy to predict the things like for number of hours you study, can predict
to get the marks or these are the marks expected to get.

Regression line is found using TRIAL & ERROR method.
in this we need to fix either m or C. Suppose we fix it at 45 angle - m value. And see which of these is giving least error.
The points where we get the list error is called intercept value. 

Once we fix the intercept, keep changing the angle. so by changing the angle we try to find out the best fit line.
By changing the every angle, The error will start decreasing. 

Once the regression line or best fit line created. it means the model is learnt means trained already.
Chat Gpt model has learnt, trained already means regression line created. we know the regresson line means training is done.

You trained the model well, it will predicted auto. Best fit line is brown line and blue line is for rejection.

NOTES2:-
In linear Regression each and every line has an equation. Y = mx + c
Where c is intercept or constant.
m is the slop or angle
Each line will have fixed but different m and c value. 
Linear Regression is all about finding m which is slope and C constant value.
Change in Y axis and X axis is called Delta: 

At 45 degree, your delta X and delta y will be same. when both same means slope is 1.
AS we go above 45 degree, your slope will become more than 1.
And below 45 degree, slope will become less than 1.

The step are :-
 First fine best fit line
 find the value of m and c
 and then find for every x, there is y value.   

Simple Linear Regression: One column in X is called Simple Linear Regression. and hear we deal with bi-variant problems:
                        2 columns means x and y.    
'''

'''
MAIN NOTES:-

First we need to draw the scatter plot.
Second figure out the relationship between them or not.
Third, if relationship, then need to find out the regression line
Fourth, once we find the regression line, job is done.
Regression line means we got the prediction line
Whatever value we put as input, will get the predicted value for the same.. means output for Y.
'''

# Regression : is a Supervised Learning where we need to predict a numerical value.
# X - Features, input. Features we take which are relevant.
# Y - Target, Output.
# Based on history data, we put Regression algoritham
# Simple Linear Regression
# No data cleaning or preprocessing required for this. like missing value, scaling,


## Simple linear Regression Process after 16092023 code.

link = "D:\\TOP MENTOR\\Dataset-master\\Dataset-master\\student_scores.csv"
import pandas as pd

df = pd.read_csv(link)
print(df)

# EDA - Exploratory Data Analysis:-
# Will find out if there is realtionship in x and y or not?
import matplotlib.pyplot as plt

plt.scatter(df["Hours"], df["Scores"])  # Scores need to predict so on y axis and hours will be given so put on X.
plt.show()

# so the conclusion is by the scatterplot, that x value and y value wll be increasing together.
# The number of hours study will be impacting on the scores which we got. There is a correlation.

# Perform the correlation now. will divide into training set and test set.
# Before that will find out what is X and What is Y?
X = df.iloc[:, :1].values
Y = df.iloc[:, 1].values
print("X vaue is \n", X)
print(("Y value is \n", Y))

from sklearn.model_selection import train_test_split  # This is for supervised learning, when we have already data.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=100)

# Preprocessing steps are done. and now we need to run the model.
## Run the model - Simple Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)  # will create our regression line.
# Cofficinet - m and Intercept means C in machine learning.
print("Cofficeint = ", regressor.coef_)
print("Intercept = ", regressor.intercept_)

'''
Cofficeint =  [9.83055209]  =m # we get one cofficeint as we have one x else more will get.
Intercept =  4.041462933688052 =c 
Everytime we run, will get different value for m and c. Because when we split the data in training and test set
we have never fixed what 25% data will be used for test and 75% for training. So it will change auto by changing
the value of training set and test set.

To fixing the dataset for both we use random_state with split function.
y = mx + c
y = 9.8X + 4.04 # it means if I study 10 hours a day, I can expect 100 marks that's prediction.

For random_state = 100 fixing the values of m and c
Cofficeint  =  [10.24428684]
Intercept =  2.86908983451535
Regression line means y = mx + c
y = 10.24X + 2.86c
'''

# EVALUATING THE PERFORMANCE OF TRAINING DATA ITSELF.
y_pred = regressor.predict(X_train)  # from here code need to understand.
result = pd.DataFrame({"Actual = ": Y_train, "Predicted = ": y_pred})
print(result)

# Evaluating the model as training is done.
from sklearn import metrics

mae = metrics.mean_absolute_error(Y_train, y_pred)
print("TRAINING / BIAS : Mean Absolute Error = ", mae)
# Mean absolute error is one of the accepted value as error. This is one way to convert negative to positive.

mse = metrics.mean_squared_error(Y_train, y_pred)
print("TRAINING : Mean Squared Error = ",
      mse)  # This is not preferred method, as we started with and ended with Squared only.

rmse = mse ** 0.5
print("TRAINING :Root Mean Squared Error =", rmse)

r2 = metrics.r2_score(Y_train, y_pred)
print("TRAINING: R Squared Value = ", r2)

# So far we have trained the model. Now we are testing the model.
### testing the model with test or validation data

y_pred = regressor.predict(X_test)  # from here code need to understand.
result = pd.DataFrame({"Actual = ": Y_test, "Predicted = ": y_pred})
print(result)

# Evaluating the model as testing is done.
from sklearn import metrics

mae = metrics.mean_absolute_error(Y_test, y_pred)
print("VALIDATION / VARIANCE : Mean Absolute Error = ", mae)
# Mean absolute error is one of the accepted value as error. This is one way to convert negative to positive.

mse = metrics.mean_squared_error(Y_test, y_pred)
print("VALIDATION : Mean Squared Error = ",
      mse)  # This is not preferred method, as we started with and ended with Squared only.

rmse = mse ** 0.5
print("VALIDATION : Root Mean Squared Error =", rmse)

r2 = metrics.r2_score(Y_test, y_pred)
print("VALIDATION : R Squared Value = ", r2)

'''
TRAINING: Mean Absolute Error =  6.5463373391121635
TRAINING : Mean Squared Error =  84.80772672051485
TRAINING :Root Mean Squared Error = 9.20911107113574
TRAINING: R Squared Value =  0.8674271500578972

Mean Absolute Error =  5.6521375098502675
Mean Squared Error =  41.23135017357897
Root Mean Squared Error = 6.421164238171998
R Squared Value =  0.8704371576025407  # This is closure to 1 means near by the best fit line.
'''
# R Square - 1-error with regression model.
'''
R Square = 1 - Regression Error / Average Error
Lets say you have 0 error means all the points on the regression line. you got the best line.

R Square for Best fit model = 1 - 0/some_error = 1 here some error is 0
R Square for worst case = 1 - average error = 0 where regression error = avgerror as no point are there on the line
                    so not able to draw regression line, we dont have regressin line.
                      so count only average line. there is no correlation.
Where avgerror =1

It means R Square value range is from 0 to 1.

NEXT ALGORITHAM TO UNDERSTAND THIS CONCEPT                 
Note: when error is high, it means accuracy is very low
Whenever Accuracy is high, Error is very low.

Training error is also called BIAS
Validation is also called VARIANCE or called TEST error.

WHEN KEEP ON INCREASING THE DATA OR  KNOWLEDGE, TRAINING DATA WILL GO BIAS MEANS WILL GO DOWN.

THE AREA IN CENTER OR BETWEEN UNDERFITING AREA OR OVERFITING AREA IS CALLED BEST FIT AREA, THE BEST VALUE WE GET THERE
IT MEANS THE MORE DATA WE PROVIDE, WILL START DECRESAING TRAINING ERROR AND HIGHING ACCURAY AND DECREASING TESTING ERROR WITH THAT.

BIAS AND VARIANCE BOTH ARE HIGH IN UNDERFITING AREA.
BIAS IS LOW AND VARIANCE IS HIGH IN OVERFITING AREA.

Bias is with your training data and Variance is your test data or validation data.

r2 =0 means regression line = avegrae line, no diffrence at all
r2 =1 mens there is no error perfect regression line. it is the best fit line where all the point
are lies on the regression line.

We want mae, mse, rmse, r2 these all closer to zero. means no erro, erros as closes to be zero.
R2 irrestive of any dataset will be between 0 to 1. other than that will changes but wnat to be nearer to zero.
so the error can be minimized,

mae, mse, rmse for same data set
r2 irrestive of any data set

'''






















