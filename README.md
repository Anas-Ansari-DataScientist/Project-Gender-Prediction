# Predicting Gender

## Introduction:-

In this notebook, I'm going to do an **end to end project of Machine Learning Project** in which we will take height(in feet) and weight(in kgs) as an input and will predict the gender of that particular person. I'm going to go over a basic Machine Learning pipeline from start to finish to show what a typical data science workflow looks like.

The dataset for this project was collected from **Kaggle.com**.The dataset contains three columns one weight one of height and labeled as male and female as the third column. The shape of the dataset is (10000,3).

In this project i have used 4 Machine learning algorithm.

**-KNN**

**-Random forest**

**-Decision Tree**

**-Logistic Regression**

For the purposes of this project, the following preprocessing steps have been made to the dataset:

-The 7 rows of data with a value of [ 994, 1317, 2014, 2014, 3285, 3757, 6624, 9285, 9285] which is totally different from the dataset. These data point can be considered as outliers and has been removed.

-The given data of height and weight were in a unit of centimeter and pound which is not popular in India so these data points are converted to units that are popular in India which is kilogram and feet.
To test my model I have deployed my model using **Flask framework** into **Heroku** environment.

## link to test my model is-

https://genderprediction-api.herokuapp.com

***Go check out my model.***

![alt text](https://github.com/Anas-coder/Project-Gender-Prediction/blob/master/Screenshot%20(7).png
)



![output](https://github.com/Anas-coder/Project-Gender-Prediction/blob/master/Output.png)


***I would also like to add a little description of my other files:-***

**Templates**:-This folder contains the index.html file which is basically the homepage of my app.

**Procfile**:-This file defines that which first I want to execute first in my case its app.py

**app.py**:-This files create a web app using the Flask framework.

**model.pkl**:-model needs to be dumped in the form of an extension and is dumped as model.pkl file

**requirements.txt**:-This file defines the library that has to get install in my environment when I am deploying into Heroku


### It's just an approach.
## I am still learning!
