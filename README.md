# HOUSE PRICES - ADVANCED REGRESSION TECHNIQUES

<img width="833" height="142" style="display: block; margin-left: auto; margin-right: auto" src="https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png">

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

<a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview">Kaggle Competition</a>

****

## 🔍 ABOUT THIS PROJECT

  The aim of this project is to analyze a dataset in order to develop a machine learning model and predict house's sale price.
  The dataset is available in a "Getting Started Competition" in Kaggle (link available above).
  
  With 79 variables, the main point was selecting the most important caracteristics to build our Machine Learning Model. I've been through data analysis, worked with null values, features select and engineering.
  
  The final Machine Learning model was Random Forest Regressor, despite i've trained Simple Linear Regression, Linear Regression Regularized, XGBoost Regressor and some more. All these models were trained with cross validation technique for the purpose of avoiding overfitting.

## 📚 HOW THIS NOTEBOOK IS ORGANIZED
  This notebook is based on a structured sequence: starting from data analysis, understanding the dataset as number of rows and it's dimensions, identifying with the main features and what they stand for, comprehending how the attributes are correlated and specially how they can explain the house price. 
  
  There were an extensive section to get familiar with these informations. To make it clear, there're were some hypothesis based on house's caracteristics (note: the hypothesis testing was not a statistical method. It was used as a "guide" for our data analysis. There's a dedicated section about it, you can check it out here: Mind Map for Data Analysis). 
  
  Still about the dataset undertanding, due to it dimensions (almost 79), an analyze was needed in order to identify which features could be unnecessary. In this step, after have knowledge of the attributes, we had moved to feature engineering.
  
  There was a good work in feature selection. As support, Boruta algorithm was implemented. 
