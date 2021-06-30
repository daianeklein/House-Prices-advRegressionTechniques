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
  
  Finally, several machine learning models were trained, starting with a average model as a baseline, following by some regression models and at last, Random Forest Regressor chosen for it's best score.
  
  All these steps are explained better in following sections.


****

## 📎   TABLE OF CONTENTS

 - 01  Personal growth and skill development 
 - 02  Dataset info - Data types, missing values and dimension analysis
 - 03  Descriptive analysis
 - 04  Feature Engineering
 - 05  Data Preparation
 - 06  Feature Selection
 - 07  Machine Learning Model
 - 08  Hyperparameter Tuning
 - 09  Performance Analysis
 - 10  Conclusion
 - 11  References

****

## 💡 Personal growth and skill development

****

##  📌   DATASET INFO

### Missing Values

  This dataset contains a lot of missing values scattered in almost 20 different variables. With the aim of understanding the exactly quantity and its impact, the table below was created:
  
  <img src="https://i.ibb.co/Fs1DRdY/Screenshot-at-jun-28-21-40-09.png" alt="Screenshot-at-jun-28-21-40-09" width="224" height="473">
  
  As we can see, variable named PoolQC (Pool quality) has 1.453 null values, which corresponds to 99.52% of our entire dataset.
  It's similar to MiscFeature (Miscellaneous feature not covered in other categories) with 96.30% missing values, Alley (Type of alley access to property) with 93.76% and so on...
  
  Intending to deal with then, we adopt the following criteria:
  
  First of all, we had checked the percentage of null values for each variable.

1. Variables with more than 40% null values were dropped from our dataset.

    PoolQC 99.52%

    MiscFeature | 96.30%

    Alley | 93.76%

    Fence | 80.75%

    FireplaceQu | 47.26%

2. Second step, we filled null values using Pandas method: ffill for all variables between 1 and 18% null values.

3. Last step, all missing values left were replaced by it's mode.
  
***

### Dimension Analysis

As said above, this dataset have almost 80 differents variables. To avoid overfitting and to make the analysis easier, better to undertand and most importantly, build a machine learning model able to generalize new data, we've worked in dataset dimension.

With a declared function, splitted in categorical and numerical variable, we could look at each value and the relative frequencie of then.

<img src="https://i.ibb.co/716pKwm/Screenshot-at-jun-29-22-25-44.png" alt="Screenshot-at-jun-29-22-25-44" width="875" height="284" data-load="full" style="">

For example, for "Street" variable (Type of road access to property), we can observe more than 99% observations concentrated in "Pave" type. These kind of situation were dropped from our main dataset once it couldn't explain house price.

<img src="https://i.ibb.co/44zmScC/Screenshot-at-jun-29-22-24-21.png" alt="Screenshot-at-jun-29-22-24-21" width="262" height="109">

So, the following numerical attributes were dropped: 'MSSubClass', 'MiscVal', 'PoolArea', 'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'KitchenAbvGr', 'BsmtHalfBath', 'LowQualFinSF', 'BsmtFinSF2'.

For categorical ones: SaleCondition','SaleType', 'PavedDrive', 'GarageCond', 'GarageQual', 'Functional', 'Electrical', 'CentralAir',
'Heating', 'BsmtFinType2', 'BsmtCond', 'RoofMatl', 'BldgType', 'Condition2', 'Condition1', 'LandSlope',
'Utilities', 'Street'

****

##  📌   DESCRIPTIVE ANALYSIS
