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
 - 08  Performance Analysis
 - 09  Conclusion
 - 10  References

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

For example, for "Street" variable (Type of road access to property), we can observe more than 99% observations concentrated in "Pave" type. These kind of situation was dropped from our main dataset once it couldn't explain house price.

<img src="https://i.ibb.co/44zmScC/Screenshot-at-jun-29-22-24-21.png" alt="Screenshot-at-jun-29-22-24-21" width="262" height="109">

So, the following numerical attributes were dropped: 'MSSubClass', 'MiscVal', 'PoolArea', 'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'KitchenAbvGr', 'BsmtHalfBath', 'LowQualFinSF', 'BsmtFinSF2'.

For categorical ones: SaleCondition','SaleType', 'PavedDrive', 'GarageCond', 'GarageQual', 'Functional', 'Electrical', 'CentralAir',
'Heating', 'BsmtFinType2', 'BsmtCond', 'RoofMatl', 'BldgType', 'Condition2', 'Condition1', 'LandSlope',
'Utilities', 'Street'

****

##  📌   DESCRIPTIVE ANALYSIS

In this section, the goal is studing our data using some basic descriptive measures. 
First of all, we took a look at our dependent variable. Although is not a perfect shape, it's close to a normal distribution, so, at first, we won't do any transformation

<img src="https://i.ibb.co/w4Jj0Fr/sale.png" alt="sale" width="485" height="267">

For our independent variables, we've splitted in categorical and numerical variables. 
For numerical variables, creating a Pandas DataFrame, we glance at some descriptive analysis such mean, median, standard desviation, min and max value and it range. The result is the following table:

<img src="https://i.ibb.co/xqX133G/Screenshot-at-jul-07-22-54-04.png" alt="Screenshot-at-jul-07-22-54-04" width="492" height="284">

Still in numerical exploration, we also plotted some charts, especially scatterplot in order to look at our data behavior and check if there's some concentration somewhere in our axis.

For categorical attributes, we mainly visualized our data using boxplot chart.

Concluding this section, we had made some important observations, as follows:

Numerical Variable Analysis

- Some numerical variable, although is int type, have a categorical behavior, as MoSold, YrSold or GarageCars, so, a scatterplot it's not usefull for meaninfull analysis. We're going to plot a boxplot chart for then.

- For other variables, we can notice some linear behavior for some dependent and independent variables, and some not.

- LotArea, YearRemodYear, YearBuilt, some variables regarded to basement característics are not so clear to explain sale price.

- For our numerical variables plotted in boxplot charts, datetime caracteristics are not so relevant. There're some differences, but not so meaninfull, specially when comparing to other variables of our dataset.

Categorical Variable Analysis

- Most of our categorical variables looks like relevant for our analysis. LostConfig and LotShape, for instance, have some differences, but very subles differences. Possibly, it won't make huge difference. Once again, variables regarded to basement caracteristics don't have many statistical differences.

***

##  📌   FEATURE ENGINEERING

We can divide the work in this section in two steps:

1. Variable transformation

   Adding, replacing and dropping some features.

2. Hypothesis

   In this "subsection", we've created some hypothesis in order to analyze our data, examine correlation and look over which features are more and less important to our Machine Learning model:

<hr>
REMODELED HOUSES

  A) Remodeled houses cost more
  
   Remodeled houses dont' cost more.

   Mean Price:
    
   Modeled Houses: 179096.307471

   Not modeled Houses: 182583.659686

   Conclustion: Variable not so relevant for our model.

  B) Remodeled houses have a better quality condition and then, cost more

   There's a difference, specially for rating 9, but it's not possible to consider as a importante variable.
<hr>
GARAGE

  A). Houses with more car capacity cost more

  B). Houses with no car space cost less

   Sale price increase if there’s more garage space until 3 garages cars. Houses with capacity for 4 cars have it’s sale price lower.

   This variable was considered as importante for our ML model.

   III. Houses with more than one type of garage cost more

   There’s big differences among the garage type. Carport garage’s have the lowest price, even considering it’s outlier.
<hr>
BASEMENT

  A) Houses with unfinished basement cost less

  This hypothesis is not true. There’s a difference regarding to price sale, especially for basement classified in GLQ (Good living quarters) category. All categories left don’t change too much.

  We're going to determine if this variable is importante for our ML model in other analysis later. (Not sure about it’s impact)

  B). Houses with bad basement quality have worst quality rating and cost less 

  Houses classified as Excellent and Fair have it’s price more distributed than good and typical. Same for basement type.

  We're going to determine if this variable is importante for our ML model in other analysis later. (Not sure about it’s impact)

<hr>
KITCHEN

   A) Houses with good quality rating cost more

  Kitchen classified in “Excellent” category have it’s price higher while kitchen with “Fair” category have it’s price lower.
<hr>
BATH

A) Houses with full bath + half bath cost more

Full bath have more impact to house price than Half bath. Considering both (half + full bath), we can see a huge price difference. This variable is importante for our ML model.
<hr>
LOT CONFIGURATION AND AREA

A) Houses with more than one frontage (entry) cost more

B) Irregular lots shape cost less

There’s not big price difference among log configuration categories. This variable is not importante for our ML model.

Lot shape, despite it’s small difference, won’t be considered as importante for our machine learning model.
    
**Univariate and Multivariate Analysis**

Still in feature engineering section, we analyzed some correlation between numerical variables.
For categorical variables we had some boxplot charts.

***

##  📌   DATA PREPARATION

In this step, we've prepared our data to better perform in Machine Learning models. Normalization and standardization techiniques were applied in numerical variables.

For categorical variables, label and ordinal encoders were used in order to transform them.

***

##  📌   FEATURE SELECTION
  For feature selection, in the first instance, we considered the previous data analysis. 
Then, implementing Boruta Algorithm, we had as result some features considered as important for the Machine Learning model. 

Note: Boruta algorithm is a wrapper based on random forest classification algorithm. 

In our previous analysis, we've considered the following features as important:
KitchenQual | Kitchen quality
FullBath | Full bathrooms above grade
HalfBath | Half baths above grade
Variables related to garage (garage type, garage area, car space)

Features considered as not important:
LotArea
Lot Shape
Variables related to area and lot shape
Variables related to house modelling.

Comparing Boruta result to our data analysis, we selected some variables for our Machine Learning model, as follows: Neighborhood, KitchenQual, LotFrontage, OverallQual, BsmtFinSF1, 1stFlrSF, 2ndFlrSF,GrLivArea, GarageArea, OpenPorchSF, TOTAL_BATH, GarageFinish_Unf, OverallCond, GarageType

***

##  📌   MACHINE LEARNING MODEL

Our first model was a average measure that was a "model" used as a baseline. In other words, the goal was to get a better model than the average values, otherwise, why should we manipulate, clean and do a bunch of analysis if a simple mean function would have a better performance?

Then, we have tried the following models:

- Linear Regression
- Linear Regression Regularized Model
- Random Forest Regressor
- XGBoost

All of these models (except for 'average model' we've trained with cross validation method.

We got different results:

|ModelName|MAE|MAPE|RMSE|
|-------|----|----|----|
|Random Forest Regressor|	17954.102066|0.107018|27806.906338|
|XGBoost Regressor|19921.151306|0.113741|32920.497776|
|Linear Regression|22590.355094|0.136715|35596.407737|
|Linear Regression - Lasso|22587.863747|0.136674|35600.977790|
|AVG Model|59303.878026|0.382768|83697.724296|

Looking at mean and standard deviation applyed in Cross Validation techniques, we had the following results: 

|LR_CV|LRR_CV	|RF_CV|XGB_CV|
|---|---|---|---|
|Mean|0.693962|0.694535|0.795081|0.741522|
|STD|0.237337|0.236138|0.153559|0.204619|


***

##  📌   PERFORMANCE ANALYSIS

Our first criterion to choose the best model was RMSE result (root-mean-square deviation), which Random Forest Regressor had the best one.

Secondly, analyzinh Cross validation results, we could see that Random Forest Regressor had the lower STD and therefore, the best result we were looking for.

Taking a look at our residuals, we got the following results:
Mean Residuals: 1351.79

Mean SalePrice: 178547.92

Mean Predictions: 177196.12

As we can notice by looking at these numbers, our residuals have a normal behaviour, exactly what we were expecting.

We can see its distibution more clearly in a histogram:

<img src="https://i.ibb.co/ccMvB6Z/Screenshot-at-Sep-27-21-05-54.png" alt="Screenshot-at-Sep-27-21-05-54" width="489" height="365">

Below, we have a line plot where we can analyze the differece between our predictions and the real price. We do have some differences (already expected) but in general, the values are pretty homogeneous.

<img src="https://i.ibb.co/sj3VCf9/Screenshot-at-Sep-27-21-06-07.png" alt="Screenshot-at-Sep-27-21-06-07" width="496" height="361">

##  📌   CONCLUSION

Finally, we can see our predictions and the real house price:

<img src="https://i.ibb.co/sFFjZ9q/Screenshot-at-Sep-27-21-13-26.png" alt="Screenshot-at-Sep-27-21-13-26" width="1008" height="249" data-load="full" style="">

***



