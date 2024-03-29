# Forecasting a Carbon Zero City
 
![Carbon Zero City Map](images/CO20_map.jpg)

## Project Goal

The project goal is to help energy efficiency professionals identify the best energy efficiency projects in a city, and for anyone to see how making these changes can impact a city’s carbon zero goal.

## Context

Energy efficiency improvements for grid electrical consumption is a core strategy in Gainesville’s Climate Action Plan for achieving carbon zero goal by 2035. This dashboard aligns with Gainesville’s long-term carbon zero goal by tracking how 5,384 energy efficiency improvements impact the city's KPI for grid-electrical emissions, monthly average kWh per capita (see y-axis above). The property reductions impacted in the two-year reporting cycle make up ~5% of all buildings in Gainesville, but their 100% improvement reduces grid electrical consumption by 34% between January 2020 and January 2023. Moving the slider from left to right shows the reductions impact of these projects as they are completed. Planning over a two-year period allows people to contribute more actively to Gainesville’s Climate Action Plan 2-year reporting cycle, which allows the city to track its overall progress.

Throughout each two-year reporting cycle, the data supports action and better decision-making by identifying the best energy efficiency opportunities in a city to tackle first. The energy efficiency projects are prioritized based on the difference between the predicted and actual ICC building code release used in a building’s construction. If a property is predicted to have used the 1990 building code but is recorded as having used the 1996 code release, there is a higher likelihood of more significant reductions. This prioritization method allows all properties in the dataset to be ranked from highest to lowest by reduction opportunity. This prioritization reduces more emissions for the city and has increased financial benefits for residents and energy efficiency companies. 

The interactive demo of this project is viewable [here](http://katiejohnsonsf.pythonanywhere.com/).

## Impact

This dashboard helps people in the energy efficiency field prioritize energy efficiency projects for maximum impact and allows them to see how their work is impacting monthly average kWh per capita. It allows program managers at local non-profits, utilities, and energy efficiency retrofitting companies to size markets and prioritize reduction opportunities for a city. It helps answer the questions:

>
> 1. What is the optimal approach for addressing energy efficiency opportunities in a city?
>
> 2. To what extent will those energy efficiency improvements help a city get closer to its Carbon Zero goal?
>

[![Simulation of Carbon Zero City](https://j.gifs.com/OMmoKY.gif)](https://vimeo.com/user108223732/review/500521198/9c9e9c5b1f)
<br>
 
## Forecast Data
 
My focus was on electrical consumption data from the City of Gainesville website. The data contains over 9 million records dating from 2012 to September 2020. Covering these 8 years, the dataset includes monthly electricity use at all addresses in Gainesville. The data also contains latitude and longitude for each address.
<br>
 
## Premise
 
Hundreds of cities globally have committed to Carbon Neutrality by 2035 - including New York and Seattle. I was curious how a smaller city that hadn't (yet) made a commitment might be doing towards a path to carbon neutrality. I chose Gainesville, FL due to its size and easily available electrical consumption data.
 
This project aims to use machine learning to:  
1. Accurately predict future monthly electrical consumption

2. Prioritize addresses from highest to lowest energy efficiency improvement potential for optimal emissions reduction in project planning, and 

3. Simulate how making those improvements would impact future electrical consumption bringing the City closer to a carbon zero goal.
 
Predicting individual and aggregate reduction benchmarks could help expand investment and consumer cost savings opportunities in energy efficiency improvements.
<br>
 
## Overview of the Data
 
### Forecast Data Sources
 
* U.S. Census Bureau on population for the City of Gainesville
   * Gainesville_Population
* City of Gainesville's website
   * monthly_kwh_consumption by address
* EPA eGrid
   * Florida carbon intensity: 931.84 lb / MWh
<br>

## Variable Creation

Combining data into a single dataframe, I started by aggregating electrical consumption from individual addresses to find the city's average electrical consumption for each month since January 2012. Using Pandas groupby() and .sum(), I found 106 monthly values. I normalized the city's total consumption for each month by Gainesville's population each year to get my target variable, monthly average electrical consumption per capita. Below are the summary statistics and the plotted target values:


### Summary Statistics for target variable (monthly kwh / capita)

|       | avg_kwh_capita 
| :---        |    :----:  
| Count      | 106  
| Mean   | 1058.96 
| Std   | 270.92
| Min | 8.90     
| Max   | 1699.58  

<br>

### Plot of Avg Kwh Consumption per Capita
 
![Avg Monthly kWh per Capita](images/kWh_per_capita_diff.png)
 
Charting electrical consumption above shows a seasonal pattern and large drop at the start of the COVID pandemic. Both of these time index dependencies must be removed to create a set of stationary observations.

![Avg Monthly kWh per Capita](images/med_count_hist_diff.png)

<br>
The COVID anomaly is also apparent by looking at this histogram showing counts for each electrical consumption value, which are in two distinct groups. The COVID values lie completely outside of the Gaussian distribution of the rest of the data.      
<br>
<br>
 
 ## Box-Jenkins Method Stationary Process
 
By removing the observations in the data that occured during the COVID pandemic, I can achieve better stationarity in my data. To validate this, I used the Augmented Dickey-Fuller (ADF) test to find p-values for my data before and after removing COVID dates.
 

 ### ADF P-value with and without COVID observations

| with COVID | without COVID     |
| :---        |    :----:   |   
| 0.563      | 0.240  |

 <br>
 
While not within the 0.05 threshold yet for stationarity, removing observations during COVID significantly improved P-value.
 
I also checked how transforming my data using a log transform would impact my stationarity. It slightly increased my P-value from 0.240378 to 0.289347 so I did not use the log transform.

 Finally, I took the first and second difference of the data and ran an ADF test on each as well as for two regression orders. Here are the p-values for each:  
  <br>

| Regression Order      | Diff 1 | Diff 2     |
| :---        |    :----:   |          ---: |
| c      | 0.180      | 2.552e-16  |
| ctt   | 0.620       | 1.113e-13e      |

 <br>

The data that was diffed twice is stationary with slightly better performance with a constant regression order.

## Seasonal Decomposition
 
To get a clearer picture of the observed, trend, seasonal, residual patterns, I used seasonal decomposition to plot these in the target value. Looking at the graphs, the seasonal pattern is apparent. For the period parameter, I selected 12 because the sampling frequency is monthly (taken 12 times per year) over eight years. The residuals seem to be randomly distributed meaning that my data is stationary. 
 
<br>
 
![Decomposition](images/diffed_seas_decomp.png)
 
## Model
 
### SARIMA
 
Since my data has a strong seasonal component, I used the Seasonal AutoRegressive Integrated Moving Average (SARIMA) model, which includes a parameter for seasonal length. This allows the model to achieve better stationarity without further manual transformations. 
 
<br>
 
![SARIMA AC and PAC functions](images/AC_PAC_functions.png)

I found the autocorrelation and partial autocorrelation to get a rough sense for the complexity of my model for estimating a range of parameters for grid search. 
 
 
### Grid Search for Optimizing SARIMAX Parameters
I used GridSearch to iterate through possible values for the following SARIMA parameters both non-seasonal (lower case) and seasonal (uppercase):
* p, P (autoregressive terms)
* d, D (differencing needed to reach stationarity)
* q, Q (number of moving average terms - e.g. lags of the forecast errors)
* s (seasonal length in the data)
 
The GridSearch combination with the lowest AIC of 627.256 (indicating the strength of the model) was: 

| Parameter| Optimized value     |
| :---        |    :----:   |   
| autoregressive terms (p, P)      | 0, 2  |
| differencing needed to reach stationarity | 1, 2  |
| number of moving average terms     | 2, 1  |
| seasonal length in the data      | 12  |
<br>

Here's how it looks in code form 
```
SARIMA(0, 1, 2),(2, 2, 1, 12)
 ```
 
<br>
 
### Monthly Electricity Consumption Forecast to 2035 with SARIMA
(does not reflect increases in population past 2020)

![Prediction to 2035](images/pred_plot_diffed.png)

![Prediction to 2035 with Confidence Intervals](images/zoom_out_ci_diffed.png)

### Monthly Electrical Consumption Forecast within the Nearer Term: More Reasonable

![Prediction to 2022](images/pred_2022_diffed.png)
 
## Model Evaluation
 
### SARIMA
AIC: 627.257

<br>


# Regression Model for Predicting Energy Efficiency Opportunity

## EDA and data cleaning for building data


### Building Data Sources: 

City of Gainesville Building Permitting data
* Latitude
* Longitude

City of Gainesville
* monthly kwh consumption

Alachua Country Property Appraiser Improvement data
* Effective_YrBlt
* Heated_SquareFeet

### Rows of data after joining: 10,838

<br>

## Building Data EDA

To explore the characteristics of 10,838 properties, I plotted the count of buildings by effective year built. This shows a large spike in construction during the '80's with gradual decrease since.

![Prediction of cost to 2022](images/building_count_by_year.png)

To look at the overall energy efficiency of the properties, I created a feature for kwh / heated square feet. This plot shows efficiency for year built, which shows peak consumption during the 80's and 90's with a decrease since.

![Prediction of cost to 2022](images/efficiency_over_time.png)

## Model Selection

### Target Variable

The International Code Council releases new building codes in three year cycles. This release cycle began in 1927 with 30 code releases to date. External research on U.S. building codes and EDA of the building data shows consumption per square foot steadily decreasing overtime indicating that a higher code release is a signal for energy efficiency performance. I used the code release number (ranging 1-30) as a target for predicting energy efficiency improvement opportunity. 

### Predictor Variables 

I used a corellation matrix to remove colinear features selecting the following five features for training my energy efficiency improvement opportunity prediction model: 

* Latitude
* Longitude 
* Heated Square Feet
* Kwh per square foot
* Avg kwh 

To generate predictions, I split data 50/50 into training and test sets to maximize the number of predictions from unseen data. 

After training six models, Random Forest was the best performing with a MSE of 8.099. 

| Regression Model| Performance (RMSE)    |
| :---        |    :----:   |   
| Linear      | 3.45 |
| Lasso | 3.48 |
| Ridge     | 3.46 |
| KNN      | 3.65 |
| OLS      | 3.41 |
| Random Forest      | 2.97 |

<br>

## Feature Importance

From the Random Forest regressor, plotting shap values shows that location information had the largest impact on predictions. 

![Prediction of cost to 2022](images/shap_values.png)

<b>Latitude</b>

Looking at a more detailed description of feature importance below, latitude has a variable impact on the code release year prediction - having both high and low effects in both the positive and negative directions. North and South, there are no consistent efficiency patterns showing up in this data.
<br>

<b>Longitude</b>

However, the impact of longitude might be the most suprising having low positive impact and mostly a high negative effect. The further west the building is, the lower the predicted code release year. Perhaps this is caused by the development patterns where the Eastern part of the city contains more new housing.
<br>

<b>Heated Square Feet</b>

Next to location, this feature has the next largest impact. It has a high positive impact and lower negative impact. This is slightly surprising. One explaination might be that larger buildings are generally newer and, therefore, more efficient.
<br>

<b>Kwh per squarefoot</b>

This feature has a low negative impact on code release year. This might be explained by the inverse relationship with heated square feet.

<b>Avg kwh</b>

This feature has a high positive effect on predictions. It is somewhat surprising that greater consumption increases the efficiency standard prediction, but its magnitude is still relative small compared to location for impacting predictions.  

![Prediction of cost to 2022](images/feature_importance_shap.png)

<br>

## Building Efficiency Predictions

This plot shows the distribution of efficiency performance of 5,384 properties relative to efficiency standards when each was built using the residual of actual vs predicted code release. 

When the predicted building code release is less than the actual, the residual is a positive value. Positive residuals in this plot indicate there may be an opportunity for energy efficiency professionals to have a bigger impact because the building is predicted to be performing at a lower standard than was used in its construction.

![Prediction of cost to 2022](images/efficiency_values_dist.png)

## Simulating a Carbon Zero City

This application shows a 4-part rollout of the predicted energy efficiency improvements from highest impact to lowest and how each wave of improvements impacts monthly average kwh per capita in Gainesville. 

In the simulation, the efficiency improvements take place every 6 months between Jan 2020 and Jan 2022. This shows how a city might track progress and plan improvements for reaching a carbon zero by 2035 goal by planning and tracking action in two year intervals. 

[![Simulation of Carbon Zero City](https://j.gifs.com/OMmoKY.gif)](https://vimeo.com/user108223732/review/500521198/9c9e9c5b1f)
<br>


## Next Steps

As a next step, I'm interested in expanding efficiency-related data. Analyzing residential, commercial and industrial properties separately with property type-specific efficiency indicators to further segment the market.
 
I'm also interested in partnering with field experts or organizations to model other emissions sources to complete a full carbon zero forecast. These emissions sources include transportation, agriculture, industry, waste stream and energy production.
 
 
## Sources
* U.S. Census Bureau (city population data)
https://data.census.gov/cedsci/profile?g=1600000US1225175
* City of Gainesville Electrical Consumption data https://data.cityofgainesville.org/Utilities/GRU-Customer-Electric-Consumption/gk3k-9435
* City of Gainesville Building Permits https://data.cityofgainesville.org/Building-Development/Building-Permits/p798-x3nx
* Alachua County Property Assessor Improvements Data https://data2018-11-15t152007183z-acpa.opendata.arcgis.com/pages/gis-cama-data
* Carbon Neutral Bellingham (other emissions sources) https://carbonneutralbellingham.com/
* Banner image (Carbon Zero: Imagining Cities That Can Save the Planet  https://www.amazon.com/Carbon-Zero-Imagining-Cities-Planet-ebook/dp/B00AEWHU8E)
 
 
 Copyright © 2022 Katie Johnson
 
 
 

