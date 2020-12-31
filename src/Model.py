import pandas as pd
import geopandas as gpd
import numpy as np 
from numpy import log
import gmaps 
import gmaps.datasets 
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('ggplot')
from sorted_months_weekdays import Month_Sorted_Month

# Arima class
from statsmodels.compat.pandas import Appender

import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D

from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE
from statsmodels.tsa.statespace.tools import diff
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
    innovations, innovations_mle)
from statsmodels.tsa.arima.estimators.gls import gls as estimate_gls

from statsmodels.tsa.arima.specification import SARIMAXSpecification
import statsmodels.api
import statsmodels as sm
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

import itertools


class Model():
    
    '''
    ==Function==
    Run SARIMAX Model
    
    ==Output==
    1 Model - SARIMAX

    1 subplots 
        [1] Forecast Plot with associated MSE score

        
        
    ==Parameters==
    |order_method| - selection of various methods of looking for/selecting an SARIMAX order
        - 'Grid Search'

    ==Included Functions==
    +load_df
    +train
    +test
    +grid_search
    +evaluate_SARIMAX_model
    +plot_SARIMAX_pred

    
    
   '''
    
    def __init__(self):
        self.self = self
        self.files = ['stationary-data/not_diffed.csv']
    
    def load_stationary_data(self):
        print(" 1 of 6 |    Reading in data \n         |       not_diffed.csv")
        covid_rem_df = pd.read_csv('./stationary-data/not_diffed.csv')
        return covid_rem_df
    
    def create_train_set(self, covid_rem_df):
        '''
        ==Function==
        Split data into train(80%) and test(20%) dataframes 
        
        ==Returns==
        Train dataframe
        
        '''
        covid_rem_df = self.load_stationary_data()
        tr_start,tr_end = '2012-02-28','2018-05-31'

        train_diff0 = covid_rem_df[tr_start:tr_end]
        return train_diff0

    def create_test_set(self, covid_rem_df):
        '''
        ==Function==
        Split data into train(80%) and test(20%) dataframes 
        
        ==Returns==
        test dataframe 
        
        '''
        covid_rem_df = self.load_stationary_data()
        te_start,te_end = '2018-06-30','2020-01-31'
        test_diff0 = covid_rem_df[te_start:te_end]
        return test_diff0
    
    def grid_search(self, train_diff0):
        '''
        ==Function==
        Set parameter range
        Find optimized parameters for SARIMAX 
        
        ==Returns==
        List of all parameter combinations
        
        '''
        train_diff0 = self.create_train_set()

        p = range(0,3)
        q = range(1,3)
        d = range(1,2)
        s = range(1,12)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = list(itertools.product(p, d, q, s))
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(train_diff0,
                                            order=param,
                                            seasonal_order=param_seasonal)
                    results = mod.fit(max_iter = 50, method = 'powell')

                    param_results = 'SARIMAX{},{} - AIC:{}'.format(param, param_seasonal, results.aic)
                except:
                    continue
        return param_results


    def evaluate_sarimax_model(self, train_diff0):
        '''
        ==Function ==
        Pushes through SARIMAX models 
        ==Returns==
        RMSE
        '''
        train_diff0 = self.create_train_set()
        mod = sm.tsa.statespace.SARIMAX(train_diff0, order=(0, 1, 2), seasonal_order=(2, 1, 1, 4))
        results = mod.fit(disp=False)
        predictions = results.predict(start=78, end=97)
        predictions['Actual'] = train_diff0['avg_kwh_capita']
        error=rmse(predictions['predicted_mean'], predictions['Actual'])
        return error
    
    def plot_SARIMAX_pred(self):
        '''
        ==Function==
        Creates a subplot from the SARIMAX Model:
            [1] depicting forecasts
            [1] associated residual distribution plots
        '''
        #train_test_split with difference = 0
        
        train_diff0 = self.create_train_set()
        test_diff0 = self.create_test_test()
        
        # Create and fit model using optimized parameters 
        mod = sm.tsa.statespace.SARIMAX(train_diff0, order=(0, 1, 2), seasonal_order=(2, 1, 1, 4))
        results = mod.fit()
        
        # Create predictions from model 
        predictions = results.predict(start=78, end=97)
        act = pd.DataFrame(test_diff0)
        predictions=pd.DataFrame(predictions)
        predictions.reset_index(drop=True, inplace=True)
        predictions.index=test.index
        predictions['Actual'] = act['avg_kwh_capita']
        predictions.rename(columns={1:'Pred'}, inplace=True)

        #create dictionary of dates and predictions to plot
        index = pd.date_range(start='2020-02-28', end='2035-02-28', freq='M')
        columns = ['pred_kwh']
        future_kwh = pd.DataFrame(index=index, columns=columns)

        pred_dict = {}
        for i, date in enumerate(future_kwh.index):
            pred_dict[date] = predictions.iloc[i]
        future_kwh['pred_kwh'] = pd.Series(pred_dict)
        future_kwh['pred_kwh'].plot(legend=True, color='red', figsize=(20,8))
        plt.show() 

        #===Plot
        predictions['Actual'].plot(figsize=(20,8), legend=True, color='blue')
        predictions['predicted_mean'].plot(legend=True, color='red', figsize=(20,8))
        
        show.plt()

    # if __name__ == "__main__":