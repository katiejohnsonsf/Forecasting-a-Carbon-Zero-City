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
from pmdarima.utils import diff_inv

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
    +inverse_diff
    +evaluate_SARIMAX_model
    +plot_SARIMAX_pred

    
    
   '''
    
    def load_stationary_data(self, filepath):
        diffed = pd.read_csv(filepath)
        return diffed
    
    def create_train_set(self, diffed):
        '''
        ==Function==
        Split data into train(80%) and test(20%) dataframes 
        
        ==Returns==
        Train dataframe
        
        '''
        diffed = self.load_stationary_data()
        tr_start,tr_end = '2012-02-28','2018-05-31'

        train = covid_rem_df[tr_start:tr_end]
        return train

    def create_test_set(self, diffed):
        '''
        ==Function==
        Split data into train(80%) and test(20%) dataframes 
        
        ==Returns==
        test dataframe 
        
        '''
        diffed = self.load_stationary_data()
        te_start,te_end = '2018-06-30','2020-01-31'
        test = covid_rem_df[te_start:te_end]
        return test
    
    def grid_search(self, train):
        '''
        ==Function==
        Set parameter range
        Find optimized parameters for SARIMAX 
        
        ==Returns==
        List of all parameter combinations
        
        '''
        train = self.create_train_set()

        p = range(0,3)
        q = range(1,3)
        d = range(1,2)
        s = range(1,12)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = list(itertools.product(p, d, q, s))
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(train,
                                            order=param,
                                            seasonal_order=param_seasonal)
                    results = mod.fit(max_iter = 50, method = 'powell')

                    param_results = 'SARIMAX{},{} - AIC:{}'.format(param, param_seasonal, results.aic)
                except:
                    continue
        return param_results
    
    def load_orig_data(self):
        cov_rem = pd.read_csv('./stationary-data/cov_rem.csv')
        return cov_rem

    def inv_diff(df_orig_column ,df_diff_column, periods):
        # Generate np.array for the diff_inv function - it includes first n values(n = 
        # periods) of original data & further diff values of given periods
        value = np.array(df_orig_column[:periods].tolist()+df_diff_column[periods:].tolist())

        # Generate np.array with inverse diff
        inv_diff_vals = diff_inv(value, periods,1)[periods:]
        return inv_diff_vals

    def evaluate_sarimax_model(self, train):
        '''
        ==Function ==
        Pushes through SARIMAX models 
        ==Returns==
        RMSE
        '''
        train = self.create_train_set()
        mod = sm.tsa.statespace.SARIMAX(train, order=(0, 1, 2), seasonal_order=(2, 1, 1, 4))
        results = mod.fit(disp=False)
        predictions = results.predict(start=78, end=97)
        predictions['Actual'] = train['avg_kwh_capita']
        error=rmse(predictions['predicted_mean'], predictions['Actual'])
        return error
    
    def plot_SARIMAX_pred(self):
        '''
        ==Function==
        Creates a plot of electricity consumption predictions from the SARIMAX Model:
            [1] depicting forecasts
            [1] associated residual distribution plots
        '''
        #train_test_split 
        
        train = self.create_train_set()
        test = self.create_test_test()

        # Train a SARIMAX model with observed values 
        e_consum_model = sm.tsa.statespace.SARIMAX(future_kwh, order=(0, 1, 2), seasonal_order=(2, 1, 1, 4)).fit()


        # specify number of forecasts
        preds = e_consum_model.get_prediction(start='2018-01-31', end='2022-01-31', dynamic=False)
        pred_ci = preds.conf_int()
        pred_ci = preds.conf_int()


        ax = future_kwh['2018-01-31': '2022-01-31'].plot(label='observed')
        preds.predicted_mean.plot(ax=ax, label='Forecast', alpha=.8, figsize=(15, 7))
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], facecolor='green', alpha=0.1)

        ax.set_title("Forecast of Monthly Avg Electrical Consumption per Capita: Gainesville, FL", fontsize=18)
        ax.set_xlabel('Date', fontsize='x-large')
        ax.set_ylabel('Avg monthly kwh consumed', fontsize='x-large')
        plt.legend(['Observed', 'Predicted', '95% Confidence Interval'], loc=3, fontsize='large')

        fig1 = plt.gcf()
        plt.savefig('images/pred_2022_diffed.png')
        plt.show()
        plt.draw()


if __name__ == "__main__":

    # variables below
    filepath = './stationary-data/diffed_data.csv'
    start = '2020-02-28'
    end = '2035-02-28'
    actual = act['avg_kwh_capita']
    
    # functions for testing
    load_stationary_data(self, filepath)
    create_train_set(self, diffed)
    create_test_set(self, diffed) 
    grid_search(self, train)
    load_orig_data(self)
    undiffed_preds = inv_diff(cov_rem['avg_kwh_capita'], predictions['predicted_mean'], 12)
    evaluate_sarimax_model(self, train)
    plot_SARIMAX_pred(self)