import pandas as pd
from datetime import datetime
from statsmodels.tsa.stattools import adfuller


# =============================================================================
# DF PREP
# =============================================================================
class Prep():
    '''
    ==Function==
    Prepare electrical consumption data for time series analysis 
    
    ==Columns==
    Service Address, Service City, Month, Year, Date, kWH, Consumption, 
    Latitude, Longitude, Location        
    
    ==Index==
    Ordered by Date and Location
    
    ==Customer Segment==
    Residential, commercial and industrial  
    
    ==Null Values==
    Observations containing Null values for kWH were removed 

    ==Population Adjustment==
    Adjusts monthly average consumption per capita to reflect population in the year of consumption
    The adjustment is made using data provided by The U.S. Census Bureau for the City of Gainesville.
    
    ==Target Variable==
    create target variable, kWH_per_capita 
    kWH_per_capita  = average monthly kWH consumption / population of consumption year
    
    compile
    ==Returns==
    Stationary dataframe 
    
    ==Prints==
    data
    '''

        
    def __init__(self):
        self.self = self
        self.files = ['data/GRU_Customer_Electric_Consumption.csv']
    
    def load_pop_df(self):
        print('PREP'.center(76,'-'))
        print(" 2 of 7 |    Reading in data \n         |       population.csv")
        population_df = pd.read_csv('./data/population.csv')
        return population_df
        
    def load_kwh_df(self):
        print('PREP'.center(76,'-'))
        print(" 1 of 7 |    Reading in data \n         |       GRU_Customer_Electric_Consumption \n         ")
        kWH_df = pd.read_csv('data/GRU_Customer_Electric_Consumption.csv')
        return kWH_df

    def monthly_mean(self):
        kWH_df = self.load_kwh_df()
        print(' 3 of 7 |    Grouping by monthly average kWH consumption')
        kWH_df = kWH_df.groupby('Date', dropna=True, sort=True).mean()
        return kWH_df

    def join_dataframes(self):
        population_df = self.load_pop_df()
        kWH_df = self.monthly_mean()
        print(' 4 of 7 |    Joining kWH and popluation dataframes')
        pop_e_df = kWH_df.join(population_df.set_index('Date'), how='left', lsuffix='_left', rsuffix='_right')
        pop_e_df = pop_e_df.drop('Year_right', axis=1)
        pop_e_df = pop_e_df.rename(columns={'Year_left': 'Year'})
        return pop_e_df

    def target_variable(self):
        pop_e_df = self.join_dataframes()
        print(' 5 of 7 |    Creating target variable: avg_kwh_capita')
        pop_e_df['avg_kwh_capita'] = pop_e_df['KWH Consumption'] / pop_e_df['Population']
        return pop_e_df

    def rem_covid_dates(self):
        pop_e_df = self.target_variable()
        print(' 6 of 7 |    Removing COVID dates')
        data_start, cov_start = '2012-01-31','2020-01-31'
        cov_rem = pop_e_df['avg_kwh_capita'][data_start:cov_start]
        return cov_rem

    def diff_twice(self):
        cov_rem = self.rem_covid_dates()
        print(' 7 of 7 |    Taking second difference to stationize')
        diffed = cov_rem.diff().diff().dropna()
        return diffed

    def compile(self):
        diffed = self.diff_twice()
        return diffed

if __name__ == "__main__":   
    
    # read in population data for the city of Gainesville
    load_pop_df(self)
    
    # read in electricity consumption data
    load_kwh_df(self)

    # Groupby date of electrical consumption and aggregate values by mean for each date
    monthly_mean(self)

    # join the popluation dataframe with the average monthly dataframe
    join_dataframes(self)

    # Create a feature column for monthly average consumption per capita
    target_variable(self)
     
    # remove COVID dates
    rem_covid_dates(self)

    # take second difference to make data stationary 
    diff_twice(self)

