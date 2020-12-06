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
    
    == Stationarity==
    Uses ADFuller method to test for stationarity.
    If data is not stationary, uses differencing to achieve stationarity.
    Completes ADF testing again
    
    compile
    ==Returns==
    Stationary dataframe 
    
    ==Prints==
    data
    '''

        
    def __init__(self):
        self.self = self
        self.files = ['data/GRU_Customer_Electric_Consumption.csv']
        
    def load(self):
        print('PREP'.center(76,'-'))
        print(" 1 of 11 |    Reading in data \n         |       GRU_Customer_Electric_Consumption \n         ")
        kWH_df = pd.read_csv('data/GRU_Customer_Electric_Consumption.csv')
        print("         |    Reading in data \n         |       population.csv")
        population_df = pd.read_csv('./data/population.csv')

        return kWH_df, population_df

    def sort(self):
        kWH_df = self.load()
        print(' 2 of 11 |    Sorting by Month and Year column values')
        kWH_df = kWH_df.sorted()

        return kWH_df

    def refine(self):
        df_kWH = self.sort()
        print(' 3 of 11 |    Refining to only Year, Month, Date, KWH Consumption')
        print(type(df_kWH))
        df_kWH = df_kWH[['Year', 'Month', 'Date', 'KWH Consumption']]
        return df_kWH

    def date_index(self):
        population_df = self.load()
        kWH_df = self.refine()
        print(' 4 of 11 |    Assigning Date as index')
        population_df.set_index('Date')
        kWH_df['Date'] = pd.to_datetime(df['Date'])
        kWH_df = kWH_df.set_index('Date')
        return population_df, kWH_df

    def monthly_mean(self):
        kWH_df = self.date_index()
        print(' 5 of 11 |    Grouping by monthly average kWH consumption')
        kWH_df = kWH_df.groupby('Date', dropna=True, sort=True).mean()
        return kWH_df

    def join_dataframes(self):
        population_df = self.date_index()
        kWH_df = self.monthly_mean()
        print(' 6 of 11 |    Joining kWH and popluation dataframes')
        pop_e_df = kWH_df.join(population_df.set_index('Date'), how='left', lsuffix='_left', rsuffix='_right')
        pop_e_df = pop_e_df.drop('Year_right', axis=1)
        pop_e_df = pop_e_df.rename(columns={'Year_left': 'Year'})
        return pop_e_df

    def target_variable(self):
        pop_e_df = self.join_dataframes()
        print(' 7 of 11 |    Creating target variable: avg_kwh_capita')
        pop_e_df['avg_kwh_capita'] = pop_e_df['KWH Consumption'] / pop_e_df['Population']
        return pop_e_df

    def stationarity(self):
        pop_e_df = self.target_variable()
        print(' 9 of 11 |    Testing for stationarity')
        if round(adfuller(pop_e_df)[1],4) < 0.51:
            print("         |       ADF P-value: {} \n         |       Time Series achieved stationarity. \n         |       Reject ADF H0".format(round(adfuller(pop_e_df)[1],4)))
            print('prep complete'.upper().center(76,'-'))
            return pop_e_df
        else:
            print('         |       ADF P-value: {} \n         |       Time Series is not stationary.   \n         |       Fail to reject ADF H0'.format(round(adfuller(pop_e_df)[1],4)))
            print('10 of 11 |    Removing observations within COVID dates to achieve stationarity')
            data_start, cov_start = '2012-01-31','2020-01-31'
            pop_e_df = pop_e_df['avg_kwh_capita'][data_start:cov_start]
            print('11 of 11 |    Testing for stationarity on COVID-free data')
            if round(adfuller(pop_e_df)[1],4) < 0.51:
                print('         |       ADF P-value: {} \n         |       Differenced data achieved stationarity. \n         |       Reject ADF H0'.format(round(adfuller(first_ord)[1],4)))
                print('prep complete'.upper().center(76,'-'))
                return pop_e_df
            else:
                print('After removing observsations during COVID, data is still not stationary. \
                Consider applying other methods.')
                print('prep complete'.upper().center(76,'-'))
                return pop_e_df
    
    def compile(self):
        pop_e_df = self.stationarity()[0]
        return pd.DataFrame(pop_e_df)

if __name__ == "__main__":   
    pop_e_df = Prep().compile()
