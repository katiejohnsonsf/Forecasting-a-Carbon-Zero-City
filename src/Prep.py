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
    
    def load_pop_df(self, filepath_pop):
        print('PREP'.center(76,'-'))
        print(" 2 of 7 |    Reading in data \n         |       population.csv")
        population_df = pd.read_csv(filepath_pop)
        return population_df
        
    def load_kwh_df(self, filepath_kwh):
        print('PREP'.center(76,'-'))
        print(" 1 of 7 |    Reading in data \n         |       GRU_Customer_Electric_Consumption \n         ")
        kwh_df = pd.read_csv(filepath_kwh)
        return kwh_df

    def monthly_mean(self, kwh_df):
        print(' 3 of 7 |    Grouping by monthly average kWH consumption')
        kwh_df = kwh_df.groupby('Date', dropna=True, sort=True).mean()
        return kwh_df

    def join_dataframes(self, population_df, kwh_df):
        print(' 4 of 7 |    Joining kWH and popluation dataframes')
        pop_e_df = kwh_df.join(population_df.set_index('Date'), how='left', lsuffix='_left', rsuffix='_right')
        pop_e_df = pop_e_df.drop('Year_right', axis=1)
        pop_e_df = pop_e_df.rename(columns={'Year_left': 'Year'})
        return pop_e_df

    def target_variable(self, pop_e_df, targ_col_name):
        print(' 5 of 7 |    Creating target variable: avg_kwh_capita')
        pop_e_df['avg_kwh_capita'] = pop_e_df['KWH Consumption'] / pop_e_df['Population']
        return pop_e_df

    def rem_covid_dates(self, pop_e_df):
        print(' 6 of 7 |    Removing COVID dates')
        cov_rem = pop_e_df[targ_col_name][96:]
        return cov_rem

    def diff_twice(self, cov_rem):
        print(' 7 of 7 |    Taking second difference to stationize')
        diffed = cov_rem.diff().diff().dropna()
        return diffed

if __name__ == "__main__":   

    p = Prep()

    # Variables
    filepath_pop = '../data/population.csv'
    filepath_kwh = '../data/GRU_Customer_Electric_Consumption.csv'
    targ_col_name = 'avg_kwh_capita'
    
    
    # function for testing

    # read in population data for the city of Gainesville
    pop_data_load = p.load_pop_df(filepath_pop)
    
    # read in electricity consumption data
    kwh_load = p.load_kwh_df(filepath_kwh)

    # Groupby date of electrical consumption and aggregate values by mean for each date
    get_avg = p.monthly_mean(kwh_load)

    # join the popluation dataframe with the average monthly dataframe
    join_pop_kwh = p.join_dataframes(pop_data_load, kwh_load)

    # Create a feature column for monthly average consumption per capita
    avg_consum = p.target_variable(join_pop_kwh, targ_col_name)
     
    # remove COVID dates
    rem_covid = p.rem_covid_dates(avg_consum)

    # take second difference to make data stationary 
    stationize = p.diff_twice(rem_covid)

    # # print results
    # print(f"Population filepath: {filepath_pop}")
    # print(f"kwh filepath: {filepath_kwh}.")
    # print(f"target variable: {pop_e_df}.")
    # print(f"Remove COVID anomaly: {cov_rem}")
    # print(f"Make stationary: {cov_rem}")
