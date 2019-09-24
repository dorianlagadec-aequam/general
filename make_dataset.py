'''Module for the proper dataset generation'''

#import os
#import pdb
from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ta

def _convert_time(date, input_format, output_format):
    '''Use the datetime formats to convert one string to another. For us, useful to unify formats'''
    return dt.strptime(date, input_format).strftime(output_format)

def _column_name(variable, indicator, window=''):
    '''Generalizes the format of column names in our datafrme, using the technical
    indicator and the proper window'''
    return str(variable) + '_' + indicator + '_' + str(window)


#risk_premia_path = '../data_raw/risk_premia.csv'
#macros_path = '../data_raw/macros.csv'

class ProperDataset:
    '''Class useful to transform raw datasets, whose format won\'t change in the future,
    into a proper dataset ready for study'''
    def __init__(self, risk_premia_path, macros_path, use_risk_premia_and_macros=True):
        risk_premia_df = pd.read_csv(risk_premia_path, header=0, index_col=0)
        risk_premia_df = risk_premia_df.iloc[1:]
        risk_premia_df = risk_premia_df.rename(columns={'Monmentum':'Momentum'})
        input_format = '%d/%m/%Y'
        output_format = '%Y-%m-%d'
        ####### Formatting the Risk Premia prices dataframe
        new_index = [_convert_time(day, input_format, output_format) \
                for day in list(risk_premia_df.index)]
        risk_premia_df = pd.DataFrame(np.array(risk_premia_df), index=pd.DatetimeIndex(new_index), \
                columns=risk_premia_df.columns).astype(float)
        risk_premia_df.index.name = 'date'
        if use_risk_premia_and_macros:      
            ####### Formatting the macroeconomic indicators dataframe
            macros_df = pd.read_csv(macros_path, header=0, index_col=0)
            macros_df = pd.DataFrame(np.array(macros_df), index=pd.DatetimeIndex(macros_df.index), \
                    columns=macros_df.columns).astype(float)

            ####### Joining both for when we will use them

            total_df = risk_premia_df.join(macros_df, how='outer')
        else:
            total_df = risk_premia_df.copy()

        self.use_risk_premia_and_macros = use_risk_premia_and_macros
        self.total_df = total_df
        self.risk_premia_cols = risk_premia_df.columns
        self.macros_cols = macros_df.columns

    def add_technical_indicators(self, rsi_window=[7, 14], macd_window=[12, 26], ma_window=[7, 14], ema_window=[7, 14]):
        '''Method that adds some chosen technical indicators to the working dataframe
            MACD : Moving Average Convergence Difference, MA : Moving Average, EMA : 
            Exponential MA, RSI : Relative Strength Indicator'''
        for risk_premium in self.risk_premia_cols:
            #Returns
            new_return_name = _column_name(risk_premium, 'Return')
            self.total_df[new_return_name] = self.total_df[risk_premium].pct_change(1)
            #RSI
            for window in rsi_window:
                new_rsi_name = _column_name(risk_premium, 'RSI', window)
                self.total_df[new_rsi_name] = ta.momentum.rsi(self.total_df[risk_premium], n=window, fillna=False)
            #MACD
            new_macd_name = _column_name(risk_premium, 'MACD')
            self.total_df[new_macd_name] = ta.trend.macd(self.total_df[risk_premium],\
                 n_fast=macd_window[1], n_slow=macd_window[0], fillna=False)
            #MA
            for window in ma_window:
                new_ma_name = _column_name(risk_premium, 'MA', window)
                self.total_df[new_ma_name] = ta.volatility.bollinger_mavg(self.total_df[risk_premium], n=window, fillna=False)
            #EMA
            for window in ema_window:
                new_ema_name = _column_name(risk_premium, 'EMA', window)
                self.total_df[new_ema_name] = ta.trend.ema_indicator(self.total_df[risk_premium], n=window, fillna=False)
        if self.use_risk_premia_and_macros:
            #Same as above, but without returns because it makes no sense for macro indicators
            for macro in self.macros_cols:
                #RSI
                for window in rsi_window:
                    new_rsi_name = _column_name(macro, 'RSI', window)
                    self.total_df[new_rsi_name] = ta.momentum.rsi(self.total_df[macro], n=window, fillna=False)
                #MACD
                new_macd_name = _column_name(macro, 'MACD')
                self.total_df[new_macd_name] = ta.trend.macd(self.total_df[macro],\
                     n_fast=macd_window[1], n_slow=macd_window[0], fillna=False)
                #MA
                for window in ma_window:
                    new_ma_name = _column_name(macro, 'MA', window)
                    self.total_df[new_ma_name] = ta.volatility.bollinger_mavg(self.total_df[macro], n=window, fillna=False)
                #EMA
                for window in ema_window:
                    new_ema_name = _column_name(macro, 'EMA', window)
                    self.total_df[new_ema_name] = ta.trend.ema_indicator(self.total_df[macro], n=window, fillna=False)

    # def restrict_to(self, start_time=None, end_time=None, variable='', technical_indicator=''):
    #     '''Restricts the observed dataframe to a timeframe window and to chosen variables
    #     and technical indicators

    #     Dates must be passed in 'yyyy-mm-dd' format
    #     For several values in variable or technical indicator, provide them in a tuple

    #     To search for variables, please look self.risk_premia_cols or self.macros_cols'''
    #     if technical_indicator != '':
    #         lst = list(technical_indicator)
    #         for i in range(len(lst)):
    #             lst[i] = '_' + lst[i] + '_'
    #         tup = '|'.join(lst)
    #     else:
    #         tup = technical_indicator
    #     return(self.total_df.loc[start_time:end_time, self.total_df.columns.str.startswith(variable)\
    #         & self.total_df.columns.str.contains(tup)])

    def restrict_to(self, start_time=None, end_time=None, variable='', technical_indicator=''):
        '''Restricts the observed dataframe to a timeframe window and to chosen variables
        and technical indicators

        Dates must be passed in 'yyyy-mm-dd' format
        For several values in variable or technical indicator, provide them in a tuple

        To search for variables, please look dataset.risk_premia_cols or dataset.macros_cols'''
        if (technical_indicator != '') & (type(technical_indicator)==tuple):
            # print('here')
            lst = list(technical_indicator)
            for i in range(len(lst)):
                lst[i] = '_' + lst[i] + '_'
            tup = '|'.join(lst)
        else:
            tup = technical_indicator
        # print(tup)
        # import pdb ; pdb.set_trace()
        return(self.total_df.loc[start_time:end_time, self.total_df.columns.str.startswith(variable)\
            & self.total_df.columns.str.contains(tup)])






if __name__ == '__main__':
    #CHEMINS AMBIGUS A CHANGER
    test = ProperDataset("data/risk_premia.csv", "data/macros.csv")
    test.add_technical_indicators()
    #pdb.set_trace()
    # test.total_df.to_csv('data_transformed/test.csv')
    print(test.restrict_to('2010-01-01','2010-01-15',variable='Value', technical_indicator='EMA'))