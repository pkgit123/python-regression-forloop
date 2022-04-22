import pandas as pd
import numpy as np
from operator import attrgetter

pd.set_option("display.max_columns", 999)
pd.set_option("display.max_rows", 999)

%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print('Seaborn version: ', sns.__version__)

from sklearn.linear_model import LinearRegression

import re
import os

# import pandas_datareader as web
from datetime import datetime

%%time
'''
CPU times: user 5.19 s, sys: 1.25 s, total: 6.44 s
Wall time: 13.8 
'''

df_options_raw = pd.read_csv(s3_csv_path)

# exclude the futures and pending M&A deals
mask_exclude_tickers = df_options_raw['UnderlyingSymbol'].isin(ls_exclude)
df_options = df_options_raw.loc[~mask_exclude_tickers]

print(df_options.shape)
print(df_options.columns)
print('Number of records with no ask price: ', (df_options['Ask']==0).sum())
print('Number of records with ask price: ', (df_options['Ask']>0).sum())
print()

df_options.head()

def add_to_options(df_options):
    '''
    '''

    # ====================================================================================================
    # Discovered a faster way of converting string to datetime
    # https://www.reddit.com/r/learnpython/comments/6evlv5/faster_ways_to_convert_from_string_to_datetime/
    # ====================================================================================================

    print('Starting: ', pd.Timestamp.now())

    # add expiration date as date field
    # df_trade_ideas['exp_date'] = pd.to_datetime(df_trade_ideas['Expiration']).dt.date
    exp_date_cache = {k: pd.to_datetime(k) for k in df_options['Expiration'].unique()}
    df_options['exp_date'] = df_options['Expiration'].map(exp_date_cache)

    print('Converted expiration, now convering data_date: ', pd.datetime.now())

    # add data-date as date field
    # df_trade_ideas['data_date'] = pd.to_datetime(df_trade_ideas[' DataDate']).dt.date
    data_date_cache = {k: pd.to_datetime(k) for k in df_options[' DataDate'].unique()}
    df_options['data_date'] = df_options[' DataDate'].map(data_date_cache)

    print('Converted data_date, now calculating days_to_expire: ', pd.datetime.now())

    # add days to expiration
    # df_trade_ideas['days_to_expire'] = (df_trade_ideas['exp_date']-df_trade_ideas['data_date']).apply(attrgetter('days'))
    df_options['days_to_expire'] = (df_options['exp_date']-df_options['data_date'])
    df_options['days_to_expire'] = [x.days for x in df_options['days_to_expire'] ]

    print('Calculated days_to_expire, now adding lambda leverage: ', pd.datetime.now())
    
    # add bid-ask spread, and % of ask
    df_options['ba_spread'] = df_options['Ask'] - df_options['Bid']
    df_options['ba_pct_ask'] = df_options['ba_spread'] / df_options['Ask']

    # add Lambda implied leverage
    df_options['lambda_lev'] = df_options['Delta']*(df_options['UnderlyingPrice']/df_options['Ask'])
    df_options['lambda_lev'] = df_options['lambda_lev'].replace([np.inf, -np.inf], np.nan)

    # express IV as percentage rather than decimal, better for linear regression slope
    df_options['iv_pct'] = df_options['IV']*100

    # absolute value of delta
    df_options['abs_delta'] = df_options['Delta'].abs()

    # absolute value of lambda
    df_options['abs_lambda'] = df_options['lambda_lev'].abs()
    
    # theta vega
    df_options['theta_vega'] = df_options['Theta'].divide(df_options['Vega'])
    df_options['theta_vega'] = df_options['theta_vega'].replace([np.inf, -np.inf], np.nan)
    df_options['theta_vega'] = df_options['theta_vega'].fillna(0)
    
    # ===============================================================================================
    # add pandas column based on delta range
    # https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/
    # ===============================================================================================

    # create a list of our conditions
    ls_delta_cond = [
        ((df_options['Delta'] > 0.80) & (df_options['Delta'] <= 1.00)),
        ((df_options['Delta'] > 0.60) & (df_options['Delta'] <= 0.80)),
        ((df_options['Delta'] > 0.40) & (df_options['Delta'] <= 0.60)),
        ((df_options['Delta'] > 0.20) & (df_options['Delta'] <= 0.40)),
        ((df_options['Delta'] > 0.00) & (df_options['Delta'] <= 0.20)),
        ((df_options['Delta'] < -0.80) & (df_options['Delta'] >= -1.00)),
        ((df_options['Delta'] < -0.60) & (df_options['Delta'] >= -0.80)),
        ((df_options['Delta'] < -0.40) & (df_options['Delta'] >= -0.60)),
        ((df_options['Delta'] < -0.20) & (df_options['Delta'] >= -0.40)),
        ((df_options['Delta'] < -0.00) & (df_options['Delta'] >= -0.20)),
        ((df_options['Delta'] == 0)),
    ]

    # create a list of the values we want to assign for each condition
    ls_delta_cat = [
        'c_80_100', 'c_60_80', 'c_40_60', 'c_20_40', 'c_0_20', 
        'p_80_100', 'p_60_80', 'p_40_60', 'p_20_40', 'p_0_20',
        'delta_zero'
    ]

    # create a new column and use np.select to assign values to it using our lists as arguments
    df_options['delta_cat'] = np.select(ls_delta_cond, ls_delta_cat)
    
    # create column with symbol, type, expiration
    df_options['symbol_type_exp'] = (
        df_options['UnderlyingSymbol'].astype(str) 
        + '_' + df_options['Type'].astype(str)
        + '_' + df_options['exp_date'].astype(str)
    )
    
    # add column for delta-weighted-openinterest
    df_options['dw_openint'] = df_options['abs_delta'] * df_options['OpenInterest'] 
    
    # add column for Volume divided by OpenInterest
    df_options['volume_div_oi'] = df_options['Volume'] / df_options['OpenInterest'] 

    print('Ending: ', pd.Timestamp.now(), '\n')
    
    return df_options
  
  %%time
'''
CPU times: user 22.5 s, sys: 523 ms, total: 23 s
Wall time: 23 s
'''

df_options = add_to_options(df_options)

%%time
'''
6659
5882
CPU times: user 3min 28s, sys: 1.22 s, total: 3min 30s
Wall time: 3min 30s
'''


# create dictionary to store delta-weighted openinterest
di_ticker_dwopeninterest = {}

# create list to store tickers with errors
ls_errors__dwopeninterest = []

for each_symbol_type_exp in df_short_tenor['symbol_type_exp'].unique():
    
    try:

        # create temp df
        each_df = df_short_tenor.query(" symbol_type_exp == @each_symbol_type_exp ")

        # sort by abs_delta
        each_df = each_df.sort_values('abs_delta', ascending=False)

        # calculate number of strikes with increasing dw_openinterest
        each_df['shift_dw_openint'] = each_df['dw_openint'].shift(1)
        each_df['pos_dw_openint'] = each_df['dw_openint'] > each_df['shift_dw_openint']

        # calculate metrics pos_dw_openint 
        num_pos_dw_openint = each_df['pos_dw_openint'].sum()
        num_strikes_dw_openint = each_df['pos_dw_openint'].shape[0]
        pct_pos_dw_openint = each_df['pos_dw_openint'].mean()
        
        # calculate metrics dw_openint
        mean_dw_oi = each_df['dw_openint'].mean()
        max_dw_oi = each_df['dw_openint'].max()
        pct_max_mean_dw_oi = max_dw_oi/mean_dw_oi
        
        # calculate metrics on volume and openinterest 
        sum_openinterest = each_df['OpenInterest'].sum()
        sum_volume = each_df['Volume'].sum()
        pct_volume_openint = sum_volume/sum_openinterest
        
        
        # ==================
        # linear regression
        # ==================
        
        # filter only upward downward deltas, from low deltas (high dw-openint) to high deltas (low dw-openint)
        filter_each_df = each_df.query(" pos_dw_openint == True ")
        
        # only run regression if more than 2 deltas, otherwise throw error
        assert filter_each_df.shape[0] > 2

        # input variable, convert to numpy array
        X = np.array(filter_each_df['abs_delta']).reshape(-1, 1)

        # output variable, no need to convert to numpy array
        y = filter_each_df['dw_openint']

        # run linear regression, calculate fit score, slope, and y-intercept
        lin_reg = LinearRegression().fit(X, y)
        model_score = lin_reg.score(X, y)
        model_slope = lin_reg.coef_[0]
        y_intercept = lin_reg.intercept_

        # =====================
        # store output results
        # =====================
        
        # store metrics as a tuple
        di_ticker_dwopeninterest[each_symbol_type_exp] = (
            num_pos_dw_openint, num_strikes_dw_openint, pct_pos_dw_openint, 
            mean_dw_oi, max_dw_oi, pct_max_mean_dw_oi,
            sum_openinterest, sum_volume, pct_volume_openint,
            model_score, model_slope, y_intercept
        )
        
    except:
        
        ls_errors__dwopeninterest.append(each_symbol_type_exp)

    
print(len(di_ticker_dwopeninterest.keys()))
print(len(ls_errors__dwopeninterest))

# in order to convert our results (dictionary of tuples) to dataframe, create list of column names
ls_col_names = [
    'num_pos_dw_oi', 'num_strikes_dw_oi', 'pct_pos_dw_oi', 
    'mean_dw_oi', 'max_dw_oi', 'pct_max_mean_dw_oi',
    'sum_openinterest', 'sum_volume', 'pct_volume_openint',
    'model_score', 'model_slope', 'y_intercept'
]

# convert dictionary results to dataframe
df_ticker_dwopeninterest = pd.DataFrame.from_dict(di_ticker_dwopeninterest, orient='index', columns=ls_col_names)

# reset index, convert index column name to ticker
df_ticker_dwopeninterest = df_ticker_dwopeninterest.reset_index().reset_index(drop=True).rename(columns={'index': 'symbol_type_exp'})
df_ticker_dwopeninterest = df_ticker_dwopeninterest.sort_values('pct_pos_dw_oi', ascending=False)
