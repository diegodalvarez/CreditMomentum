# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:06:22 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

from CDSIndexCollect import CreditDefaultSwapDataCollector

'''
Momentum Strategies in Credit: A Review of different trend strategies applied
to credit indices
'''

class MomentumCreditSignals(CreditDefaultSwapDataCollector):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.momentum_signal = os.path.join(self.data_path, "MomentumSignals")
        
        if os.path.exists(self.momentum_signal) == False: os.makedirs(self.momentum_signal)
        
    def _get_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.assign(
            px_rtn  = lambda x: np.log(x.px) - np.log(x.px.shift()),
            cum_rtn = lambda x: np.cumsum(x.px_rtn)))
        
        return df_out
    
    def _get_regression(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.assign(
            roll_cov = lambda x: x.cum_rtn.rolling(window = window).cov(x.input_val),
            roll_var = lambda x: x.input_val.rolling(window = window).var(),
            beta     = lambda x: x.roll_cov / x.roll_var,
            lag_beta = lambda x: x.beta.shift()).
            dropna())

        return df_out
    
    def _get_regression_signals(self, df: pd.DataFrame, windows: dict) -> pd.DataFrame: 
        
        df_tmp = (df.reset_index(
            drop = True).
            reset_index().
            sort_values("date").
            rename(columns = {"index": "input_val"}).
            assign(input_val = lambda x: x.input_val + 1))
        
        df_out = (pd.concat([
            self._get_regression(df_tmp, windows[key]).assign(window = key) 
            for key in windows.keys()]))
        
        return df_out
        
    def get_regression_signals(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.momentum_signal, "BetaSignals.parquet")
        try:
            
            if verbose == True: print("Trying to find beta signals")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it now")
        
            df_rtn = (self.get_cds_indices().groupby(
                "security").
                apply(self._get_rtn).
                reset_index(drop = True))
            
            windows = {
                "4week": 4*5,
                "5week": 5*5,
                "6week": 6*5,
                "7week": 7*5,
                "8week": 8*5}
            
            df_out = (df_rtn.groupby(
                "security").
                apply(self._get_regression_signals, windows).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _get_zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                roll_mean  = lambda x: x.beta.ewm(span = window, adjust = False).mean(),
                roll_std   = lambda x: x.beta.ewm(span = window, adjust = False).std(),
                z_score    = lambda x: (x.beta - x.roll_mean) / x.roll_std,
                lag_zscore = lambda x: x.z_score.shift()).
            drop(columns = ["roll_mean", "roll_std"]).
            dropna())
        
        return df_out
    
    def generate_regression_zscore(self, window: int = 365, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.momentum_signal, "ZScoreSignal.parquet")
        try:
            
            if verbose == True: print("Trying to find data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, collecting it")
            df_out = (self.get_regression_signals().assign(
                group_var = lambda x: x.security + " " + x.window).
                groupby("group_var").
                apply(self._get_zscore, window).
                reset_index(drop = True).
                drop(columns = ["group_var"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
df = MomentumCreditSignals().get_regression_signals()
df = MomentumCreditSignals().generate_regression_zscore()