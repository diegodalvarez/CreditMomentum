#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:18:04 2024

@author: diegoalvarez
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

from pykalman import KalmanFilter
from CreditDataCollect import CreditDataCollect

class SignalGenerator(CreditDataCollect):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.signal_path = os.path.join(self.data_path, "Signals")
        
        if os.path.exists(self.signal_path) == False: os.makedirs(self.signal_path)
        
    def _get_kalman_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_tmp        = df.sort_values("date").dropna()
        kalman_filter = KalmanFilter(
            transition_matrices      = [1],
            observation_matrices     = [1],
            initial_state_mean       = 0,
            initial_state_covariance = 1,
            observation_covariance   = 1,
            transition_covariance    = 0.01)
        
        state_means, state_covariances = kalman_filter.filter(df_tmp.log_spread)
        df_out = (df_tmp.assign(
            smooth     = state_means,
            lag_smooth = lambda x: x.smooth.shift(),
            resid      = lambda x: x.lag_smooth - x.log_spread,
            lag_resid  = lambda x: x.resid.shift()))
        
        return df_out
        
    def get_cds_kalman(self, verbose: int = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "KalmanSignals.paruqet")
        try:
            
            if verbose == True: print("Trying to find Kalman Signals")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find Kalman Data, collecting")
        
            df_out = (self.get_cds().query(
                "variable == 'SPRD'").
                rename(columns = {"value": "spread"}).
                assign(log_spread = lambda x: np.log(x.spread)).
                groupby("unique_name").
                apply(self._get_kalman_signal).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _ewma(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.assign(
            trend_signal = lambda x: x.value.ewm(span = window, adjust = False).mean(),
            lag_signal   = lambda x: x.trend_signal.shift(),
            signal_name  = window).
            dropna())
        
        return df_out
    
    def _get_ewmas(self, df: pd.DataFrame, windows: list) -> pd.DataFrame: 
        
        df_out = (pd.concat([
            self._ewma(df, window) for window in windows]))
        
        return df_out
    
    def get_kalman_ewma_signal(
            self, 
            windows: list = [20, 60],
            verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "KalmanEWMASignal.parquet")
        try:
            
            if verbose == True: print("Tryign to find Kalman EWMA Signals")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find EWMA data, collecting")
        
            df_out = (self.get_cds_kalman()[
                ["date", "unique_name", "lag_smooth", "lag_resid"]].
                dropna().
                melt(id_vars = ["date", "unique_name"]).
                query("variable != 'lag_smooth'").
                groupby("unique_name").
                apply(self._get_ewmas, windows).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

SignalGenerator().get_kalman_ewma_signal(verbose = True)
#SignalGenerator().get_cds_kalman(verbose = True)       
#SignalGenerator().get_kalman_ewma_signal(verbose = True)