#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:14:46 2024

@author: diegoalvarez
"""
import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

from CDSIndexCollect import CreditDefaultSwapDataCollector
from pykalman import KalmanFilter

'''
Trend-Following and Mean-reversion strategies on credit indices: Algorithmic 
trading strategies using kalman filters
'''

class KalmanCreditDefaultSignals(CreditDefaultSwapDataCollector):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.cds_signals = os.path.join(self.data_path, "KalmanCDSSignals")
        if os.path.exists(self.cds_signals) == False: os.makedirs(self.cds_signals)
        
        self.lookback_windows = [20, 60]
        
    def _signal_preprocess(self, df: pd.DataFrame) -> pd.DataFrame: 
        
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
            resid      = lambda x: x.lag_smooth - x.log_spread).
            reset_index(drop = True).
            reset_index().
            rename(columns = {"index": "time"}).
            assign(time = lambda x: x.time + 1).
            drop(columns = ["smooth", "lag_smooth", "spread", "px"]).
            dropna())
        
        return df_out
        
    def signal_preprocess(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.cds_signals, "KalmanCDSPreprocess.parquet")
        
        try:
            
            if verbose == True: print("Seaching for CDS Pre-process data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, collecting it")
            df_out = (self.get_cds_indices().groupby(
                "security").
                apply(self._signal_preprocess).
                reset_index(drop = True).
                dropna())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _ols(self, df: pd.DataFrame, window) -> pd.DataFrame: 
        
        df_tmp = df.sort_values("date")
        
        model = (RollingOLS(
            endog  = df_tmp.value,
            exog   = sm.add_constant(df_tmp.time),
            window = window).
            fit())
        
        df_out = (model.params.assign(
            date = df_tmp.date).
            assign(
                signal       = lambda x: x.time.shift(),
                window       = window,
                signal_group = "regression").
            dropna()
            [["signal", "date", "window", "signal_group"]])
        
        return df_out
    
    def _run_ols(self, df: pd.DataFrame, windows: list) -> pd.DataFrame: 
        
        df_out = (pd.concat([
            self._ols(df, window)
            for window in windows]))
        
        return df_out
    
    def _zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                roll_mean    = lambda x: x.value.ewm(span = window, adjust = False).mean(),
                roll_std     = lambda x: x.value.ewm(span = window, adjust = False).std(),
                z_score      = lambda x: (x.value - x.roll_mean) / x.roll_std,
                signal       = lambda x: x.z_score.shift(),
                window       = window,
                signal_group = "z_score").
            drop(columns = ["roll_mean", "roll_std", "z_score"]))
        
        return df_out
    
    def _get_zscore(self, df: pd.DataFrame, windows: list) -> pd.DataFrame: 
        
        df_out = (pd.concat([
            self._zscore(df, window)
            for window in windows]))
        
        return df_out
    
    def _get_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                px_diff = lambda x: x.px.diff(),
                px_rtn  = lambda x: x.px.pct_change()))
        
        return df_out
    
    def generate_signal(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.cds_signals, "KalmanCDSSignals.parquet")
        
        try:
            
            if verbose == True: print("Searchuing for CDS Signals")
            df_combined = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it")
            df_longer = (self.signal_preprocess().melt(
                id_vars = ["time", "date", "security"]).
                assign(group_var = lambda x: x.security + "+" + x.variable))
            
            df_regression = (df_longer.groupby(
                "group_var").
                apply(self._run_ols, self.lookback_windows).
                reset_index().
                drop(columns = ["level_1"]).
                assign(
                    security  = lambda x: x.group_var.str.split("+").str[0],
                    input_var = lambda x: x.group_var.str.split("+").str[-1]).
                drop(columns = ["group_var"]))
            
            df_zscore = (df_longer.drop(
                columns = ["time"]).
                groupby("group_var").
                apply(self._get_zscore, self.lookback_windows).
                reset_index(drop = True).
                drop(columns = ["value", "group_var"]).
                dropna().
                rename(columns = {"variable": "input_var"}))
            
            df_combined = pd.concat([df_regression, df_zscore])
            
            if verbose == True: print("Saving data\n")
            df_combined.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_combined
    
    def backtest(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.cds_signals, "Backtest.parquet")
        
        try:
            
            if verbose == True: print("Trying to find backtest")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data generating it")
            df_out = (self.get_cds_indices().groupby(
                "security").
                apply(self._get_rtn).
                reset_index(drop = True).
                drop(columns = ["log_spread", "px", "spread"]).
                melt(id_vars = ["date", "security"]).
                rename(columns = {
                    "variable": "rtn_type",
                    "value"   : 'rtn_val'}).
                merge(right = self.generate_signal(), how = "inner", on = ["date", "security"]).
                assign(signal_rtn = lambda x: np.sign(x.signal) * x.rtn_val))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
            
    
def main() -> None: 
            
    df = KalmanCreditDefaultSignals().signal_preprocess(verbose = True)
    df = KalmanCreditDefaultSignals().generate_signal(verbose = True)
    df = KalmanCreditDefaultSignals().backtest(verbose = True)
    
if __name__ == "__main__": main()