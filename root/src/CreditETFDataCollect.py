#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:27:11 2024

@author: diegoalvarez
"""
import os
import numpy as np
import pandas as pd

class CreditETFDataCollector:
    
    def __init__(self) -> None:
        
        self.root_path  = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path  = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.raw_path   = os.path.join(self.data_path, "RawData")
        self.clean_path = os.path.join(self.data_path, "CleanedData")
        
        if os.path.exists(self.data_path)  == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path)   == False: os.makedirs(self.raw_path)
        if os.path.exists(self.clean_path) == False: os.makedirs(self.clean_path)
        
        self.bbg_repo = r"C:\Users\Diego\Desktop\app_prod\BBGData"
        if os.path.exists(self.bbg_repo) == False: 
            self.bbg_repo = r"/Users/diegoalvarez/Desktop/BBGData"
        
        self.bbg_ticker_path = os.path.join(self.bbg_repo, "root", "BBGtickers.xlsx")
        self.tickers         = (pd.read_excel(
            io = self.bbg_ticker_path, sheet_name = "ETFIndices").
            query("subcategory == 'Bond ETF'").
            assign(ticker = lambda x: x.Security.str.split(" ").str[0]).
            ticker.
            drop_duplicates().
            to_list())
    
    def _diff_clean(
            self,
            df          : pd.DataFrame, 
            long_window : int, 
            short_window: int) -> pd.DataFrame: 
    
        df_out = (df.sort_values(
            "date").
            assign(
                long_mean  = lambda x: x.value.diff().rolling(window = long_window).mean(),
                long_std   = lambda x: x.value.diff().rolling(window = long_window).std(),
                z_score    = lambda x: np.abs((x.value.diff() - x.long_mean) / x.long_std),
                rep_mean   = lambda x: x.value.rolling(window = long_window // 10).mean(),
                rep_val    = lambda x: np.where(x.z_score > 2.5, x.rep_mean, x.value)).
            dropna()
            [["date", "value", "rep_val"]].
            assign(
                short_mean = lambda x: x.rep_val.diff().rolling(window = short_window).mean(),
                short_std  = lambda x: x.rep_val.diff().rolling(window = short_window).std(),
                z_score    = lambda x: np.abs((x.rep_val.diff() - x.short_mean) / x.short_std),
                rep_mean   = lambda x: x.rep_val.rolling(window = short_window // 10).mean(),
                rep_val    = lambda x: np.where(
                    x.z_score > 2.5, 
                    x.rep_val.shift() + x.rep_val.diff().mean(), 
                    x.rep_val))
            [["date", "value", "rep_val"]].
            assign(security = df.name[0]))
        
        return df_out
        
    def _px_clean(
            self, 
            df          : pd.DataFrame, 
            long_window : int, 
            short_window: int) -> pd.DataFrame: 
    
        df_out = (df.sort_values(
            "date").
            assign(
                long_mean  = lambda x: x.value.rolling(window = long_window).mean(),
                long_std   = lambda x: x.value.rolling(window = long_window).std(),
                z_score    = lambda x: np.abs((x.value - x.long_mean) / x.long_std),
                rep_mean   = lambda x: x.value.rolling(window = long_window // 10).mean(),
                rep_val    = lambda x: np.where(x.z_score > 2.5, x.rep_mean, x.value)).
            dropna()
            [["date", "value", "rep_val"]].
            assign(
                short_mean = lambda x: x.rep_val.rolling(window = short_window).mean(),
                short_std  = lambda x: x.rep_val.rolling(window = short_window).std(),
                z_score    = lambda x: np.abs((x.rep_val - x.short_mean) / x.short_std),
                rep_mean   = lambda x: x.rep_val.rolling(window = short_window // 10).mean(),
                rep_val    = lambda x: np.where(x.z_score > 2.5, x.rep_val.shift() + x.rep_val.diff().mean(), x.rep_val))
            [["date", "value", "rep_val"]].
            assign(security = df.name[0]))
    
        return df_out
    
    
    def get_bond_fundamentals(
            self,
            short_window: int = 10,
            long_window : int = 200,
            verbose     : bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "CreditETFFundamentals.parquet")
        try:
            
            if verbose == True: print("Trying to find data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it now")
            
            raw_path = os.path.join(self.bbg_repo, "ETFIndices", "BondPricing")
            paths    = [os.path.join(raw_path, path) for path in os.listdir(raw_path)]
            df_out   = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                groupby(["security", "variable"]).
                apply(self._diff_clean, long_window, short_window, include_groups = False).
                drop(columns = ["security"]).
                reset_index().
                drop(columns = ["level_2"]).
                groupby(["security", "variable"]).
                apply(self._px_clean, long_window, short_window, include_groups = False).
                drop(columns = ["security"]).
                reset_index().
                drop(columns = ["level_2"]))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def get_px(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "CreditETFPrices.parquet")
        try:
            
            if verbose == True: print("Trying to find data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it now")
        
            raw_path = os.path.join(self.bbg_repo, "data")
            paths    = [os.path.join(raw_path, path + ".parquet") for path in self.tickers]
            df_out   = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                drop(columns = ["variable"]).
                rename(columns = {"value": "px"}))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.assign(
            px_diff = lambda x: x.px.diff(),
            px_pct  = lambda x: x.px.pct_change(),
            px_bps  = lambda x: x.px_diff / x.mod_dur).
            dropna())
        
        return df_out

    def prep_data(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.clean_path, "CreditETFData.parquet")
        try:
            
            if verbose == True: print("Trying to find credit ETF Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, collecting it now")
            df_out = (self.get_bond_fundamentals().merge(
                right = self.get_px(), how = "inner", on = ["date", "security"]).
                drop(columns = ["value"]).
                pivot(index =  ["date", "px", "security"], columns = "variable", values = "rep_val").
                reset_index().
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                rename(columns = {
                    "YAS_YLD_SPREAD"         : "yld_sprd",
                    "YAS_MOD_DUR"            : "mod_dur", 
                    "YAS_ISPREAD_TO_GOVT"    : "ispread",
                    "AVERAGE_WEIGHTED_COUPON": "WAC",
                    "YAS_BOND_YLD"           : "yld"}).
                query("security == security.min()").
                groupby("security").
                apply(self._get_rtn, include_groups = False).
                reset_index().
                drop(columns = ["level_1"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
def main() -> None:
    
    CreditETFDataCollector().get_px(verbose = True)
    CreditETFDataCollector().get_bond_fundamentals(verbose = True)
    CreditETFDataCollector().prep_data(verbose = True)
    
if __name__ == "__main__": main()