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
        
        self.root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path = os.path.join(self.repo_path, "data")
        self.raw_path  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path)  == False: os.makedirs(self.raw_path)
        
        self.bbg_repo = r"C:\Users\Diego\Desktop\app_prod\BBGData"
        if os.path.exists(self.bbg_repo) == False: self.bbg_repo = r"/Users/diegoalvarez/Desktop/BBGData"
        
        self.bbg_ticker_path = os.path.join(self.bbg_repo, "root", "BBGtickers.xlsx")
        self.tickers         = (pd.read_excel(
            io = self.bbg_ticker_path, sheet_name = "ETFIndices").
            query("subcategory == 'Bond ETF'").
            assign(ticker = lambda x: x.Security.str.split(" ").str[0]).
            ticker.
            drop_duplicates().
            to_list())
        
    def _get_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                px_rtn  = lambda x: x.px.pct_change(),
                px_diff = lambda x: x.px.diff()))
        
        return df_out
        
    def _get_px(self) -> pd.DataFrame: 
        
        paths = ([
            os.path.join(self.bbg_repo, "data", ticker + ".parquet")
            for ticker in self.tickers])
        
        df_out = (pd.read_parquet(
            path = paths, engine = "pyarrow").
            drop(columns = ["variable"]).
            rename(columns = {"value": "px"}).
            groupby("security").
            apply(self._get_rtn).
            reset_index(drop = True))
        
        return df_out
    
    def _get_yld(self) -> pd.DataFrame:
        
        paths = ([
            os.path.join(self.bbg_repo, "ETFIndices", "BondPricing", ticker + ".parquet")
            for ticker in self.tickers])
        
        df_out = (pd.read_parquet(
            path = paths, engine = "pyarrow").
            pivot(index = ["date", "security"], columns = "variable", values = "value").
            reset_index().
            rename(columns = {
                "AVERAGE_WEIGHTED_COUPON": "WAC",
                "YAS_MOD_DUR"            : "MOD_DUR",
                "YAS_BOND_YLD"           : "YAS_YLD",
                "YAS_ISPREAD_TO_GOVT"    : "YAS_ISPREAD",
                "YAS_YLD_SPREAD"         : "YAS_SPREAD"}))
        
        return df_out
    
    def get_credit_etfs(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "CreditETFData.parquet")
        try:
            
            if verbose == True: print("Trying to find Credit ETF Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
        
            if verbose == True: print("Couldn't find data, collecting it")
            df_out = (self._get_px().merge(
                right = self._get_yld(), how = "inner", on = ["date", "security"]).
                assign(
                    px_bps          = lambda x: x.px_diff / x.MOD_DUR,
                    security        = lambda x: x.security.str.split(" ").str[0],
                    log_yas_yld     = lambda x: np.log(x.YAS_YLD),
                    log_ispread_yld = lambda x: np.log(x.YAS_ISPREAD),
                    log_yas_spread  = lambda x: np.log(x.YAS_SPREAD)))
            
            if verbose == True: print("Saving Data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
if __name__ == "__main__": CreditETFDataCollector().get_credit_etfs(verbose = True)