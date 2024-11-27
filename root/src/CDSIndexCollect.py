#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:40:54 2024

@author: diegoalvarez
"""
import os
import numpy as np
import pandas as pd

class CreditDefaultSwapDataCollector:
    
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
        self.df_tickers = (pd.read_excel(
            io = self.bbg_ticker_path, sheet_name = "cds_tickers")
            [["Security", "Description"]].
            assign(file_name = lambda x: x.Description.str.replace(" ", "") + ".parquet"))
        
    def get_cds_indices(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "CreditDefaultSwapData.parquet")
        try:
            
            if verbose == True: print("Trying to find Credit Default Swap Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it")
            
            paths = ([
                os.path.join(self.bbg_repo, "data", name)
                for name in self.df_tickers.file_name.drop_duplicates().to_list()])
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                drop(columns = ["variable"]).
                assign(
                    second_last = lambda x: x.security.str.split(" ").str[-2],
                    variable    = lambda x: np.where(x.second_last == "PRC", x.second_last, "SPRD"),
                    front_name  = lambda x: x.security.str.split("GEN").str[0],
                    last_name   = lambda x: x.security.str.split("GEN").str[-1].str.strip().str.split(" ").str[0],
                    new_name    = lambda x: x.front_name.str.strip() + " " + x.last_name)
                [["date", "new_name", "variable", "value"]].
                rename(columns = {"new_name": "security"}).
                pivot(index = ["date", "security"], columns = "variable", values = "value").
                reset_index().
                dropna().
                rename(columns = {
                    "PRC" : "px",
                    "SPRD": "spread"}).
                assign(log_spread = lambda x: np.log(x.spread)))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
if __name__ == "__main__": CreditDefaultSwapDataCollector().get_cds_indices()