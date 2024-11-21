# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 06:36:58 2024

@author: Diego
"""

import os
import pandas as pd

class CreditDataCollect:
    
    def __init__(self):
        
        self.root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.note_path = os.path.join(self.root_path, "notebooks")
        self.data_path = os.path.join(self.repo_path, "data")
        self.raw_path  = os.path.join(self.data_path, "RawData")
        self.prep_path = os.path.join(self.data_path, "data_guide.xlsx")
        
        if os.path.exists(self.note_path) == False: os.makedirs(self.note_path)
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path)  == False: os.makedirs(self.raw_path)
        
        self.bbg_data_path   = r"C:\Users\Diego\Desktop\app_prod\BBGData\root\BBGTickers.xlsx"
        self.bbg_data        = r"C:\Users\Diego\Desktop\app_prod\BBGData\data"
        self.data_prep_path  = os.path.join(self.data_path, "data_guide.xlsx")
        self.credit_etf_path = r"C:\Users\Diego\Desktop\app_prod\BBGData\ETFIndices\BondPricing"
        
        # credit tickers
        self.df_credit_guide = (pd.read_excel(
            io = self.prep_path, sheet_name = "credit_indices"))
        
        # cds tickers from BBG data
        self.df_cds_tickers_raw = (pd.read_excel(
            io = self.bbg_data_path, sheet_name = "cds_tickers")
            [["Security", "Description"]].
            assign(file_name = lambda x: x.Description.str.replace(" ", "") + ".parquet"))
        
        # cds tickers from current repo
        self.df_cds_name = (pd.read_excel(
            io = self.prep_path, sheet_name = "cds_indices").
            assign(spec_name = lambda x: x.name + " " + x.maturity))

    def get_cds(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.raw_path, "CDSIndices.parquet")
        try:
            
            if verbose == True: print("Trying to find CDS Indices")
            df_cds = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, now collecting it")
        
            df_cds_raw = pd.concat([
                pd.read_parquet(
                    path = os.path.join(self.bbg_data, name), 
                    engine = "pyarrow") 
                for name in self.df_cds_tickers_raw.file_name.to_list()])
            
            df_cds_name = (pd.read_excel(
                io = self.data_prep_path, sheet_name = "cds_indices").
                assign(spec_name = lambda x: x.name + " " + x.maturity))
            
            matched_names = (df_cds_name[
                ["spec_name", "maturity"]].
                groupby("spec_name").
                agg("count").
                query("maturity == 2").
                index.
                to_list())
            
            security = (df_cds_name.query(
                "spec_name == @matched_names").
                security.
                to_list())
            
            df_cds = (df_cds_raw.query(
                "security == @security").
                merge(right = df_cds_name, how = "inner", on = ["security"]).
                assign(unique_name = lambda x: x.name + " " + x.maturity).
                query("spec_name == @matched_names").
                drop(columns = ["spec_name", "group", "name", "variable"]).
                assign(
                    date     = lambda x: pd.to_datetime(x.date).dt.date,
                    variable = lambda x: x.security.str.split(" ").str[-2]).
                drop(columns = ["security"]))
            
            if verbose == True: print("Saving data")
            df_cds.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_cds
    
    def get_credit_etfs(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "CreditETFs.parquet")
        try:
            
            if verbose == True: print("Trying to find credit ETF Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, generating it")    
        
            etf_fundamentals_path = ([
                os.path.join(self.credit_etf_path, path)
                for path in os.listdir(self.credit_etf_path)])
            
            df_fundamentals = (pd.read_parquet(
                path = etf_fundamentals_path, engine = "pyarrow"))
            
            etf_prices_path = ([
                os.path.join(self.bbg_data, path)
                for path in os.listdir(self.credit_etf_path)])
            
            df_prices = (pd.read_parquet(
                path = etf_prices_path, engine = "pyarrow"))
            
            df_combined = (pd.concat(
                [df_fundamentals, df_prices]).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date))
            
            df_out = (pd.read_excel(
                io = self.bbg_data_path, sheet_name = "tickers").
                rename(columns = {"Security": "security"}).
                merge(right = df_combined, how = "inner", on = ["security"]).
                drop(columns = ["Category", "Subcategory", "Frequency"]).
                assign(security = lambda x: x.security.str.split(" ").str[0]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
        
def main() -> None:     
    
    CreditDataCollect().get_cds(verbose = True)
    CreditDataCollect().get_credit_etfs(verbose = True)
    
if __name__ == "__main__": main()