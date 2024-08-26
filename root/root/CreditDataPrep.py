# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 06:36:58 2024

@author: Diego
"""

import os
import pandas as pd

class CreditDataPrep:
    
    def __init__(self):
        
        self.data_path = r"C:\Users\Diego\Desktop\app_prod\research\CreditMomentum\data"
        self.credit_path = r"C:\Users\Diego\Desktop\app_prod\BBGData\credit_indices_data"
        self.bbg_data_path = r"C:\Users\Diego\Desktop\app_prod\BBGData\root\BBGTickers.xlsx"
        self.bbg_data = r"C:\Users\Diego\Desktop\app_prod\BBGData\data"
        self.data_prep_path = os.path.join(self.data_path, "data_guide.xlsx")
        
        # credit tickers
        self.df_credit_guide = (pd.read_excel(
            io = self.data_prep_path, sheet_name = "credit_indices"))
        
        # cds tickers from BBG data
        self.df_cds_tickers_raw = (pd.read_excel(
            io = self.bbg_data_path, sheet_name = "cds_tickers")
            [["Security", "Description"]].
            assign(file_name = lambda x: x.Description.str.replace(" ", "") + ".parquet"))
        
        # cds tickers from current repo
        self.df_cds_name = (pd.read_excel(
            io = self.data_prep_path, sheet_name = "cds_indices").
            assign(spec_name = lambda x: x.name + " " + x.maturity))
        
    def _get_credit(self):
        
        df_credit_raw = (pd.read_parquet(
            path = self.credit_path, engine = "pyarrow").
            drop(columns = ["variable"]).
            merge(right = self.df_credit_guide, how = "inner", on = ["security"]))
        
        return df_credit_raw

    def _get_cds(self):
        
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
            query("spec_name == @matched_names"))
        
        return df_cds
    
    def get_data(self):
        
        keep_cols = ["date", "security", "value", "group_short", "asset_class"]
        
        df_credit = (self._get_credit().assign(
            asset_class = "credit")
            [keep_cols])
        
        df_cds = (self._get_cds().rename(
            columns = {"group": "group_short"}).
            assign(asset_class = "CDS")
            [keep_cols])
        
        df_out = pd.concat([df_credit, df_cds])
        return df_out