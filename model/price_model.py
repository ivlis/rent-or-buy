# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python [conda env:zillow]
#     language: python
#     name: conda-env-zillow-py
# ---

# +
import numpy as np
import pandas as pd

import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV, KFold

from .helper import load_data_by_urban_codes, load_loan_apr_monthly, load_hpi_master
from .periodic_model import GeneratePeriodic, Derivatives, SavgolFilter, SelectFeatures


# +
class PriceModel:
    
    _lin_reg = Pipeline(
                    steps =[
                            ('select_features', SelectFeatures()),
                            ('period_generation', GeneratePeriodic()),
                            ('scaling', StandardScaler()),
                            ('model', Lasso())
                           ] 
                   )
  
    _smoothing = Pipeline(steps=[
                        ('savgol_apr', SavgolFilter(column='apr', window_length=11, poly_order=3)),
                        ('savgol_hpi', SavgolFilter(column='hpi_sa', window_length=11, poly_order=3))
                        ])
    _derivatives = Pipeline(steps=[
                        ('div_apr', Derivatives(column='apr_savgol', order=2)),
                        ])

    _preprocess = Pipeline(steps=[
        ('smoothing', _smoothing),
        ('derivatives', _derivatives)
    ])

    def _load_features(self):
        loan_apr = load_loan_apr_monthly()
        hpi = load_hpi_master('New England Division')
        features = loan_apr.merge(hpi[['Date', 'hpi_sa']], right_on='Date', left_index=True)
        
        self.prp_features = self._preprocess.transform(features)  #Prepocessed features
        
    def _load_targets(self):
        house_prices, selected_counties = load_data_by_urban_codes(state='MA')
        house_prices_combined = []
        for rooms in range(1, 5):
            df = house_prices[['Date', 'urban_code', f'MedianListingPrice_{rooms}Bedroom']].copy()
            df['rooms'] = rooms
            df = df.rename(columns = {f'MedianListingPrice_{rooms}Bedroom': 'ListingPrice'})
            house_prices_combined.append(df)
        house_prices_combined = pd.concat(house_prices_combined, axis=0)
        house_prices_combined = house_prices_combined.dropna()
        house_prices_combined.ListingPrice /= 1000
        self.house_prices_combined = house_prices_combined
        
    def __init__(self):
        self._load_features()
        self._load_targets()
        self.features_and_targets = self.house_prices_combined.merge(self.prp_features, on='Date')
        
    def _fit_market(self, urban_code, rooms):
        Xy = self.getXy(urban_code, rooms)
        y = Xy['ListingPrice']
        parameters = {'period_generation__harmonics': list(range(0,7)),
                      'model__alpha': np.logspace(-1,3,3)}
        cvgrid = GridSearchCV(self._lin_reg, parameters, cv=KFold(5, shuffle=True, random_state=20), iid=False)
        cvgrid.fit(Xy,y)
        return cvgrid
        
    def getXy(self, urban_code, rooms):
        df = self.features_and_targets
        return df[(df.urban_code == urban_code) & (df.rooms==rooms)]

    def get_model(self, urban_code, rooms):
        df = self.models
        return df.loc[(df.urban_code == urban_code) & (df.rooms==rooms), 'model'].iloc[0]
    
    def fit_all(self):
        models = pd.DataFrame()
        
        for r in range(1,5):
            for uc in range(1,4):
                model = self._fit_market(urban_code=uc, rooms=r)
                models = models.append({'rooms': r, 'urban_code': uc, 'model': model}, ignore_index=True)
        
        models['rooms'] = models['rooms'].astype(int)
        models['urban_code'] = models['urban_code'].astype(int)
        self.models=models
        return None
    
    def predict(self, features):
        
        prp_features = self._preprocess.transform(features.copy())  #Prepocessed features
        
        all_y_hats = []
        
        for r in range(1,5):
            for uc in range(1,4):
                y_hat = self.get_model(urban_code=uc, rooms=r).predict(prp_features)
                out = pd.DataFrame()
                out['Date'] = prp_features.Date 
                out['y_hat'] = y_hat
                out['urban_code'] = uc
                out['rooms'] = r
                
                all_y_hats.append(out)
                
        
        all_y_hats = pd.concat(all_y_hats, axis=0)
        
        all_y_hats.loc[all_y_hats.urban_code==1, 'density'] = 'urban'
        all_y_hats.loc[all_y_hats.urban_code==2, 'density'] = 'suburban'
        all_y_hats.loc[all_y_hats.urban_code==3, 'density'] = 'rural'
        
        return all_y_hats
        
                
      
# -

