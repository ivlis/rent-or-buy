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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import datetime

import gzip


# -

def load_data_by_urban_codes(state='MA', min_year=2011):
    # load sales/rent data by county
    with gzip.open('./data/zecon/County_time_series.csv.gz', 'rb') as fgz: 
        df_county = pd.read_csv(fgz)
    df_county.Date = pd.to_datetime(df_county.Date)
    df_county = df_county[df_county.Date.dt.year>=min_year]
    
    #load county codes
    df_county_codes = pd.read_excel('./data/NCHSURCodes2013.xlsx')
    df_county_codes.rename(columns={
        'FIPS code': 'RegionName', 
        '2013 code': 'urban_code', 
        'State Abr.': 'state', 
        'County name': 'county_name'},
        inplace=True)
    df_county_codes.drop(['CBSA title', 'CBSA 2012 pop', '2006 code', '1990-based code', 'County 2012 pop'], axis=1, inplace=True)
    df_county_codes = df_county_codes[df_county_codes.state==state]
    
    df_listings_count = pd.merge(df_county, df_county_codes, on='RegionName').groupby('RegionName').agg(
    {'MedianListingPrice_1Bedroom':'count', 'MedianRentalPrice_1Bedroom':'count'})

    # Select only counties with enough data
    df_considered_counties = pd.merge(df_listings_count[(df_listings_count != 0).all(axis=1)],
         df_county_codes, left_index=True, right_on='RegionName')
    df_considered_counties.drop(['MedianListingPrice_1Bedroom','MedianRentalPrice_1Bedroom'], axis=1, inplace=True)
    
    
    # We need better weighting here
    df_urban_code = pd.merge(df_county, df_considered_counties[['urban_code', 'RegionName']], how='inner', on='RegionName')\
        .groupby(['Date', 'urban_code'])\
        .mean().reset_index()
    df_urban_code.drop('RegionName',axis=1, inplace=True) # does not make sense now
    
    return df_urban_code, df_considered_counties
    

def load_loan_apr_monthly():
    loan_apr = pd.read_csv('./data/MORTGAGE30US.csv', parse_dates=[0], index_col=0)
    loan_apr.rename(columns={'DATE':'Date', 'MORTGAGE30US':'apr'}, inplace=True)
    loan_apr.apr = loan_apr.apr/100
    loan_apr = loan_apr.resample('M').mean()
    return loan_apr


def load_hpi_master(place_name=None):
    hpi_master = pd.read_csv('./data/HPI_master.csv')
    hpi_master = hpi_master.rename(columns = {'yr': 'year', 'period':'month', 'index_sa': 'hpi_sa', 'index_nsa': 'hpi_nsa'})
    hpi_master['day'] = 1
    hpi_master['Date'] = pd.to_datetime(hpi_master[['year', 'month', 'day']]) + pd.tseries.offsets.MonthEnd()
    hpi_master = hpi_master[(hpi_master.hpi_type=='traditional') & (hpi_master.hpi_flavor =='purchase-only')]
    hpi_master = hpi_master.drop(columns = ['hpi_type', 'hpi_flavor', 'year', 'month', 'day'])
#     hpi_master = hpi_master.rename(columns={'index_sa':'hpi_sa'})
    hpi = hpi_master[
        (hpi_master.place_name==place_name) & (hpi_master.frequency == 'monthly')
    ] if place_name else hpi_master
    
    return hpi


