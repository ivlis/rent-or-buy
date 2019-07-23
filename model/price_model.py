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

import pandas as pd
from sklearn.pipeline import Pipeline

from .helper import (
    load_houseprices_by_urban_codes,
    load_hpi_master,
    load_loan_apr_monthly,
)
from .transformations import Derivatives, SavgolFilter, SelectFeatures
from .model import Model


# +
class PriceModel(Model):

    _preprocess = Pipeline(
        steps=[
            ("feature_selection", SelectFeatures(["Date", "apr", "hpi_sa"])),
            ("savgol_apr", SavgolFilter(column="apr", window_length=11, poly_order=3)),
            (
                "savgol_hpi",
                SavgolFilter(column="hpi_sa", window_length=11, poly_order=3),
            ),
            ("div_apr", Derivatives(column="apr_savgol", order=2)),
        ]
    )

    def _load_features(self):
        loan_apr = load_loan_apr_monthly()
        hpi = load_hpi_master("New England Division")
        features = loan_apr.merge(
            hpi[["Date", "hpi_sa"]], right_on="Date", left_index=True
        )

        self.features = features

    def _load_targets(self):
        house_prices, selected_counties = load_houseprices_by_urban_codes(state="MA")
        house_prices_combined = []
        for rooms in range(1, 5):
            df = house_prices[
                ["Date", "urban_code", f"MedianListingPrice_{rooms}Bedroom"]
            ].copy()
            df["rooms"] = rooms
            df = df.rename(columns={f"MedianListingPrice_{rooms}Bedroom": "target"})
            house_prices_combined.append(df)
        house_prices_combined = pd.concat(house_prices_combined, axis=0)
        house_prices_combined = house_prices_combined.dropna()
        house_prices_combined.target /= 1000
        self.targets = house_prices_combined


# -
