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
import pandas as pd
from sklearn.pipeline import Pipeline

from .helper import (
    load_houseprices_by_urban_codes,
    load_hpi_master,
    load_loan_apr_monthly,
    load_fmr_by_region,
)

from .transformations import Derivatives, SavgolFilter, SelectFeatures
from .model import Model


# +
class RentModel(Model):

    _preprocess = Pipeline(
        steps=[
            ("feature_selection", SelectFeatures(["Date", "apr", "fmr"])),
            ("savgol_apr", SavgolFilter(column="apr", window_length=11, poly_order=3)),
            ("div_apr", Derivatives(column="apr_savgol", order=4)),
        ]
    )

    def _load_features(self):
        house_prices, selected_counties = load_houseprices_by_urban_codes(state="MA")

        fmr_index = load_fmr_by_region(selected_counties)

        loan_apr = load_loan_apr_monthly()

        self.features = fmr_index.merge(loan_apr, left_on='Date', right_index=True)

    def _load_targets(self):
        house_prices, selected_counties = load_houseprices_by_urban_codes(state="MA")
        rent_prices_combined = []
        for rooms in range(1, 5):
            df = house_prices[
                ["Date", "urban_code", f"MedianRentalPrice_{rooms}Bedroom"]
            ].copy()
            df["rooms"] = rooms
            df = df.rename(columns={f"MedianRentalPrice_{rooms}Bedroom": "target"})
            rent_prices_combined.append(df)
        rent_prices_combined = pd.concat(rent_prices_combined, axis=0)
        rent_prices_combined = rent_prices_combined.dropna()
        rent_prices_combined = rent_prices_combined.reset_index(drop=True)
        rent_prices_combined = rent_prices_combined[
            (rent_prices_combined.Date.dt.year > 2012)
            | (rent_prices_combined.rooms != 1)
            | (rent_prices_combined.urban_code != 2)
        ]
        self.targets = rent_prices_combined


# -
