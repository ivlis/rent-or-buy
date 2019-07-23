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
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

# -


class SavgolFilter(BaseEstimator, TransformerMixin):
    def __init__(self, column=None, window_length=0, poly_order=0):
        self.column = column
        self.window_length = window_length
        self.poly_order = poly_order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        out = X.copy()
        out[self.column + "_savgol"] = savgol_filter(
            out[self.column], self.window_length, self.poly_order
        )
        out = out.drop(columns=[self.column])
        return out


class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features=0):
        self.harmonics = harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        out = X
        try:
            out = X.drop(columns=["ListingPrice"])
        except KeyError:
            pass
        return out[["Date", "apr_savgol", "hpi_sa_savgol", "apr_savgol_div_1"]]


class SelectFeaturesR(BaseEstimator, TransformerMixin):
    def __init__(self, harmonics=0):
        self.harmonics = harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        out = X
        try:
            out = X.drop(columns=["RentalPrice"])
        except KeyError:
            pass
        return out[["Date", "fmr", "apr_savgol", "apr_savgol_div_1"]]


class GeneratePeriodic(BaseEstimator, TransformerMixin):
    def __init__(self, harmonics=0):
        self.harmonics = harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        out = X.copy()
        out["month"] = out.Date.dt.month
        out = out.drop(columns=["Date"])
        for h in range(1, self.harmonics + 1):
            out[f"sin{h}_month"] = np.sin(h * 2 * np.pi * out.month / 12)
            out[f"cos{h}_month"] = np.cos(h * 2 * np.pi * out.month / 12)
        out = out.drop(columns=["month"])
        return out


class Derivatives(BaseEstimator, TransformerMixin):
    def __init__(self, column=None, order=0):
        self.column = column
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        out = X.copy()
        for d in range(1, self.order + 1):
            out[self.column + f"_div_{d}"] = np.gradient(
                out[self.column + ("" if d == 1 else f"_div_{d-1}")]
            )
        return out
