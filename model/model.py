import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .transformations import GeneratePeriodic


# +
class Model:

    _lin_reg = Pipeline(
        steps=[
            ("period_generation", GeneratePeriodic()),
            ("scaling", StandardScaler()),
            ("model", Lasso()),
        ]
    )

    def __init__(self):
        self._load_features()
        self._load_targets()

        self.prp_features = self._preprocess.transform(
            self.features
        )  # Prepocessed features

        self.features_and_targets = self.targets.merge(self.prp_features, on="Date")

        self.density_dict = pd.DataFrame(
            [{1: "urban", 2: "suburb", 3: "rural"}]
        ).T.rename(columns={0: "density"})

        self.features_and_targets = self.features_and_targets.merge(
            self.density_dict, how="left", left_on="urban_code", right_index=True
        )

    def _fit_market(self, urban_code, rooms):
        X, y = self.getXy(urban_code, rooms)

        parameters = {
            "period_generation__harmonics": list(range(0, 7)),
            "model__alpha": np.logspace(-1, 3, 3),
        }
        cvgrid = GridSearchCV(
            self._lin_reg,
            parameters,
            cv=KFold(5, shuffle=True, random_state=20),
            iid=False,
        )

        cvgrid.fit(X, y)

        return cvgrid

    def getXy(self, urban_code, rooms):
        df = self.features_and_targets
        df = df[(df.urban_code == urban_code) & (df.rooms == rooms)]
        y = df.target
        X = df.drop(columns=["target", "rooms", "urban_code", "density"])
        return X, y

    def get_model(self, urban_code, rooms):
        df = self.models
        return df.loc[
            (df.urban_code == urban_code) & (df.rooms == rooms), "model"
        ].iloc[0]

    def fit_all(self):
        models = pd.DataFrame()

        for r in range(1, 5):
            for uc in range(1, 4):
                model = self._fit_market(urban_code=uc, rooms=r)
                models = models.append(
                    {"rooms": r, "urban_code": uc, "model": model}, ignore_index=True
                )

        models["rooms"] = models["rooms"].astype(int)
        models["urban_code"] = models["urban_code"].astype(int)

        models = models.merge(
            self.density_dict, how="left", left_on="urban_code", right_index=True
        )
        self.models = models
        return None

    def predict(self, features):

        prp_features = self._preprocess.transform(
            features.copy()
        )  # Prepocessed features

        all_y_hats = []

        for r in range(1, 5):
            for uc in range(1, 4):
                y_hat = self.get_model(urban_code=uc, rooms=r).predict(prp_features)
                out = pd.DataFrame()
                out["Date"] = prp_features.Date
                out["y_hat"] = y_hat
                out["urban_code"] = uc
                out["rooms"] = r

                all_y_hats.append(out)

        all_y_hats = pd.concat(all_y_hats, axis=0)

        return all_y_hats


# -
