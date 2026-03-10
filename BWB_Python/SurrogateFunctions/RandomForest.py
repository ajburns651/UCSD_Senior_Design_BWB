# ────────────────────────────────────────────────
# 155A BWB - RF Surrogate Model
# ────────────────────────────────────────────────
"""
ld_surrogate.py
---------------
Random Forest surrogate model that predicts cruise L/D from BWB geometry
and mission inputs, trained on Monte Carlo data.

Usage
-----
# Train once and save:
    surrogate = LDSurrogate()
    surrogate.train("Monte_Carlo_Results_Expanded.csv")
    surrogate.save("ld_surrogate.pkl")

# Load and predict:
    surrogate = LDSurrogate.load("ld_surrogate.pkl")
    ld = surrogate.predict(
        spans      = [4.08, 6.50, 19.00, 2.00],
        root_chords= [43.00, 31.18, 9.00, 3.00],
        tip_chords = [31.18, 9.00, 3.00, 0.80],
        sweeps     = [62.00, 67.00, 37.00, 40.00],
        dihedrals  = [0.00, 0.00, 8.00, 9.25],
        range_m    = 7000 * 1852,
    )
    print(f"Predicted L/D: {ld:.3f}")
"""

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# Column order must match training feature order exactly
FEATURE_COLS = [
    "span_0",    "span_1",    "span_2",    "span_3",
    "root_0",    "root_1",    "root_2",    "root_3",
    "tip_0",     "tip_1",     "tip_2",     "tip_3",
    "sweep_0",   "sweep_1",   "sweep_2",   "sweep_3",
    "dihedral_0","dihedral_1","dihedral_2","dihedral_3",
    "range_m",
]

class LDSurrogate:
    """Random Forest surrogate for cruise L/D prediction."""

    def __init__(self, n_estimators=300, max_features="sqrt",
                 min_samples_leaf=2, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=1,
        )
        self.feature_cols = FEATURE_COLS
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, csv_path: str, test_size: float = 0.02, verbose: bool = True):
        """
        Parse the Monte Carlo CSV, extract features, and train the RF model.

        Parameters
        ----------
        csv_path   : Path to Monte_Carlo_Results_Expanded.csv
        test_size  : Fraction of data held out for evaluation
        verbose    : Print R² / MAE on the held-out test set
        """
        df = pd.read_csv(csv_path)
        X, y = self._parse_dataframe(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.model.fit(X_train, y_train)
        self._trained = True

        if verbose:
            y_pred = self.model.predict(X_test)
            print(f"[LDSurrogate] Training complete on {len(X_train)} samples.")
            print(f"  Test R²  : {r2_score(y_test, y_pred):.4f}")
            print(f"  Test MAE : {mean_absolute_error(y_test, y_pred):.4f} (L/D units)")
            imp = pd.Series(self.model.feature_importances_, index=self.feature_cols)
            print("  Top 5 features by importance:")
            for feat, val in imp.nlargest(5).items():
                print(f"    {feat:<14} {val:.4f}")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        spans:       list,
        root_chords: list,
        tip_chords:  list,
        sweeps:      list,
        dihedrals:   list,
        range_m:     float,
    ) -> float:
        """
        Predict cruise L/D for a single design point.

        Parameters
        ----------
        spans        : Section half-spans [m], length 4
        root_chords  : Section root chords [m], length 4
        tip_chords   : Section tip chords  [m], length 4
        sweeps       : Section sweep angles [deg], length 4
        dihedrals    : Section dihedral angles [deg], length 4
        range_m      : Mission range [m]

        Returns
        -------
        float : Predicted L/D
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        row = (
            list(spans)
            + list(root_chords)
            + list(tip_chords)
            + list(sweeps)
            + list(dihedrals)
            + [range_m]
        )
        X = pd.DataFrame([row], columns=self.feature_cols)
        return float(self.model.predict(X)[0])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict L/D for a DataFrame that already has columns matching FEATURE_COLS.
        Useful for parameter sweeps.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")
        return self.model.predict(df[self.feature_cols])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Serialize the trained surrogate to a .pkl file."""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_cols": self.feature_cols}, f)
        print(f"[LDSurrogate] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LDSurrogate":
        """Load a previously saved surrogate."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model = payload["model"]
        obj.feature_cols = payload["feature_cols"]
        obj._trained = True
        print(f"[LDSurrogate] Loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_nums(s: str):
        return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]

    def _parse_dataframe(self, df: pd.DataFrame):
        """Extract feature matrix X and target y from the raw CSV DataFrame."""
        dim_parsed = df["Dimension Inputs"].apply(self._parse_nums)
        assert all(len(x) == 20 for x in dim_parsed), \
            "Expected 20 values per row in 'Dimension Inputs' (5 arrays × 4 sections)."

        X = pd.DataFrame(dim_parsed.tolist(), columns=self.feature_cols[:-1])
        X["range_m"] = df["Range"].values
        y = df["L_D_Cruise"].values
        return X, y


# ──────────────────────────────────────────────────────────────────────
# Function To Load Surrogate Model and Compute L/D Prediction
# ──────────────────────────────────────────────────────────────────────
def predict_cruise_ld(spans, rootcs, tipcs, sweeps, dihedrals, range):

    surrogate = LDSurrogate.load("SurrogateFunctions/ld_surrogate.pkl")
    ld = surrogate.predict(spans = spans, root_chords = rootcs, tip_chords = tipcs, sweeps = sweeps, dihedrals = dihedrals, range_m = range)

    return ld


# ──────────────────────────────────────────────────────────────────────
# Quick Script To Train The RF Model
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else "../Monte_Carlo_Backup_Temp.csv"

    # Train
    surrogate = LDSurrogate()
    surrogate.train(csv, verbose=True)
    surrogate.save("ld_surrogate.pkl")

    # Predict
    #spans =  [4.94243282, 7.36036455, 20.91146521, 2.42656384]
    #root_chords = [43.7511432 , 31.62556581 , 4.14286904,  2.6707789 ]
    #tip_chords = [31.62556581,  4.14286904,  2.6707789,   0.51400213]
    #sweeps = [55.22614128, 56.08207268, 44.76424512, 39.06508792]   
    #dihedrals   = [ 0.00,  0.00,  8.00, 9.25]
    #range_m     = 7000 * 1852

    #ld = predict_cruise_ld(spans, root_chords, tip_chords, sweeps, dihedrals, range_m)

    print(f"RF Surrogate Model predicted L/D: {ld:.3f}")