"""Train the AtmosDays place-learning forecast: gradient-boosted trees (LightGBM)
that learn how the GFS-family forecast misses (measured minus forecast) for
temperature, dew point and wind, from the time of year, time of day, how far
ahead, where, the model's terrain height and the forecast itself.

Honest testing: whole years are held back (never seen in training) and the
corrected forecast is compared with the raw one on them. The model that ships
is then retrained on every year with the same settings.

Output (one JSON file the website evaluates itself, no outside service):
  {"version", "trained", "features", "targets": {name: {"trees": [...], "base"}},
   "report": {...}, "typicalMiss": {...}}
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os

import lightgbm as lgb
import numpy as np

from ml.data import FEATURES, TARGETS

TEST_YEARS = {2004, 2012, 2019, 2023}  # never trained on for the report
VALID_YEARS = {2016, 2022}  # picks how many trees, then also kept out of the report
PARAMS = {"objective": "l1", "learning_rate": 0.06, "num_leaves": 63, "min_data_in_leaf": 400, "feature_fraction": 0.9,
          "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 0}
MAX_ROUNDS = 600
F = {name: i for i, name in enumerate(FEATURES)}
LEAD_BUCKETS = [(1, 48, "1-2 days"), (49, 96, "3-4 days"), (97, 168, "5-7 days")]
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}


def load(data_dir):
    Xs, Ys, Ms = [], [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(path)
        if list(z["features"]) != FEATURES or not len(z["X"]):
            continue
        Xs.append(z["X"]); Ys.append(z["Y"]); Ms.append(z["M"])
    if not Xs:
        raise SystemExit("no training chunks yet")
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms)


def years_fast(hours):
    days = (hours // 24).astype("datetime64[D]")
    return days.astype("datetime64[Y]").astype(np.int64) + 1970


def months(M):
    days = (M[:, 1] // 24).astype("datetime64[D]")
    return (days.astype("datetime64[M]").astype(np.int64) % 12) + 1


def export_tree(node, out):
    """Flatten one LightGBM tree into arrays: feature (-1 = leaf), threshold, left, right, NaN goes left, value."""
    i = len(out["f"])
    for k in out:
        out[k].append(0)
    if "leaf_value" in node:
        out["f"][i], out["v"][i] = -1, round(float(node["leaf_value"]), 5)
        return i
    out["f"][i] = int(node["split_feature"])
    out["t"][i] = float(node["threshold"])
    # Where a missing value goes. LightGBM only sends NaN the "default" way when the
    # feature had NaN while it was binned (missing_type "NaN"); otherwise (missing_type
    # "None") a NaN is read as 0 and compared like any number.
    if node.get("missing_type", "None") == "None":
        out["n"][i] = 1 if 0.0 <= out["t"][i] else 0
    else:
        out["n"][i] = 1 if node.get("default_left", True) else 0
    out["l"][i] = export_tree(node["left_child"], out)
    out["r"][i] = export_tree(node["right_child"], out)
    return i


def export(booster):
    dump = booster.dump_model()
    trees = []
    for t in dump["tree_info"]:
        out = {"f": [], "t": [], "n": [], "l": [], "r": [], "v": []}
        export_tree(t["tree_structure"], out)
        trees.append(out)
    return trees


def predict_exported(trees, X):
    """Same walk the website does (used here to check the export matches LightGBM).

    Compare in float64, as LightGBM and the website (JavaScript numbers) do. The
    training rows are float32; comparing a float32 value with a threshold makes
    numpy round the threshold to float32 too, and a station's exact latitude,
    longitude or height can then land on the wrong side of a split.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.zeros(len(X))
    for tr in trees:
        for r in range(len(X)):
            i = 0
            while tr["f"][i] >= 0:
                x = X[r, tr["f"][i]]
                go_left = tr["n"][i] if np.isnan(x) else x <= tr["t"][i]
                i = tr["l"][i] if go_left else tr["r"][i]
            y[r] += tr["v"][i]
    return y


def mae(a):
    return round(float(np.mean(np.abs(a))), 2) if len(a) else None


def train_target(X, Y, years, mon, j, log):
    name = TARGETS[j]
    fc = X[:, F[name]]
    ok = np.isfinite(Y[:, j]) & np.isfinite(fc)
    resid = Y[:, j] - fc
    test = ok & np.isin(years, list(TEST_YEARS))
    valid = ok & np.isin(years, list(VALID_YEARS))
    train = ok & ~test & ~valid
    log(f"{name}: {train.sum():,} training rows, {valid.sum():,} for tuning, {test.sum():,} held back")
    dtr = lgb.Dataset(X[train], resid[train], feature_name=FEATURES, free_raw_data=True)
    dva = lgb.Dataset(X[valid], resid[valid], reference=dtr)
    b = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
    rounds = max(50, b.best_iteration or MAX_ROUNDS)
    raw_err = -resid[test]  # forecast minus measured
    ml_err = b.predict(X[test], num_iteration=rounds) - resid[test]
    lead = X[test, F["lead"]]
    rep = {"rowsHeldBack": int(test.sum()), "rawMiss": mae(raw_err), "aiMiss": mae(ml_err)}
    rep["better"] = round(100 * (1 - rep["aiMiss"] / rep["rawMiss"]), 1) if rep["rawMiss"] else None
    rep["byLead"] = {lab: {"raw": mae(raw_err[(lead >= a) & (lead <= z)]), "ai": mae(ml_err[(lead >= a) & (lead <= z)])} for a, z, lab in LEAD_BUCKETS}
    seas = np.array([SEASONS[m] for m in mon[test]]) if test.sum() else np.array([])
    rep["bySeason"] = {s: {"raw": mae(raw_err[seas == s]), "ai": mae(ml_err[seas == s])} for s in ("winter", "spring", "summer", "fall")}
    rep["byTestYear"] = {str(y): {"raw": mae(raw_err[years[test] == y]), "ai": mae(ml_err[years[test] == y])} for y in sorted(TEST_YEARS) if (years[test] == y).any()}
    if name == "t_f":
        rep["within3"] = {"raw": round(float(np.mean(np.abs(raw_err) <= 3)) * 100, 1), "ai": round(float(np.mean(np.abs(ml_err) <= 3)) * 100, 1)}
    typical = {lab: rep["byLead"][lab]["ai"] for _, _, lab in LEAD_BUCKETS}
    log(f"{name}: held-back years, raw miss {rep['rawMiss']} -> AI {rep['aiMiss']} ({rep['better']}% better), {rounds} trees")
    # The model that ships learns from every year, with the tuned size.
    allrows = ok
    final = lgb.train(PARAMS, lgb.Dataset(X[allrows], resid[allrows], feature_name=FEATURES), rounds)
    trees = export(final)
    chk = np.flatnonzero(allrows)[:200]
    diff = np.max(np.abs(predict_exported(trees, X[chk]) - final.predict(X[chk])))
    if diff > 1e-3:
        raise SystemExit(f"{name}: exported trees disagree with LightGBM by {diff}")
    return trees, rep, typical, int(allrows.sum())


def main(data_dir, out_path, log=print):
    X, Y, M = load(data_dir)
    years = years_fast(M[:, 1] - X[:, F["lead"]].astype(np.int64))
    mon = months(M)
    log(f"{len(X):,} rows from {years.min()} to {years.max()}")
    model = {"version": 1, "kind": "LightGBM gradient-boosted trees, predicts measured minus forecast",
             "trained": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "features": FEATURES, "targets": {}, "report": {"testYears": sorted(TEST_YEARS), "years": [int(years.min()), int(years.max())]},
             "typicalMiss": {}}
    total = 0
    for j, name in enumerate(TARGETS):
        trees, rep, typical, n = train_target(X, Y, years, mon, j, log)
        model["targets"][name] = {"trees": trees}
        model["report"][name] = rep
        model["typicalMiss"][name] = typical
        total = max(total, n)
    model["report"]["examples"] = total
    with open(out_path, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    with open(os.path.splitext(out_path)[0] + "-report.json", "w") as f:
        json.dump({k: model[k] for k in ("version", "kind", "trained", "report", "typicalMiss")}, f, indent=1)
    log(f"saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return model
