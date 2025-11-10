# decision_tree_weather.py
# Decision Tree workflow for ClimateWins Exercise 1.5
#
# How to run:
#   python decision_tree_weather.py
#
# Outputs:
#   - prints training/testing accuracy
#   - saves confusion matrices images under ./outputs/dt_confusions/
#
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from common_utils import (
    load_weather_and_answers, build_xy, train_test_split_fixed,
    evaluate_and_print, plot_multi_confusion_matrices, DROP_STATIONS_DEFAULT
)

# ---------- PATHS (edit if needed) ----------
WEATHER_CSV = r"/Users/davidscheider/anaconda_projects/ClimateWins/02 Data/Original Data/Dataset-weather-prediction-dataset-processed.csv"
ANSWERS_CSV = r"/Users/davidscheider/anaconda_projects/ClimateWins/02 Data/Original Data/Dataset-Answers-Weather_Prediction_Pleasant_Weather.csv"
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LOAD & PREP ----------
merged, key_cols = load_weather_and_answers(
    WEATHER_CSV, ANSWERS_CSV, drop_stations=DROP_STATIONS_DEFAULT, verbose=True
)
X, y = build_xy(merged, key_cols, verbose=True)

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split_fixed(X, y, test_size=0.3, random_state=42)

# ---------- MODEL ----------
# Note: Large trees can overfit and be slow; start with constraints, then relax if needed.
dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,            # try values like 10, 20 for pruning-by-constraint
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
dt.fit(X_train, y_train)

# ---------- EVAL ----------
y_train_pred = dt.predict(X_train)
y_test_pred  = dt.predict(X_test)
acc_tr, acc_te = evaluate_and_print("DecisionTree", y_train, y_train_pred, y_test, y_test_pred)

# ---------- PRUNING DISCUSSION ----------
# If training accuracy >> testing accuracy (and testing noticeably lower), the tree likely overfits.
# You can experiment with max_depth, min_samples_leaf, and ccp_alpha (cost-complexity pruning) like:
#   path = dt.cost_complexity_pruning_path(X_train, y_train)
#   dt_p = DecisionTreeClassifier(ccp_alpha=chosen_alpha, random_state=42).fit(X_train, y_train)

# ---------- VISUALIZE TREE (optional; can be very large) ----------
# WARNING: For big data this plot can be huge and slow. Uncomment if needed.
# fig, ax = plt.subplots(figsize=(16, 12))
# plot_tree(dt, filled=True, max_depth=3)   # cap displayed depth for readability
# fig.tight_layout()
# fig.savefig(OUTPUT_DIR / "dt_tree_preview.png", dpi=150)

# ---------- CONFUSION MATRICES ----------
# Provide multiple scenarios (you can add pruned versions, or different random_state splits).
y_true_list = [y_test]
y_pred_list = [y_test_pred]
titles      = [f"DT Test (acc={acc_te:.3f})"]
saved = plot_multi_confusion_matrices(
    y_true_list, y_pred_list, titles, cols=3, figpath=str(OUTPUT_DIR / "dt_confusions")
)
print("Confusion matrix images:", saved)
print("Done.")