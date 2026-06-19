# F1-Pitstop_prediction

Add F1 pit stop prediction model with LightGBM (ROC-AUC 0.93+)

Implemented binary classification pipeline to predict whether a car
will pit on the next lap, using lap telemetry data (351k train /
88k test rows, 15 features).

Models evaluated:
- Logistic Regression (elasticnet, l1_ratio=0.6, saga solver)
  with StandardScaler preprocessing — baseline ROC-AUC ~0.85
- Random Forest (n_estimators=300, max_depth=15,
  class_weight=balanced) — ROC-AUC ~0.90
- LightGBM (n_estimators=1000, num_leaves=63, learning_rate=0.05,
  scale_pos_weight tuned to class ratio ~4.03) — ROC-AUC 0.93+

Key decisions:
- Used scale_pos_weight/class_weight to handle class imbalance
  (pit stops are ~20% of laps)
- Switched from XGBoost to LightGBM due to 32-bit/64-bit
  dlopen incompatibility on local environment
- Tuned early stopping rounds (50 -> 200) after initial run
  stopped prematurely at iteration 25
- Cleaned whitespace in feature names to remove LightGBM warnings

Output: submission.csv with predicted pit probabilities (id, PitNextLap)
