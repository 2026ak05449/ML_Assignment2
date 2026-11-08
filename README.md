# 🛒 Shopper Intent — ML Assignment 2 (BITS Pilani M.Tech AIML)

**Live app:** _<paste your Streamlit app link here>_

## a. Problem statement
An online store logs every browsing session: how many pages of each kind were visited, how long was spent on them,
bounce/exit rates, page value, month, visitor type and so on. The task is to **predict whether a session ends in a
purchase** (`Revenue = True/False`). This is a **binary classification** problem with imbalanced classes
(~15.5 % of sessions are purchases).

Five classifiers are trained and compared on the same data, and an interactive Streamlit app lets anyone upload
test data, choose a model and see its evaluation metrics, confusion matrix and classification report.

## b. Dataset description
**Source:** UCI Machine Learning Repository — [Online Shoppers Purchasing Intention Dataset (ID 468)](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(Sakar, C.O., Polat, S.O., Katircioglu, M. et al., 2018).

| Property | Value |
|---|---|
| Instances | 12,330 sessions (12,205 after removing 125 duplicate rows) |
| Input features | 17 (10 numeric, 1 boolean, 6 categorical) |
| Target | `Revenue` — True (purchase) / False (no purchase) |
| Class balance | 84.5 % False · 15.5 % True |
| Missing values | None |

| Feature | Type | Description |
|---|---|---|
| Administrative, Informational, ProductRelated | int | Number of pages of each type visited |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | float | Seconds spent on each page type |
| BounceRates | float | Share of visitors who entered on a page and left without another request |
| ExitRates | float | Share of page views where the page was the last of the session |
| PageValues | float | Average value of the pages visited before a transaction |
| SpecialDay | float | Closeness of the visit date to a special day (0–1) |
| Month | categorical | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | categorical (coded) | Technical and traffic-source info |
| VisitorType | categorical | Returning_Visitor / New_Visitor / Other |
| Weekend | boolean | Whether the visit was on a weekend |

**Preprocessing:** duplicates dropped; numeric features standardised (`StandardScaler`); categorical features one-hot
encoded (`OneHotEncoder`, unknown categories ignored). Stratified 80 / 20 train–test split (`random_state = 2026`).
The preprocessing is saved inside each model pipeline, so the app takes raw CSV columns as input.
Hyper-parameters were tuned with 5-fold stratified `GridSearchCV` optimising F1.

## c. GitHub repository
**Repository:** _<paste your GitHub repo link here>_

```
├── app.py                               # Streamlit app
├── ML_Assignment2_Model_Training.ipynb  # EDA, training, evaluation, model saving
├── requirements.txt
├── README.md
├── test_data.csv                        # held-out test split (2,441 rows, features + Revenue)
├── data/online_shoppers_intention.csv   # original UCI dataset
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── model_info.json                  # feature lists and model file names
│   └── test_metrics.csv                 # metrics table below
└── .streamlit/config.toml               # app theme
```

## d. Models used
Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine (RBF kernel).

### Comparison table (test set, 2,441 sessions)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8669 | 0.9084 | 0.5556 | 0.7461 | 0.6369 | 0.5666 |
| Decision Tree | 0.8689 | **0.9310** | 0.5522 | **0.8586** | **0.6721** | **0.6176** |
| KNN | 0.8763 | 0.7843 | **0.6923** | 0.3770 | 0.4881 | 0.4501 |
| Naive Bayes | 0.8177 | 0.8241 | 0.4418 | 0.6257 | 0.5179 | 0.4190 |
| Support Vector Machine | **0.8816** | 0.9188 | 0.5955 | 0.7592 | 0.6674 | 0.6032 |

Precision, Recall and F1 are for the positive class (purchase). AUC for the SVM uses its decision-function scores.

Best hyper-parameters: LR `C=0.01, class_weight=balanced` · DT `max_depth=4, min_samples_leaf=20, class_weight=balanced` ·
KNN `n_neighbors=5, weights=uniform` · NB `var_smoothing=0.1` · SVM `C=1, class_weight=balanced`.

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Solid linear baseline (AUC 0.908, F1 0.637). Tuning chose strong regularisation (C = 0.01) with balanced class weights, which lifted purchase recall to 0.75 at the cost of precision (0.56). Its linear boundary cannot capture the threshold-like effect of `PageValues`, so it trails the tree and SVM. |
| **Decision Tree** | Best overall: highest AUC (0.931), F1 (0.672), MCC (0.618) and recall (0.859). A shallow tree (depth 4, min 20 samples per leaf) is enough because `PageValues > 0` is an almost rule-like split for purchases; the depth limit keeps it from overfitting. Precision (0.55) is moderate because balanced weights make it flag more sessions as buyers. |
| **KNN** | Highest precision (0.69) and good accuracy (0.876) but the weakest recall (0.38), F1 (0.488) and AUC (0.784). With 63 one-hot columns, distances are dominated by sparse categorical dimensions, and majority voting favours the 84 % "no purchase" class — so it misses most buyers. |
| **Naive Bayes** | Lowest accuracy (0.818) and MCC (0.419). The Gaussian independence assumption is violated (BounceRates and ExitRates are ~0.91 correlated; durations are highly skewed, not normal) and one-hot features are not Gaussian. A large `var_smoothing` (0.1) was needed to make it stable. Fast and simple, but the least reliable here. |
| **Support Vector Machine** | Close second: best accuracy (0.882), AUC 0.919, F1 0.667, MCC 0.603. The RBF kernel with balanced class weights captures the non-linear interactions better than Logistic Regression, with a better precision/recall balance (0.60 / 0.76) than the tree. Slowest model to train and the largest saved file. |
| **Overall winner** | **Decision Tree** — best F1, AUC, MCC and recall, while also being small, fast and easy to explain. Accuracy alone is misleading on this imbalanced data (predicting "no purchase" always would score 84 %), so F1 and MCC were used to pick the winner. SVM is a strong alternative when precision matters more. |

## Running locally
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
streamlit run app.py
```
To retrain, open `ML_Assignment2_Model_Training.ipynb` and run all cells (also needs `ipykernel`).

## Streamlit app features
- **CSV upload** of test data (or use the bundled `test_data.csv`, which can also be downloaded from the sidebar)
- **Model selection dropdown** for the five saved models
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix, ROC curve and classification report** for the selected model
- **Compare all models** tab with a metrics table and chart
- **Predictions** tab with a downloadable CSV of predictions

Models are only loaded and used for prediction in the app — nothing is retrained on the uploaded data.
