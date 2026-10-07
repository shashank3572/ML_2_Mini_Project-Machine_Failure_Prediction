# Machine Failure Prediction Using Machine Learning

A machine learning mini-project that predicts industrial machine failures using operating sensor data from the **AI4I 2020 Predictive Maintenance Dataset**. Five classification algorithms are trained, evaluated, and compared to identify the best-performing model for early failure detection.

---

## 📌 Overview

Predictive maintenance aims to detect potential machine failures **before** they occur, reducing downtime and maintenance costs. This project builds and benchmarks ML models that classify whether a machine will fail based on real-time operating parameters.

---

## 👥 Team

| Name | Roll Number |
|---|---|
| Shreyas S D | 1AM23CI142 |
| Shashank B | 1AM23CI138 |
| Syed Taha | 1AM23CI163 |
| V Nageshwar | 1AM23CI175 |
| Shadin Muhammed | 1AM23CI136 |

---

## 📂 Dataset

- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** [Kaggle – stephanmatzka/predictive-maintenance-dataset-ai4i-2020](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- **License:** CC-BY-NC-SA-4.0
- **Size:** 10,000 rows × 14 columns

### Features

| Column | Description |
|---|---|
| `UDI` | Unique identifier |
| `Product ID` | Product serial number |
| `Type` | Machine quality variant (L / M / H) |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Spindle rotation speed |
| `Torque [Nm]` | Applied torque |
| `Tool wear [min]` | Cumulative tool wear |
| `Machine failure` | **Target** (0 = No Failure, 1 = Failure) |
| `TWF, HDF, PWF, OSF, RNF` | Specific failure mode flags (dropped before training) |

---

## 🔄 Project Workflow

```
1. Loading & Understanding Dataset
2. Dataset Understanding (shape, dtypes, missing values, duplicates)
3. Data Preprocessing (drop irrelevant cols, encode Type, split & scale)
4. Exploratory Data Analysis (distributions, correlation, boxplots)
5. Model Training (5 algorithms)
6. Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
7. Classification Report + Confusion Matrix + Feature Importance
8. Results & Discussion
9. Conclusion
```

### Preprocessing Steps
- Dropped `UDI` and `Product ID` (no predictive value).
- One-hot encoded `Type` → `Type_L`, `Type_M` (dropped `Type_H` for reference).
- Dropped failure-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to avoid **target leakage**.
- Train/test split: **80% / 20%**, stratified on target.
- Standard-scaled features for Logistic Regression.

---

## 🤖 Models Trained

| # | Model | Notes |
|---|---|---|
| 1 | Logistic Regression | Baseline; requires scaled features |
| 2 | Decision Tree | Unscaled features |
| 3 | Random Forest | 100 estimators |
| 4 | Gradient Boosting | Default config |
| 5 | XGBoost | 100 estimators, `logloss` eval metric |

---

## 📊 Results

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 1 | **XGBoost** | **0.9875** | **0.8909** | **0.7206** | **0.7967** | **0.9729** |
| 🥈 2 | Gradient Boosting | 0.9860 | 0.8846 | 0.6765 | 0.7667 | 0.9698 |
| 🥉 3 | Random Forest | 0.9815 | 0.8780 | 0.5294 | 0.6606 | 0.9616 |
| 4 | Decision Tree | 0.9780 | 0.6818 | 0.6618 | 0.6716 | 0.8254 |
| 5 | Logistic Regression | 0.9675 | 0.6364 | 0.1029 | 0.1772 | 0.8995 |

### 🏆 Best Model: **XGBoost**
- **F1-Score: 79.67%** — best balance of precision and recall.
- **ROC-AUC: 97.29%** — excellent class separability.
- Confusion matrix: correctly detected **49 of 68** actual failures (only 6 false positives).

> ⚠️ **Note:** Logistic Regression reaches 96.75% accuracy but only **10.29% recall** — a reminder that accuracy alone is misleading on imbalanced datasets.

---

## 🔍 Feature Importance (XGBoost)

Top contributors to failure prediction:

1. **Torque [Nm]** — most influential
2. **Air Temperature [K]**
3. **Tool Wear [min]**
4. Process Temperature, Rotational Speed, Machine Type

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — visualization
  - `scikit-learn` — preprocessing, models, metrics
  - `xgboost` — gradient boosted trees
  - `kaggle` — dataset download

---

## 🚀 How to Run

1. **Open the notebook** in Google Colab (`ML2_Mini_Project.ipynb`).
2. **Set up Kaggle credentials** — the notebook uses the Kaggle API:
   - Upload your `kaggle.json` to `/content/` or set `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.
3. **Run all cells sequentially** — the notebook handles dataset download, unzipping, preprocessing, training, and evaluation automatically.

```bash
# For local execution
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kaggle
jupyter notebook ML2_Mini_Project.ipynb
```

---

## 📈 Output Artifacts

The notebook generates:

- 📉 **EDA plots** — feature distributions, correlation heatmap, boxplots
- 📊 **Accuracy & multi-metric comparison bar charts**
- 🧮 **Confusion matrices** for all 5 models (`cm_*.png`)
- 📉 **ROC curves** for all 5 models (single plot)
- ⭐ **Feature importance plot** for XGBoost

---

## ✅ Conclusion

Among the five evaluated algorithms, **XGBoost** proved to be the most reliable model for predicting machine failures on the AI4I 2020 dataset. Its sequential boosting strategy captures complex non-linear interactions between torque, temperature, and tool wear far better than linear or single-tree baselines.

The project demonstrates that ML-driven predictive maintenance can enable **early failure detection**, reduce unplanned downtime, and support data-driven maintenance decisions in industrial settings.

---

## 📜 License

- **Code:** Free to use for academic purposes.
- **Dataset:** [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — attribution required, non-commercial use only.
