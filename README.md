# M5 Time-Series Forecasting Project

A retail sales forecasting pipeline built on the **Kaggle M5 Forecasting – Accuracy** competition dataset. The project predicts daily unit sales for Walmart products across multiple stores using LightGBM, outperforming a simple lag-based baseline.

---

## 📂 Dataset

**Source:** [M5 Forecasting – Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

| File | Description |
|------|-------------|
| `sales_train_validation.csv` | Daily unit sales per item/store (wide format) |
| `calendar.csv` | Date metadata – weekday, events, SNAP flags |
| `sell_prices.csv` | Weekly sell price per item/store |

> Only the first 3 stores are used for faster experimentation.

---

## 🔧 Pipeline Overview

### 1. Data Loading & Subset Selection
- Download dataset via Kaggle API
- Load the three core CSVs
- Filter to 3 stores to reduce memory usage

### 2. Data Transformation
- Melt sales from wide → long format (one row per item-date)
- Merge calendar features (weekday, month, year, events, SNAP indicators)
- Merge sell prices on `store_id`, `item_id`, `wm_yr_wk`

### 3. Feature Engineering
| Feature | Description |
|---------|-------------|
| `lag_1`, `lag_7`, `lag_28` | Lagged sales values |
| `roll_mean_7`, `roll_mean_28` | Rolling mean over lag_1 |
| `day`, `weekofyear` | Calendar features |
| `sell_price` | Item price at time of sale |
| Event & SNAP flags | Holiday and food-stamp indicators |

### 4. Train / Validation Split
- Validation set: **last 28 days** of the dataset
- Training set: all data before the validation window
- Rows with NaN lag/rolling features are dropped

### 5. Baseline Model
- **Naive lag-7**: predict today's sales = sales 7 days ago
- Evaluated with RMSE and MAE on the validation set

### 6. LightGBM Model
```
objective      : regression
metric         : rmse
learning_rate  : 0.05
num_leaves     : 64
min_data_in_leaf: 80
feature_fraction: 0.8
bagging_fraction: 0.8
num_boost_round : 1200 (early stopping @ 50)
```
- Categorical features handled natively by LightGBM
- Early stopping based on validation RMSE

---

## 📊 Results

| Model | RMSE | MAE | NRMSE |
|-------|------|-----|-------|
| Baseline (lag_7) | 2.953569 | 1.415870 | 1.759372 |
| LightGBM (full train) | 2.136240 | 1.124034 | 1.272508 |
| LightGBM (last 2 years) | 2.132454 | 1.122434 | 1.270253 |
| LightGBM (+ store lag) | 2.141787 | 1.117007 | 1.275813 |

LightGBM reduces RMSE by ~28% and MAE by ~21% over the naive baseline. Training on the last 2 years yields the best overall RMSE/NRMSE, while adding store-level lag features achieves the lowest MAE.

---

## 📈 Visualizations

- **Forecast vs Actual** – line plot for an example item over the 28-day validation window
- **Feature Importance** – horizontal bar chart for top 20 features
- **Error by Store / Category** – mean absolute error breakdown

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install lightgbm kaggle pandas numpy scikit-learn matplotlib
```

### Kaggle API Setup
Place your `kaggle.json` credentials in `~/.kaggle/` and accept the competition rules on Kaggle before running.

### Run
Open and execute `Time-Series_Forecasting_project.ipynb` cell by cell (Google Colab recommended for free GPU/RAM).

---

## 🗂️ Project Structure
```
├── Time-Series_Forecasting_project.ipynb   # Main notebook
└── README.md                               # This file
```

---

## 🛠️ Tech Stack
- **Python 3**
- **Pandas / NumPy** – data wrangling
- **LightGBM** – gradient boosting model
- **Scikit-learn** – evaluation metrics
- **Matplotlib** – visualizations
- **Kaggle API** – dataset download
