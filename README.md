# House Price Prediction Using Machine Learning

## Problem Statement
Predict house prices based on different features such as area, quality, rooms, and other property attributes.

---

## Dataset
- Source: Kaggle House Prices Dataset
- File: `dataset/train.csv`
- Target Variable: `SalePrice`

---

## Workflow

1. Data Preprocessing
   - Missing value handling
   - Categorical encoding

2. Model Training
   - Random Forest Regressor

3. Evaluation
   - Mean Squared Error (MSE)
   - R² Score

4. Model Saving
   - Saved using pickle (`model.pkl`)

---

## Model Performance
- R² Score: ~0.89
- MSE: ~8e8 (depends on run)

---

## How to Run

```bash
pip install -r requirements.txt
python train.py