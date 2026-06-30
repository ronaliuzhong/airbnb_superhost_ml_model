# Airbnb Superhost Classification—Logistic Regression Model Selection

This project trains a logistic regression model to predict whether an Airbnb host is a "superhost," using the ML life cycle's evaluation and deployment steps: hyperparameter tuning, model evaluation, feature selection, and persistence.

## Problem Definition

- **Label**: `host_is_superhost` (`True` / `False`) — a binary classification problem.
- **Features**: All remaining columns in the dataset (already one-hot encoded, scaled, and imputed).
- **Dataset**: `airbnbData_train.csv`, pre-processed and ready for modeling.

## Workflow

1. **Data Preparation**
   Loaded the dataset, separated features (`X`) from the label (`y`), and split into training/test sets (90/10 split).

2. **Baseline Model**
   Trained a `LogisticRegression` model using scikit-learn's default hyperparameter `C=1.0`. Evaluated using accuracy and a confusion matrix.

3. **Hyperparameter Tuning**
   Ran `GridSearchCV` (5-fold cross-validation, scored on ROC-AUC) over 10 candidate values of `C`:
   `[10^-5, 10^-4, ..., 10^4]`
   The best-performing value found was **C = 1000** (weaker regularization, more complex model).

4. **Optimal Model**
   Retrained a logistic regression model using `C=1000` and evaluated it the same way as the baseline (confusion matrix, accuracy).

5. **Precision-Recall Curves**
   Plotted precision-recall curves for both the default and tuned models to visualize the trade-off across classification thresholds.

6. **ROC Curves & AUC**
   Plotted ROC curves for both models and computed AUC scores to compare overall performance independent of any single threshold.

7. **Feature Selection (Deep Dive)**
   Used `SelectKBest` with the `f_classif` scoring function to select the top 5 most predictive features, retrained the tuned model on this reduced feature set, and recomputed AUC.
   - **Finding**: Increasing the number of selected features (e.g., to 10) did not meaningfully improve AUC — suggesting the top 5 features already capture most of the predictive signal, and additional features mainly add computation cost without accuracy gains.

8. **Model Persistence**
   Saved the trained model using `pickle` to `Superhost_Classification_Model.pkl`, then verified it could be reloaded and used for predictions.

## Files

| File | Description |
|---|---|
| `ModelSelectionForLogisticRegression.ipynb` | Main notebook with the full workflow |
| `data_LR/airbnbData_train.csv` | Preprocessed training dataset |
| `Superhost_Classification_Model.pkl` | Pickled, trained logistic regression model (`C=1000`) |

## Key Results

- **Best hyperparameter**: `C = 1000` (found via grid search, optimized for ROC-AUC)
- **Model comparison**: Default (`C=1`) vs. tuned (`C=1000`) models were compared via confusion matrices, precision-recall curves, and ROC-AUC.
- **Feature selection**: Top 5 features (via `SelectKBest`/`f_classif`) achieved AUC comparable to using more features, indicating diminishing returns beyond the top 5.

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

Load the saved model and generate predictions:

```python
import pickle

with open('Superhost_Classification_Model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

> **Note**: The pickled model was trained on the top 5 selected features (from the feature selection step), so `X_new` should contain only those same 5 columns, in the same order.
