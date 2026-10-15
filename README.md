# ChurnGuard

ChurnGuard is a customer churn prediction project built with Python, scikit-learn, and Streamlit. It lets users score customers from a CSV file or try an individual customer in a demo form.

## Problem

Customer churn can reduce recurring revenue. This project uses customer account and service information to estimate whether a telecom customer is likely to leave. The predictions can help a retention team decide which customers to review.

## Features

- Score multiple customers by uploading a CSV file
- Try an individual customer prediction with the demo form
- View a churn probability and predicted outcome
- Download scored customer results
- Compare several classification models using cross-validation

## Dataset

The model was developed using the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which contains 7,043 customer records and a `Churn` target.

The raw dataset is not included in this repository. To train the model, download the CSV from Kaggle and place it at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The file in `data/sample/` contains synthetic example customers for trying the app. They are not training examples.

## Modeling approach

The project prepares numeric and categorical fields in a scikit-learn pipeline. Missing numeric values are filled with the median, categorical values with the most frequent value, and categorical fields are one-hot encoded. `customerID` is excluded from training because it identifies a customer rather than describing churn risk.

Three models were compared:

- Logistic Regression
- Random Forest
- HistGradientBoosting

Model comparison used five-fold stratified cross-validation. The selected model was refit on the training split and evaluated on a separate held-out test split. Logistic Regression was selected because it had the highest mean cross-validation ROC-AUC.

## Model results

Cross-validation averages:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.748 | 0.516 | 0.801 | 0.628 | 0.846 |
| Random Forest | 0.771 | 0.552 | 0.722 | 0.626 | 0.837 |
| HistGradientBoosting | 0.793 | 0.636 | 0.509 | 0.565 | 0.835 |

Selected model, held-out test results:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.738 | 0.504 | 0.783 | 0.614 | 0.841 |

ROC-AUC was used to select the model because it compares churn-risk ranking across thresholds. Accuracy is reported as well, but it does not show how well the model identifies the smaller churn group by itself. At the default classification threshold, the selected model’s held-out precision was 0.504 and recall was 0.783.

## Run locally

Create and activate a virtual environment, then install the project dependencies.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the Kaggle CSV to `data/raw/`, then train and save the model:

```powershell
python -m src.churnguard.modeling
```

Start the Streamlit app:

```powershell
python -m streamlit run app.py
```

To try batch scoring, upload `data/sample/sample_customers.csv` in the app.

## Project structure

```text
app.py                         Streamlit application
src/churnguard/data.py         Data loading and preprocessing
src/churnguard/modeling.py     Model comparison and training
data/raw/                      Local Kaggle data; not committed
data/sample/                   Synthetic sample upload
models/                        Saved model pipeline
reports/                       Model comparison and summary
requirements.txt               Project dependencies
```

## Limitations

This project uses a public sample dataset and does not represent a deployed telecom retention system. A churn probability is an estimate, not a guarantee. The chosen classification threshold and model evaluation do not account for the different business costs of contacting a customer versus missing a customer who may churn.