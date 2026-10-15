from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_data():
    """Read the CSV and separate customer features from the churn target."""
    data = pd.read_csv(DATA_PATH)

    # Clean up column names and text values.
    data.columns = data.columns.str.strip()

    # Convert "No" and "Yes" to 0 and 1.
    churn_text = data.pop("Churn").astype("string").str.strip()
    target = churn_text.map({"No": 0, "Yes": 1})

    if target.isna().any():
        raise ValueError("The Churn column must contain only 'Yes' or 'No'.")

    # TotalCharges contains 11 blank values; convert them to missing numbers.
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].astype("string").str.strip(),
        errors="coerce",
    )

    # An ID identifies a customer but does not describe their churn risk.
    data = data.drop(columns=["customerID"], errors="ignore")

    # Remove extra spaces from categorical values.
    for column in data.select_dtypes(include=["object", "string"]).columns:
        data[column] = data[column].astype("string").str.strip()

    return data, target.astype("int64")


def split_data(features, target):
    """Create repeatable training and test groups with similar churn rates."""
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )


def make_preprocessor(features):
    """Prepare number and text columns before a model learns from them."""
    number_columns = features.select_dtypes(include=["number"]).columns
    text_columns = features.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns

    number_steps = Pipeline(
        steps=[
            # Fill missing numbers with the median.
            ("fill_missing", SimpleImputer(strategy="median")),
            # Put numeric columns on comparable scales.
            ("scale", StandardScaler()),
        ]
    )

    text_steps = Pipeline(
        steps=[
            # Fill a missing category with the most common value.
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            # Turn categories into numbers a model can use.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numbers", number_steps, number_columns),
            ("text", text_steps, text_columns),
        ]
    )