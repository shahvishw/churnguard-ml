import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .data import load_data, make_preprocessor, split_data


def main():
    # Read the data and make one held-out test split.
    features, target = load_data()
    X_train, X_test, y_train, y_test = split_data(features, target)

    # Compare three different approaches to classification.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    # Track accuracy and churn-focused scores for each model.
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    # Each fold keeps roughly the same proportion of churned customers.
    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    comparison_rows = []

    for model_name, model in models.items():
        # Keep data preparation and the model together.
        pipeline = Pipeline(
            steps=[
                ("prepare_data", make_preprocessor(X_train)),
                ("model", model),
            ]
        )

        # Compare each model across five training/validation rounds.
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=folds,
            scoring=scoring,
        )

        row = {"model": model_name}

        for metric_name in scoring:
            row[f"{metric_name}_mean"] = scores[
                f"test_{metric_name}"
            ].mean()

        comparison_rows.append(row)

    # Use average cross-validation ROC-AUC to select the model.
    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values("roc_auc_mean", ascending=False)
    best_model_name = comparison.iloc[0]["model"]

    # Train the selected model on all training data.
    best_pipeline = Pipeline(
        steps=[
            ("prepare_data", make_preprocessor(X_train)),
            ("model", models[best_model_name]),
        ]
    )
    best_pipeline.fit(X_train, y_train)

    # Evaluate once on the held-out test data.
    predictions = best_pipeline.predict(X_test)
    churn_probabilities = best_pipeline.predict_proba(X_test)[:, 1]

    holdout_metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, churn_probabilities),
    }

    # Save the trained pipeline and results for the Streamlit app and README.
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(best_pipeline, "models/best_churn_pipeline.joblib")
    comparison.to_csv("reports/model_comparison.csv", index=False)

    summary = {
        "selected_model": best_model_name,
        "selection_metric": "average cross-validation ROC-AUC",
        "holdout_metrics": {
            name: float(value) for name, value in holdout_metrics.items()
        },
    }

    Path("reports/model_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Model comparison:")
    print(comparison.round(3).to_string(index=False))
    print(f"\nSelected model: {best_model_name}")
    print("\nHeld-out test scores:")
    for name, value in holdout_metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()