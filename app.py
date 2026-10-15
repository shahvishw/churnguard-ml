from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/best_churn_pipeline.joblib")


@st.cache_resource
def load_model():
    """Load the trained pipeline once and reuse it between page updates."""
    return joblib.load(MODEL_PATH)


def get_customer_features(data, model):
    """Keep required columns and prepare uploaded values for the model."""
    data = data.copy()
    data.columns = data.columns.str.strip()

    # These are not model inputs.
    data = data.drop(columns=["customerID", "Churn"], errors="ignore")

    expected_columns = list(model.feature_names_in_)
    missing_columns = [
        column for column in expected_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The CSV is missing these required columns: "
            + ", ".join(missing_columns)
        )

    # Match the model's expected column order and ignore unrelated columns.
    data = data[expected_columns]

    preprocessor = model.named_steps["prepare_data"]
    number_columns = preprocessor.transformers_[0][2]
    text_columns = preprocessor.transformers_[1][2]

    for column in number_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in text_columns:
        data[column] = data[column].astype("string").str.strip()

    return data


def main():
    st.set_page_config(
        page_title="ChurnGuard",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("ChurnGuard")
    st.write(
        "Estimate customer churn risk from a CSV file "
        "or try a prediction for one demo customer."
    )

    if not MODEL_PATH.exists():
        st.error(
            "The trained model was not found. "
            "Run the model training step first."
        )
        st.stop()

    model = load_model()
    expected_columns = list(model.feature_names_in_)

    batch_tab, demo_tab = st.tabs(
        ["Upload customer CSV", "Demo customer"]
    )

    with batch_tab:
        st.subheader("Score customers from a CSV")
        st.caption(
            "The CSV must contain the same customer feature columns used "
            "during training. customerID and Churn are optional and ignored."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                customer_features = get_customer_features(
                    uploaded_data,
                    model,
                )

                churn_predictions = model.predict(customer_features)
                churn_probabilities = model.predict_proba(
                    customer_features
                )[:, 1]

                results = pd.DataFrame()

                if "customerID" in uploaded_data.columns:
                    results["customerID"] = uploaded_data["customerID"]

                results["Churn prediction"] = [
                    "Likely to churn" if prediction == 1 else "Likely to stay"
                    for prediction in churn_predictions
                ]
                results["Churn risk (%)"] = (
                    churn_probabilities * 100
                ).round(1)

                st.metric("Customers scored", len(results))
                st.dataframe(results, use_container_width=True)

                csv_download = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions",
                    data=csv_download,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

            except Exception as error:
                st.error(f"Could not score this file: {error}")

    with demo_tab:
        st.subheader("Try one customer")
        st.caption(
            "The form starts with typical values. Change any fields "
            "before submitting the prediction."
        )

        preprocessor = model.named_steps["prepare_data"]
        number_columns = list(preprocessor.transformers_[0][2])
        text_columns = list(preprocessor.transformers_[1][2])

        number_steps = preprocessor.named_transformers_["numbers"]
        text_steps = preprocessor.named_transformers_["text"]

        number_defaults = number_steps.named_steps[
            "fill_missing"
        ].statistics_
        text_defaults = text_steps.named_steps[
            "fill_missing"
        ].statistics_
        category_options = text_steps.named_steps["encode"].categories_

        customer = {}

        with st.form("demo_customer_form"):
            st.write("Customer details")

            for index, column in enumerate(text_columns):
                choices = [
                    str(value) for value in category_options[index]
                ]
                default = str(text_defaults[index])

                customer[column] = st.selectbox(
                    column,
                    options=choices,
                    index=choices.index(default),
                )

            for index, column in enumerate(number_columns):
                default = float(number_defaults[index])

                if column == "SeniorCitizen":
                    customer[column] = st.selectbox(
                        column,
                        options=[0, 1],
                        index=int(default),
                    )
                else:
                    customer[column] = st.number_input(
                        column,
                        min_value=0.0,
                        value=default,
                    )

            submitted = st.form_submit_button("Predict churn risk")

        if submitted:
            demo_features = pd.DataFrame(
                [customer],
                columns=expected_columns,
            )
            prediction = model.predict(demo_features)[0]
            churn_probability = model.predict_proba(demo_features)[0, 1]

            st.metric(
                "Estimated churn risk",
                f"{churn_probability * 100:.1f}%",
            )

            if prediction == 1:
                st.warning("The model predicts this customer is likely to churn.")
            else:
                st.success("The model predicts this customer is likely to stay.")

            st.caption(
                "This is a model estimate, not a guarantee of what a "
                "customer will do."
            )


if __name__ == "__main__":
    main()