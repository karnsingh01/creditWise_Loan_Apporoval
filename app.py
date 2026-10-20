
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CreditWise Loan Approval",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 CreditWise Loan Approval")
st.write(
    "Enter the applicant's details to predict loan approval "
    "using a Gaussian Naive Bayes model."
)

st.info("Educational project only. This prediction is not a real lending decision.")

DATA_PATH = Path(__file__).resolve().parent / "loan_approval_data.csv"
TARGET = "Loan_Approved"
ID_COLUMNS = {"Applicant_ID", "ApplicantID", "ID", "id"}


# ---------------- LOAD DATA ----------------
if not DATA_PATH.exists():
    st.error("Dataset not found. Put loan_approval_data.csv in the same folder as app.py.")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Unable to read dataset: {e}")
    st.stop()

if TARGET not in df.columns:
    st.error(f"Required target column '{TARGET}' was not found in the CSV.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

df = df.dropna(subset=[TARGET]).copy()

if df.empty:
    st.error("The dataset contains no rows with a loan approval label.")
    st.stop()


# ---------------- CLEAN DATA ----------------
def clean_column(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    converted = pd.to_numeric(series, errors="coerce")
    valid_count = series.notna().sum()

    if valid_count > 0 and converted.notna().sum() / valid_count >= 0.9:
        return converted

    return series.map(
        lambda value: np.nan if pd.isna(value) else str(value).strip()
    )


for column in df.columns:
    if column == TARGET:
        df[column] = df[column].astype(str).str.strip()
    else:
        df[column] = clean_column(df[column])


# ---------------- FEATURE ENGINEERING ----------------
# Match the feature engineering used in the notebook.
if "DTI_Ratio" in df.columns:
    df["DTI_Ratio_sq"] = pd.to_numeric(
        df["DTI_Ratio"], errors="coerce"
    ) ** 2

if "Credit_Score" in df.columns:
    df["Credit_Score_sq"] = pd.to_numeric(
        df["Credit_Score"], errors="coerce"
    ) ** 2

# Keep the original input columns for the form.
input_columns = [
    column for column in df.columns
    if column != TARGET and column not in ID_COLUMNS
]

# The notebook uses the squared features instead of the original columns.
drop_from_model = {
    TARGET, "Credit_Score", "DTI_Ratio",
    "Applicant_ID", "ApplicantID", "ID", "id"
}

X = df.drop(
    columns=[column for column in drop_from_model if column in df.columns],
    errors="ignore"
).copy()

if X.empty:
    st.error("No usable model features remain in the dataset.")
    st.stop()

# Convert target labels into numeric classes.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[TARGET].astype(str))

if len(label_encoder.classes_) != 2:
    st.error(
        "The target column must have exactly two classes. "
        f"Found: {list(label_encoder.classes_)}"
    )
    st.stop()

# Identify numeric and categorical features.
cat_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

num_columns = [
    column for column in X.columns
    if column not in cat_columns
]


# ---------------- TRAIN MODEL ----------------
try:
    transformers = []

    if num_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numeric_pipeline, num_columns)
        )

    if cat_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, cat_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])

    # GaussianNB requires dense features after one-hot encoding.
    # ColumnTransformer returns dense output by default when configured below.
    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])

    counts = pd.Series(y).value_counts()
    stratify_y = y if len(counts) == 2 and counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_y
    )

    model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

except Exception as e:
    st.error(f"Model training failed: {type(e).__name__}: {e}")
    st.stop()


# ---------------- APPLICANT FORM ----------------
st.subheader("📝 Applicant Information")
st.caption(
    f"Complete the {len(input_columns)} applicant fields below. "
    "ID fields are excluded."
)

# The form uses original data columns, not the engineered model columns.
# This ensures users can enter Credit_Score and DTI_Ratio normally.
form_columns = [
    column for column in input_columns
    if column not in ID_COLUMNS
]

values = {}

with st.form("loan_application_form", clear_on_submit=False):
    left, right = st.columns(2)

    for index, column in enumerate(form_columns):
        container = left if index % 2 == 0 else right
        label = column.replace("_", " ")

        with container:
            if column in cat_columns or not pd.api.types.is_numeric_dtype(df[column]):
                options = sorted(
                    df[column].dropna().astype(str).unique().tolist(),
                    key=str.lower
                )

                if options:
                    values[column] = st.selectbox(
                        label,
                        options,
                        key=f"input_{column}"
                    )
                else:
                    values[column] = st.text_input(
                        label,
                        key=f"input_{column}"
                    )

            else:
                series = pd.to_numeric(
                    df[column], errors="coerce"
                ).dropna()

                default = float(series.median()) if not series.empty else 0.0

                if column == "Credit_Score":
                    values[column] = st.number_input(
                        label,
                        min_value=0,
                        max_value=1000,
                        value=int(np.clip(round(default or 680), 0, 1000)),
                        step=1,
                        key=f"input_{column}"
                    )

                elif column == "Age":
                    values[column] = st.number_input(
                        label,
                        min_value=0,
                        max_value=120,
                        value=int(np.clip(round(default or 25), 0, 120)),
                        step=1,
                        key=f"input_{column}"
                    )

                else:
                    minimum = float(series.min()) if not series.empty else 0.0
                    maximum = float(series.max()) if not series.empty else 1000000.0

                    if minimum == maximum:
                        maximum = minimum + 1.0

                    values[column] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=float(np.clip(default, minimum, maximum)),
                        key=f"input_{column}"
                    )

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        type="primary",
        use_container_width=True
    )


# ---------------- MAKE PREDICTION ----------------
if submitted:
    try:
        # Start with the user's raw form inputs.
        applicant = pd.DataFrame(
            [{column: values[column] for column in form_columns}]
        )

        # Create engineered features exactly as in training.
        if "DTI_Ratio" in applicant.columns:
            applicant["DTI_Ratio_sq"] = (
                pd.to_numeric(applicant["DTI_Ratio"], errors="coerce") ** 2
            )

        if "Credit_Score" in applicant.columns:
            applicant["Credit_Score_sq"] = (
                pd.to_numeric(applicant["Credit_Score"], errors="coerce") ** 2
            )

        # Remove the same columns that were excluded during model training.
        applicant = applicant.drop(
            columns=[
                column for column in drop_from_model
                if column in applicant.columns
            ],
            errors="ignore"
        )

        # Match the exact model feature order.
        applicant = applicant.reindex(columns=X.columns)

        for column in cat_columns:
            if column in applicant.columns:
                applicant[column] = applicant[column].astype(object)

        prediction = int(model.predict(applicant)[0])
        probabilities = model.predict_proba(applicant)[0]

        predicted_label = str(
            label_encoder.inverse_transform([prediction])[0]
        ).strip().lower()

        approved_words = {
            "yes", "approved", "approve", "true", "1", "1.0"
        }
        rejected_words = {
            "no", "not approved", "rejected", "reject",
            "false", "0", "0.0"
        }

        if predicted_label in approved_words:
            is_approved = True
        elif predicted_label in rejected_words:
            is_approved = False
        else:
            st.error(
                f"Unrecognized prediction label: {predicted_label}. "
                "Check the values in the Loan_Approved column."
            )
            st.stop()

        st.divider()
        st.subheader("📊 Loan Approval Result")

        if is_approved:
            st.success("✅ Prediction: Loan Approved")
        else:
            st.error("❌ Prediction: Loan Not Approved")

        st.metric("Model test accuracy", f"{test_accuracy:.2%}")

        st.write("**Applicant details submitted:**")
        display_details = {
            column.replace("_", " "): values[column]
            for column in form_columns
        }
        st.dataframe(
            pd.DataFrame([display_details]),
            use_container_width=True,
            hide_index=True
        )

        # Display probability for the predicted class.
        st.metric(
            "Predicted class probability",
            f"{probabilities[list(model.classes_).index(prediction)]:.2%}"
        )

        st.warning(
            "This is an educational machine-learning prediction. "
            "It may be inaccurate or biased and must not be used as "
            "the sole basis for an actual lending decision."
        )

    except Exception as e:
        st.error(f"Prediction failed: {type(e).__name__}: {e}")
