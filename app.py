import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

st.set_page_config(
    page_title="Credit Card Default Classification",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Card Default Prediction")
st.write(
    "Evaluate multiple machine learning classification models "
    "on uploaded credit-card default test data."
)
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),
        "K-Nearest Neighbors": joblib.load(
            "model/knn.pkl"
        ),
        "Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),
        "Support Vector Machine": joblib.load(
            "model/svm.pkl"
        ),
        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        ),
    }

models = load_models()
st.subheader("1. Select a Classification Model")

selected_model_name = st.selectbox(
    "Choose a model:",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.success(f"Selected model: {selected_model_name}")
st.subheader("2. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    st.write("Uploaded Test Data Preview")
    st.dataframe(test_data.head())

    st.write("Dataset shape:", test_data.shape)
        # Check whether target column exists
    if "DEFAULT" not in test_data.columns:
        st.error("The uploaded test dataset must contain the 'DEFAULT' target column.")
        st.stop()

    # Separate features and target
    X_test_app = test_data.drop(columns=["DEFAULT"])
    y_test_app = test_data["DEFAULT"]

    # Generate class predictions
    y_pred_app = selected_model.predict(X_test_app)

    # Generate scores/probabilities for ROC-AUC
    if hasattr(selected_model, "predict_proba"):
        y_score_app = selected_model.predict_proba(X_test_app)[:, 1]

    elif hasattr(selected_model, "decision_function"):
        y_score_app = selected_model.decision_function(X_test_app)

    else:
        y_score_app = None
        st.subheader("3. Model Evaluation")

    accuracy = accuracy_score(y_test_app, y_pred_app)
    precision = precision_score(y_test_app, y_pred_app, zero_division=0)
    recall = recall_score(y_test_app, y_pred_app, zero_division=0)
    f1 = f1_score(y_test_app, y_pred_app, zero_division=0)
    mcc = matthews_corrcoef(y_test_app, y_pred_app)

    if y_score_app is not None:
        auc = roc_auc_score(y_test_app, y_score_app)
    else:
        auc = np.nan
    st.subheader("3. Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    st.subheader("4. Confusion Matrix and Classification Report")

    # Confusion Matrix
    cm = confusion_matrix(y_test_app, y_pred_app)

    st.write("#### Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Default (0)", "Actual: Default (1)"],
        columns=["Predicted: No Default (0)", "Predicted: Default (1)"]
    )

    st.dataframe(cm_df, use_container_width=True)

    # Classification Report
    st.write("#### Classification Report")

    report = classification_report(
        y_test_app,
        y_pred_app,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )