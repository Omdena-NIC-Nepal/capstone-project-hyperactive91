import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

def render():
    st.title("ML Model Training")
    st.markdown("Train classification and regression models on climate, yield, and glacier datasets.")

    # --- Task Selection ---
    task_type = st.radio("Choose Task Type", ["Classification", "Regression"])

    if task_type == "Classification":
        target_task = st.selectbox("Choose classification target:", [
            "Highheat Year (Binary)",
            "Drought Risk (Multi-Class)",
            "Cereal Yield Class (Binary)",
            "Glacier Retreat Severity (Multi-Class)"
        ])
    else:
        target_task = st.selectbox("Choose regression target:", [
            "Cereal Yield Prediction (Regression)",
            "Glacier Area Loss (Regression)",
            "Glacier Volume Loss (Regression)"
        ])

    # --- Loaders ---
    @st.cache_data
    def load_main():
        return pd.read_csv("data/preprocessed/merged_scaled.csv").dropna()

    @st.cache_data
    def load_glacier():
        return pd.read_csv("data/preprocessed/glacier_features.csv").dropna()

    # === CLASSIFICATION ===
    if task_type == "Classification":
        if target_task == "Glacier Retreat Severity (Multi-Class)":
            data = load_glacier()
            target_col = "retreat_severity"
            drop_cols = ['basin', 'sub-basin']
        else:
            data = load_main()
            if target_task == "Highheat Year (Binary)":
                data['highheat_year'] = (data['highheat_days'] >= 30).astype(int)
                target_col = 'highheat_year'
                drop_cols = ['District', 'YEAR', 'highheat_days', 'drought_risk', 'yield_class']
            elif target_task == "Drought Risk (Multi-Class)":
                def classify_spi(z):
                    if z >= -0.5: return "None"
                    elif z >= -1.0: return "Mild"
                    elif z >= -1.5: return "Moderate"
                    elif z >= -2.0: return "Severe"
                    else: return "Extreme"
                data['drought_risk'] = data['precip_zscore'].apply(classify_spi)
                target_col = 'drought_risk'
                drop_cols = ['District', 'YEAR', 'highheat_days', 'highheat_year', 'yield_class']
            elif target_task == "Cereal Yield Class (Binary)":
                threshold = data['total_yield'].median()
                data['yield_class'] = (data['total_yield'] > threshold).astype(int)
                target_col = 'yield_class'
                drop_cols = ['district_name', 'year', 'total_yield', 'highheat_year', 'drought_risk']

        # Prepare data
        X = data.drop(columns=[col for col in drop_cols if col in data.columns])
        y = data[target_col]
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.markdown(f"Encoded classes: `{list(le.classes_)}`")

        X = X.select_dtypes(include='number').dropna()
        y = pd.Series(y, index=X.index)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        st.subheader("Train Random Forest Classifier")
        n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)

        if st.button("Train Classifier"):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f" Accuracy: {acc:.3f}")
            st.text(classification_report(y_test, y_pred, zero_division=0))
            st.bar_chart(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))
            joblib.dump(model, f"data/preprocessed/interactive_model_{target_col}.joblib")

    # === REGRESSION ===
    else:
        if target_task == "Cereal Yield Prediction (Regression)":
            data = load_main()
            target_col = 'total_yield'
            drop_cols = ['district_name', 'year', 'yield_class', 'heatwave_year', 'drought_risk']
        elif target_task == "Glacier Area Loss (Regression)":
            data = load_glacier()
            target_col = 'area_loss_km2'
            drop_cols = ['retreat_severity', 'basin', 'sub-basin']
        elif target_task == "Glacier Volume Loss (Regression)":
            data = load_glacier()
            target_col = 'volume_loss_km3'
            drop_cols = ['retreat_severity', 'basin', 'sub-basin']

        X = data.drop(columns=[col for col in drop_cols if col in data.columns])
        y = data[target_col]
        X = X.select_dtypes(include='number').dropna()
        y = pd.Series(y, index=X.index)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("Train Random Forest Regressor")
        n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)

        if st.button("Train Regressor"):
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.success(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")
            st.bar_chart(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))
            joblib.dump(model, f"data/preprocessed/interactive_regressor_{target_col}.joblib")
