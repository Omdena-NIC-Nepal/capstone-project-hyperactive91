# %% [markdown]
# # ML Model

# %% [markdown]
# ### Loading the Datasets

# %%
import pandas as pd

# Load preprocessed feature datasets
climate_yearly = pd.read_csv("../data/preprocessed/climate_yearly.csv")
merged_with_coords = pd.read_csv("../data/preprocessed/merged_with_coords.csv")
merged_scaled = pd.read_csv("../data/preprocessed/merged_scaled.csv")
glacier_features = pd.read_csv("../data/preprocessed/glacier_features.csv")
glacier_long = pd.read_csv("../data/preprocessed/glacier_long.csv")


# %%
!pip show scikit-learn

# %%
import sys
sys.path.append("C:/Users/Wlink/anaconda3/Lib/site-packages")
import sklearn

# %% [markdown]
# ### ✅ Climate Zone Classification Model

# %%
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# --- Step 0: Load data and ensure output directory ---
os.makedirs("../data/preprocessed", exist_ok=True)

try:
    merged_with_coords
except NameError:
    merged_with_coords = pd.read_csv("../data/preprocessed/merged_with_coords.csv")
    print("✅ Loaded merged_with_coords.")

# --- Step 1: Assign climate zones if not present ---
def assign_climate_zone(row):
    if row['avg_temp'] >= 25:
        return 'Tropical'
    elif row['avg_temp'] >= 15:
        return 'Subtropical'
    elif row['avg_temp'] >= 5:
        return 'Temperate'
    else:
        return 'Alpine'

if 'climate_zone' not in merged_with_coords.columns:
    merged_with_coords['climate_zone'] = merged_with_coords.apply(assign_climate_zone, axis=1)

# --- Step 2: Define features and target ---
features = [
    'avg_temp', 'avg_max_temp', 'annual_precip',
    'avg_humidity', 'temp_range_stddev', 'highheat_days'
]
target = 'climate_zone'

# Drop rows with missing values
data = merged_with_coords.dropna(subset=features + [target]).copy()
X = data[features]
y = data[target]

# --- Step 3: Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- Step 4: Define models ---
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
}

summary = []

# --- Step 5: Train, Evaluate, Save ---
for name, model in models.items():
    print(f"\n🔍 {name} Evaluation:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    summary.append({
        'Model': name,
        'Test Accuracy': model.score(X_test, y_test),
        'CV Accuracy': np.mean(cv),
        'CV Std': np.std(cv)
    })

    # Save trained model
    model_key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"../data/preprocessed/climate_zone_{model_key}.joblib"
    joblib.dump(model, filename)
    print(f"💾 Saved to: {filename}")

    # Show feature importances if available
    if hasattr(model, 'feature_importances_'):
        print("📌 Top Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        for feat, score in importances.items():
            print(f"  {feat:<30} → {score:.4f}")
    else:
        print("⚠️ Feature importances not available.")

# --- Step 6: Summary ---
print("\n📋 Model Summary:")
print(pd.DataFrame(summary))


# %% [markdown]
# ### ✅ Extreme Heat Classification based on district-year climate conditions

# %%
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Define binary heatwave label
threshold = 30
climate_yearly['highheat_year'] = (climate_yearly['highheat_days'] >= threshold).astype(int)

# Prepare features and labels
X = climate_yearly.drop(columns=[
    'District', 'YEAR', 'highheat_days', 'highheat_year'
])
X = X.select_dtypes(include=[np.number]).dropna()
y = climate_yearly.loc[X.index, 'highheat_year']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Initialize models
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42)
}

# Create output folder
os.makedirs("../data/preprocessed", exist_ok=True)
summary = []

# Train, Evaluate, Save
for name, model in models.items():
    print(f"\n🔍 {name.upper()} Evaluation")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    summary.append({
        'Model': name,
        'Test Accuracy': model.score(X_test, y_test),
        'CV Accuracy': np.mean(cv),
        'CV Std': np.std(cv)
    })

    # Feature importances (for tree-based models)
    if hasattr(model, "feature_importances_"):
        print("📌 Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(importances.head(10))
    else:
        print("⚠️ Feature importances not available for this model.")

    # Save model
    path = f"../data/preprocessed/heatwave_model_{name}.joblib"
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

# === Step 8: Summary ===
print("\n📋 Model Performance Summary:")
print(pd.DataFrame(summary))


# %% [markdown]
# ### ✅ Drought Risk Category Classification using existing SPI proxy (precip_zscore) in climate_yearly

# %%
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# --- Step 1: Ensure output directory exists ---
os.makedirs("../data/preprocessed", exist_ok=True)

# --- Step 2: Classify drought risk based on SPI-like z-score ---
def classify_spi(z):
    if z >= -0.5:
        return "None"
    elif z >= -1.0:
        return "Mild"
    elif z >= -1.5:
        return "Moderate"
    elif z >= -2.0:
        return "Severe"
    else:
        return "Extreme"

climate_yearly['drought_risk'] = climate_yearly['precip_zscore'].apply(classify_spi)

# --- Step 3: Define features and target ---
X = climate_yearly.drop(columns=[
    'highheat_days', 'highheat_year', 'drought_risk',
    'District', 'YEAR'
])
X = X.select_dtypes(include=[np.number]).dropna()
y = climate_yearly.loc[X.index, 'drought_risk']

# --- Step 4: Encode target labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Step 5: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42
)

# --- Step 6: Define classifiers ---
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42)
}

summary = []

# --- Step 7: Train, evaluate, and save ---
for name, model in models.items():
    print(f"\n🔍 {name.upper()} Evaluation")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
    summary.append({
        'Model': name,
        'Test Accuracy': model.score(X_test, y_test),
        'CV Accuracy': np.mean(cv_scores),
        'CV Std': np.std(cv_scores)
    })

    # Feature importances
    if hasattr(model, 'feature_importances_'):
        print("📌 Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(importances.head(10))
    else:
        print("⚠️ Feature importances not available for this model.")

    # Save model
    path = f"../data/preprocessed/drought_model_{name}.joblib"
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

# --- Step 8: Print Summary ---
print("\n📋 Model Performance Summary:")
print(pd.DataFrame(summary))


# %% [markdown]
# ### ✅ Cereal Yield Classification

# %%
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Ensure output directory exists 
os.makedirs("../data/preprocessed", exist_ok=True)

# Create Binary Yield Label
threshold = merged_scaled['total_yield'].median()
merged_scaled['yield_class'] = (merged_scaled['total_yield'] > threshold).astype(int)

# Define Features and Labels
X_raw = merged_scaled.drop(columns=[
    'total_yield', 'yield_class', 'district_name', 'year',
    'CENTROID_LAT', 'CENTROID_LON'
])

X = pd.get_dummies(X_raw, drop_first=True)
y = merged_scaled['yield_class']

# Drop rows with missing values
X = X.dropna()
y = y.loc[X.index]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# Define and Train Models
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42)
}

summary = []

# Train, Evaluate, Save
for name, model in models.items():
    print(f"\n🔍 {name.upper()} Evaluation:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    summary.append({
        'Model': name,
        'Test Accuracy': model.score(X_test, y_test),
        'CV Accuracy': np.mean(cv_scores),
        'CV Std': np.std(cv_scores)
    })

    # Save model
    model_path = f"../data/preprocessed/yield_model_{name}.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("📌 Top Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(importances.head(10))
    else:
        print("⚠️ Feature importances not available for this model.")

# Summary
print("\n📋 Model Summary:")
print(pd.DataFrame(summary))


# %% [markdown]
# #### ✅ Glacier Retreat Severity Classification

# %%
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# --- Step 0: Ensure output directory exists ---
os.makedirs("../data/preprocessed", exist_ok=True)


X = glacier_features[[
    'glacier_area_1980', 'glacier_area_2010',
    'ice_volume_1980', 'ice_volume_2010',
    'min_elev_1980', 'min_elev_2010',
    'area_loss_km2', 'area_loss_pct',
    'volume_loss_km3', 'volume_loss_pct',
    'elev_rise_m'
]]
y = glacier_features['retreat_severity']

# --- Step 2: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Step 3: Models ---
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

summary = []

# --- Step 4: Train, evaluate, and save models ---
for name, model in models.items():
    print(f"\n🔍 {name.upper()} Evaluation:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Report
    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # CV Score
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_mean, cv_std = scores.mean(), scores.std()
    print(f"📊 5-Fold CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

    # Feature importances
    if hasattr(model, "feature_importances_"):
        print("📌 Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        for feat, val in importances.items():
            print(f"  {feat:<25} → {val:.4f}")
    else:
        print("⚠️ Feature importance not available for this model.")

    # Save model
    model_path = f"../data/preprocessed/glacier_model_{name}.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    # Add to summary
    summary.append({
        'Model': name,
        'Test Accuracy': model.score(X_test, y_test),
        'CV Accuracy Mean': cv_mean,
        'CV Accuracy Std': cv_std
    })

# --- Step 5: Summary Table ---
print("\n📋 Summary Comparison:")
print(pd.DataFrame(summary))


# %% [markdown]
# ### 🔹 Regression Models

# %% [markdown]
# ### ✅ Cereal Yield Prediction (Regression Model)

# %%
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Step 0: Ensure output directory exists ---
os.makedirs("../data/preprocessed", exist_ok=True)

y = merged_scaled['total_yield']
X_raw = merged_scaled.drop(columns=[
    'total_yield', 'yield_class', 'district_name', 'year',
    'CENTROID_LAT', 'CENTROID_LON'
])
X = pd.get_dummies(X_raw, drop_first=True)

# --- Step 2: Drop missing values ---
X = X.dropna()
y = y.loc[X.index]

# --- Step 3: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --- Step 4: Define regression models ---
models = {
    'linear_regression': LinearRegression(),
    'ridge_regression': Ridge(alpha=1.0),
    'lasso_regression': Lasso(alpha=0.1),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

summary = []

# --- Step 5: Train, Evaluate, Save ---
for name, model in models.items():
    print(f"\n🔍 {name.replace('_', ' ').title()} Evaluation:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📈 RMSE: {rmse:.2f}")
    print(f"📉 MAE : {mae:.2f}")
    print(f"🔁 R²  : {r2:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    summary.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2,
        'CV R² Mean': np.mean(cv_scores),
        'CV R² Std': np.std(cv_scores)
    })

    # Save model to disk
    model_path = f"../data/preprocessed/yield_regressor_{name}.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    # Display feature importance or coefficients
    if hasattr(model, 'feature_importances_'):
        print("📌 Top Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(importances.head(10))
    elif hasattr(model, 'coef_'):
        print("📌 Top Coefficients:")
        coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
        print(coefs.head(10))
    else:
        print("⚠️ Feature importance not available.")

# --- Step 6: Print summary ---
print("\n📋 Regression Model Summary:")
print(pd.DataFrame(summary))


# %% [markdown]
# ### ✅ Glacier Area and Volume Loss Regression

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# --- 1. Define targets ---
y_area = glacier_features['area_loss_km2']
y_volume = glacier_features['volume_loss_km3']

# --- 2. Define feature set ---
X = glacier_features.drop(columns=[
    'area_loss_km2', 'volume_loss_km3', 'retreat_severity',
    'area_loss_pct', 'volume_loss_pct',
    'basin', 'sub-basin'  # IDs
])

# --- 3. Drop missing values ---
X = X.dropna()
y_area = y_area.loc[X.index]
y_volume = y_volume.loc[X.index]

# --- 4. Train/test split ---
X_train, X_test, ya_train, ya_test = train_test_split(X, y_area, test_size=0.25, random_state=42)
_, _, yv_train, yv_test = train_test_split(X, y_volume, test_size=0.25, random_state=42)

# --- 5. Define models ---
models = {
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

os.makedirs("../data/preprocessed", exist_ok=True)

# --- 6. Evaluation and saving ---
def evaluate_and_save_model(name, model, X_train, y_train, X_test, y_test, X_all, y_all, label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n🔍 {name.upper()} ({label}) Regression:")
    print(f"📈 RMSE: {rmse:.2f} | 📉 MAE: {mae:.2f} | 🔁 R²: {r2:.4f}")

    cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring='r2')
    print(f"📊 CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if hasattr(model, 'feature_importances_'):
        print("📌 Top Feature Importances:")
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(fi.head(5))
    elif hasattr(model, 'coef_'):
        print("📌 Top Coefficients:")
        coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
        print(coefs.head(5))
    else:
        print("⚠️ Feature importances not available.")

    path = f"../data/preprocessed/glacier_regressor_{label}_{name}.joblib"
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

# --- 7. Predict and save for Area Loss ---
print("\n🌐 Predicting Glacier Area Loss:")
for name, model in models.items():
    evaluate_and_save_model(name, model, X_train, ya_train, X_test, ya_test, X, y_area, "area")

# --- 8. Predict and save for Volume Loss ---
print("\n❄️ Predicting Glacier Volume Loss:")
for name, model in models.items():
    evaluate_and_save_model(name, model, X_train, yv_train, X_test, yv_test, X, y_volume, "volume")


# %% [markdown]
# ### ✅ Heatwave Days Regression

# %%
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Step 0: Ensure output directory exists ===
os.makedirs("../data/preprocessed", exist_ok=True)


y = climate_yearly['highheat_days']

# === Step 2: Define features ===
X = climate_yearly.drop(columns=[
    'District', 'YEAR', 'highheat_days', 'highheat_year'  # remove ID/leakage
])

# One-hot encode if any categorical columns exist
X = pd.get_dummies(X, drop_first=True)

# Drop rows with missing data
X = X.dropna()
y = y.loc[X.index]

# === Step 3: Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === Step 4: Define models ===
models = {
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# === Step 5: Train and evaluate ===
summary = []

for name, model in models.items():
    print(f"\n🔍 {name.replace('_', ' ').title()} Regression:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📈 RMSE: {rmse:.2f}")
    print(f"📉 MAE : {mae:.2f}")
    print(f"🔁 R²  : {r2:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"📊 CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature insights
    if hasattr(model, 'feature_importances_'):
        print("📌 Top Feature Importances:")
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(fi.head(10))
    elif hasattr(model, 'coef_'):
        print("📌 Top Coefficients:")
        coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
        print(coefs.head(10))
    else:
        print("⚠️ Feature importances not available.")

    # Save model
    model_path = f"../data/preprocessed/heatwave_regressor_{name}.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    summary.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2,
        'CV R² Mean': cv_scores.mean(),
        'CV R² Std': cv_scores.std()
    })

# === Step 6: Summary Table ===
print("\n📋 Regression Model Summary:")
print(pd.DataFrame(summary))


# %% [markdown]
# ### ✅ Drought Severity Regression using precip_zscore (SPI proxy)

# %%
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Ensure output directory exists ===
os.makedirs("../data/preprocessed", exist_ok=True)

# === 1. Define target and features ===
y = climate_yearly['precip_zscore']  # SPI-like drought index

X = climate_yearly.drop(columns=[
    'precip_zscore', 'drought_risk', 'highheat_days', 'highheat_year',
    'District', 'YEAR'
])

# Convert categoricals (if any)
X = pd.get_dummies(X, drop_first=True)

# Drop rows with missing values
X = X.dropna()
y = y.loc[X.index]

# === 2. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === 3. Define regression models ===
models = {
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

summary = []

# === 4. Evaluate and save each model ===
for name, model in models.items():
    print(f"\n🔍 {name.upper()} Regression:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📈 RMSE: {rmse:.4f}")
    print(f"📉 MAE : {mae:.4f}")
    print(f"🔁 R²  : {r2:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"📊 CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save summary
    summary.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2,
        'CV R² Mean': cv_scores.mean(),
        'CV R² Std': cv_scores.std()
    })

    # Feature importances / coefficients
    if hasattr(model, 'feature_importances_'):
        print("📌 Top Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(importances.head(10))
    elif hasattr(model, 'coef_'):
        print("📌 Top Coefficients:")
        coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
        print(coefs.head(10))
    else:
        print("⚠️ Feature importance not available.")

    # Save model
    path = f"../data/preprocessed/drought_regressor_{name}.joblib"
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

# === 5. Print Summary Table ===
print("\n📋 Regression Model Summary:")
print(pd.DataFrame(summary))


# %% [markdown]
# ### 🔹 Forecasting

# %% [markdown]
# ### ✅ Heatwarve Days Forecasting up to 2050

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import os

#  input features 
features = [
    'avg_temp', 'avg_max_temp', 'temp_range_stddev', 'avg_humidity',
    'avg_wind', 'annual_precip', 'precip_zscore',
    'avg_temp_lag1', 'annual_precip_lag1', 'precip_zscore_lag1',
    'temp_range_stddev_lag1', 'highheat_days_lag1'
]


df = climate_yearly.copy()
df_model = df[['District', 'YEAR', 'highheat_days'] + features].dropna()

X = df_model[features]
y = df_model['highheat_days']

# === 3. Train model ===
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# === 4. Prepare forecast years and districts ===
future_years = list(range(2020, 2051))
districts = df_model['District'].unique()
forecast_rows = []

# === 5. Simulate future values per district ===
for district in districts:
    district_df = df_model[df_model['District'] == district]
    if district_df.empty:
        continue

    last_row = district_df.loc[district_df['YEAR'].idxmax()].copy()

    for year in future_years:
        new_row = {'District': district, 'YEAR': year}

        for col in features:
            if 'lag1' in col:
                base_col = col.replace('_lag1', '')
                val = last_row.get(base_col, df_model[base_col].mean())
                new_row[col] = val
            else:
                val = last_row.get(col, df_model[col].mean())
                new_row[col] = val + np.random.normal(0, 0.1)  # small noise

        forecast_rows.append(new_row)
        last_row = pd.Series(new_row)

# === 6. Predict and output ===
forecast_df = pd.DataFrame(forecast_rows)
forecast_df = forecast_df.dropna(subset=features)
forecast_df['predicted_highheat_days'] = model.predict(forecast_df[features])

# === 7. Save output ===
os.makedirs("../data/preprocessed", exist_ok=True)
forecast_df.to_csv("../data/preprocessed/highheat_days_forecast_2020_2050.csv", index=False)

# === 8. Preview ===
print("✅ Forecast Preview (2050):")
print(forecast_df[forecast_df['YEAR'] == 2050][['District', 'YEAR', 'predicted_highheat_days']].round(1))
print("💾 Forecast saved to: ../data/preprocessed/highheat_days_forecast_2020_2050.csv")


# %% [markdown]
# ### ✅ Forecasting Drought Severity to 2050

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import os

# Define features used for SPI prediction
features = [
    'annual_precip', 'avg_temp', 'avg_max_temp', 'avg_humidity',
    'avg_wind', 'temp_range_stddev',
    'annual_precip_lag1', 'avg_temp_lag1', 'temp_range_stddev_lag1'
]
target = 'precip_zscore'  # SPI proxy

df = climate_yearly.copy()
df_model = df[['District', 'YEAR', target] + features].dropna()

# Train Gradient Boosting model on historical SPI data ---
X = df_model[features]
y = df_model[target]
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

#  Forecast SPI for each district from 2020 to 2050 ---
future_years = list(range(2020, 2051))
districts = df_model['District'].unique()
forecast_rows = []

for district in districts:
    district_df = df_model[df_model['District'] == district].copy()
    if district_df.empty:
        continue

    last_known = district_df[district_df['YEAR'] == district_df['YEAR'].max()]
    if last_known.empty:
        continue

    last_row = last_known.iloc[0].copy()

    for year in future_years:
        new_row = {'District': district, 'YEAR': year}
        for col in features:
            if 'lag1' in col:
                base_col = col.replace('_lag1', '')
                val = last_row.get(base_col, df_model[base_col].mean())
                new_row[col] = val
            else:
                base_val = last_row.get(col, df_model[col].mean())
                new_row[col] = base_val + np.random.normal(0, 0.1)
        forecast_rows.append(new_row)
        last_row = pd.Series(new_row)

# --- 5. Predict SPI and classify drought severity ---
forecast_df = pd.DataFrame(forecast_rows)
forecast_df = forecast_df.dropna(subset=features)
forecast_df['predicted_spi'] = model.predict(forecast_df[features])

def classify_spi(z):
    if z >= -0.5:
        return "None"
    elif z >= -1.0:
        return "Mild"
    elif z >= -1.5:
        return "Moderate"
    elif z >= -2.0:
        return "Severe"
    else:
        return "Extreme"

forecast_df['drought_risk'] = forecast_df['predicted_spi'].apply(classify_spi)

# --- 6. Save and preview forecast ---
os.makedirs("../data/preprocessed", exist_ok=True)
forecast_path = "../data/preprocessed/drought_forecast_spi_2020_2050.csv"
forecast_df.to_csv(forecast_path, index=False)

print("✅ Drought Forecast Preview (2050):")
print(forecast_df[forecast_df['YEAR'] == 2050][['District', 'YEAR', 'predicted_spi', 'drought_risk']].round(2))
print(f"💾 Forecast saved to: {forecast_path}")


# %% [markdown]
# ### ✅ Climate Forecast Up to 2050

# %%
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Define features for forecasting ===
features = [
    'avg_temp', 'avg_max_temp', 'temp_range_stddev', 'avg_humidity',
    'avg_wind', 'annual_precip',
    'avg_temp_lag1', 'annual_precip_lag1', 'temp_range_stddev_lag1'
]


df = climate_yearly.copy()
df_model = df[['District', 'YEAR'] + features].dropna()

# Train Gradient Boosting model on historical avg_temp ===
X = df_model[features]
y = df_model['avg_temp']

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save trained model
os.makedirs("../data/preprocessed", exist_ok=True)
joblib.dump(model, "../data/preprocessed/climate_regressor_avg_temp_gradient_boosting.joblib")

# Define forecast range ===
future_years = list(range(2020, 2051))
districts = df_model['District'].unique()
forecast_rows = []

#  Forward simulate features by district ===
for district in districts:
    district_df = df_model[df_model['District'] == district]
    if district_df.empty:
        continue

    last_row = district_df.loc[district_df['YEAR'].idxmax()].copy()

    for year in future_years:
        new_row = {'District': district, 'YEAR': year}

        for col in features:
            if 'lag1' in col:
                base_col = col.replace('_lag1', '')
                val = last_row.get(base_col, df_model[base_col].mean())
                new_row[col] = val
            else:
                val = last_row.get(col, df_model[col].mean())
                new_row[col] = val + np.random.normal(0, 0.1)

        forecast_rows.append(new_row)
        last_row = pd.Series(new_row)

# Predict future avg_temp ===
forecast_df = pd.DataFrame(forecast_rows)
forecast_df = forecast_df.dropna(subset=features)
forecast_df['predicted_avg_temp'] = model.predict(forecast_df[features])

# Save results ===
forecast_path = "../data/preprocessed/climate_forecast_2020_2050.csv"
forecast_df.to_csv(forecast_path, index=False)

# Preview ===
print("✅ Climate Forecast Preview (2050):")
print(forecast_df[forecast_df['YEAR'] == 2050][['District', 'YEAR', 'predicted_avg_temp']].round(2))
print(f"💾 Forecast saved to: {forecast_path}")


# %% [markdown]
# ### ✅ Glacier Area, Ice Volume, and Minimum Elevation Forecast

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import os


os.makedirs("../data/preprocessed", exist_ok=True)

df = glacier_long.copy()

# Encode categorical variables ---
df['basin_code'] = df['basin'].astype('category').cat.codes
df['subbasin_code'] = df['sub-basin'].astype('category').cat.codes

# Define input features and target variables ---
features = ['year', 'basin_code', 'subbasin_code']
targets = ['glacier_area', 'ice_volume', 'min_elev']

# Train one model per target ---
models = {}
for target in targets:
    X = df[features]
    y = df[target]
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    models[target] = model

# Generate forecast input combinations ---
future_years = [2020, 2030, 2040, 2050]
basin_info = df[['basin', 'sub-basin', 'basin_code', 'subbasin_code']].drop_duplicates()
forecast_rows = []

for year in future_years:
    for _, row in basin_info.iterrows():
        input_dict = {
            'year': year,
            'basin_code': row['basin_code'],
            'subbasin_code': row['subbasin_code']
        }
        result = {
            'year': year,
            'basin': row['basin'],
            'sub-basin': row['sub-basin']
        }
        for target in targets:
            prediction = models[target].predict(pd.DataFrame([input_dict]))[0]
            result[f'predicted_{target}'] = round(float(prediction), 4)
        forecast_rows.append(result)

# Create forecast DataFrame ---
forecast_df = pd.DataFrame(forecast_rows)

# Preview and save ---
print("✅ Glacier Forecast for 2050:")
print(forecast_df[forecast_df['year'] == 2050].round(2))

forecast_path = "../data/preprocessed/glacier_forecast_2020_2050.csv"
forecast_df.to_csv(forecast_path, index=False)
print(f"💾 Forecast saved to: {forecast_path}")


# %%



