import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap
from sklearn.inspection import permutation_importance

# Streamlit app title and description
st.title("Manufacturing Quality Check Predictor")
st.write("""
This app predicts the Quality Check outcome (Pass/Fail) for a manufacturing process based on Sensor Reading, Product Count, and Machine Status.
The model used is an SVM classifier with optimized parameters (C=0.1, kernel='rbf').
Enter the values below to get a prediction, feature importance, SHAP explanations, and business insights.
""")

# --- Load and Preprocess the Real Dataset ---
# Update the path to where dataset.csv is located in your environment
try:
    df = pd.read_csv("dataset.csv")  # Ideal for deployed environments
except FileNotFoundError:
    st.error("Dataset file 'dataset.csv' not found. Please ensure it is in the project directory.")
    st.stop()

# Preprocess the data (same as original script)
df['QualityCheck'] = df['QualityCheck'].astype(bool)
df['MachineStatus'] = df['MachineStatus'].astype('category')
le = LabelEncoder()
df['MachineStatus_Encoded'] = le.fit_transform(df['MachineStatus'])

# Features and target
X = df[['SensorReading', 'ProductCount', 'MachineStatus_Encoded']]
y = df['QualityCheck'].astype(int)

# Split the data
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train the SVM model
model = SVC(C=0.1, kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# --- User Inputs ---
st.header("Input Parameters")
sensor_reading = st.slider("Sensor Reading", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
product_count = st.slider("Product Count", min_value=0, max_value=40, value=20, step=1)
machine_status = st.selectbox("Machine Status", options=['On', 'Off', 'Standby'])

# Encode the machine status
machine_status_encoded = le.transform([machine_status])[0]

# Create input array for prediction
input_data = np.array([[sensor_reading, product_count, machine_status_encoded]])
input_df = pd.DataFrame(input_data, columns=['SensorReading', 'ProductCount', 'MachineStatus_Encoded'])

# --- Prediction ---
prediction = model.predict(input_data)[0]
prediction_prob = model.predict_proba(input_data)[0][1]
st.header("Prediction")
st.write(f"**Quality Check Prediction**: {'Pass' if prediction == 1 else 'Fail'}")
st.write(f"**Probability of Passing**: {prediction_prob:.2%}")

# --- Feature Importance ---
st.header("Feature Importance")
perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

st.write("**Feature Importance Table**")
st.dataframe(feature_importance)

feature_importance_filtered = feature_importance[feature_importance['Importance'] > 0.0]
if not feature_importance_filtered.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_filtered, palette='viridis', ax=ax)
    plt.title('Feature Importance (Importance > 0.0)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.xlim(0, max(feature_importance_filtered['Importance'].max() + 0.01, 0.1))
    st.pyplot(fig)
    plt.close()
else:
    st.write("No features with importance > 0.0 to display in the feature importance graph.")

# --- SHAP Explanation ---
st.header("SHAP Explanation")
background = shap.sample(X_train, 100)
def model_predict(data):
    return model.predict_proba(data)[:, 1]
explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(input_df)

# Debugging: Display SHAP values
st.write("**SHAP Values for the Prediction**")
shap_df = pd.DataFrame({
    'Feature': X.columns,
    'SHAP Value': shap_values[0]
})
st.dataframe(shap_df)

# Use shap.bar_plot for better compatibility with Streamlit
fig, ax = plt.subplots(figsize=(8, 6))
shap.bar_plot(shap_values[0], feature_names=X.columns, max_display=3, show=False)
plt.title("SHAP Feature Contributions to Prediction")
st.pyplot(fig)
plt.close()

# --- Business Insights ---
st.header("Business Insights and Recommendations")
business_insights = """
### Model Performance
- **Best Model**: SVM (C=0.1, kernel='rbf')
- **Validation F1 Score (After Tuning)**: 0.672
- **Holdout F1 Score**: 0.615

### Key Insights
1. **Cluster Profiles**:
   - **Cluster 0**: High sensor readings (mean=83.60), quality pass rate (54%), mostly 'Standby'.
   - **Cluster 1**: Low sensor readings (mean=16.17), lower quality pass rate (45%), mostly 'On'.
   - **Cluster 2**: Moderate sensor readings (mean=48.99), highest quality pass rate (55%), mostly 'Off'.
2. **Important Features**: SensorReading (Importance: 0.055)
3. **Impact on QualityCheck**:
   - **SensorReading**: Higher values increase likelihood of passing quality check (Mean Pass: 53.49, Mean Fail: 49.38).

### Recommendations
- **Optimize Sensor Readings**: Maintain SensorReading around 50 to align with Cluster 2 conditions, improving quality pass rate to ~55%.
- **Reduce Standby Time**: Minimize 'Standby' mode to enhance efficiency, as Cluster 0 shows lower quality despite high readings.
- **Real-Time Monitoring**: Implement alerts for SensorReading >50 to prevent quality issues.
- **Deploy Predictive Model**: Use this SVM model (F1=0.672) for quality prediction and operational adjustments.
"""
st.markdown(business_insights)

# --- Model Performance Metrics ---
st.header("Model Performance Metrics")
y_pred_val = model.predict(X_val)
y_pred_holdout = model.predict(X_holdout)
val_metrics = {
    'Precision': precision_score(y_val, y_pred_val),
    'Recall': recall_score(y_val, y_pred_val),
    'F1': f1_score(y_val, y_pred_val),
    'ROC-AUC': roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
}
holdout_metrics = {
    'Precision': precision_score(y_holdout, y_pred_holdout),
    'Recall': recall_score(y_holdout, y_pred_holdout),
    'F1': f1_score(y_holdout, y_pred_holdout),
    'ROC-AUC': roc_auc_score(y_holdout, model.predict_proba(X_holdout)[:, 1])
}
metrics_df = pd.DataFrame({
    'Set': ['Validation', 'Holdout'],
    'Precision': [val_metrics['Precision'], holdout_metrics['Precision']],
    'Recall': [val_metrics['Recall'], holdout_metrics['Recall']],
    'F1': [val_metrics['F1'], holdout_metrics['F1']],
    'ROC-AUC': [val_metrics['ROC-AUC'], holdout_metrics['ROC-AUC']]
}).round(3)
st.dataframe(metrics_df)
