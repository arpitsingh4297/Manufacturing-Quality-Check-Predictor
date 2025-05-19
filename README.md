# Manufacturing-Quality-Check-Predictor

Overview
The Manufacturing Quality Check Predictor is a machine learning project designed to predict the quality check outcome (Pass/Fail) for a manufacturing process. The model uses a Support Vector Machine (SVM) classifier trained on features such as SensorReading, ProductCount, and MachineStatus to determine whether a product will pass or fail the quality check. The project includes a Streamlit web application that allows users to input manufacturing parameters and receive predictions, along with feature importance, SHAP explanations, and actionable business insights.
This project was developed as part of an effort to optimize manufacturing processes by identifying key factors affecting quality and providing recommendations to improve pass rates. The model achieves a validation F1 score of 0.672 and a holdout F1 score of 0.615, making it a reliable tool for quality prediction.
Features

Quality Prediction: Predicts whether a product will pass or fail the quality check based on user inputs.
Feature Importance: Displays the importance of each feature (SensorReading, ProductCount, MachineStatus_Encoded) using permutation importance.
SHAP Explanations: Provides SHAP (SHapley Additive exPlanations) bar plots to explain the contribution of each feature to the prediction.
Business Insights: Offers actionable recommendations to improve quality pass rates, such as optimizing SensorReading around 50 and reducing standby time.
Interactive Web App: Built with Streamlit, allowing users to input parameters and visualize results in a user-friendly interface.

Installation
Prerequisites

Python 3.8 or higher
Git (for cloning the repository)

Steps

Clone the Repository:
git clone https://github.com/your-username/manufacturing-quality-check-predictor.git
cd manufacturing-quality-check-predictor


Create a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Install the required Python packages using the provided requirements.txt file:
pip install -r requirements.txt

The requirements.txt includes:
streamlit==1.31.0
pandas==2.2.0
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.13.0
scikit-learn==1.4.0
imbalanced-learn==0.12.0
shap==0.45.0


Verify Installation:Ensure all dependencies are installed correctly by running a simple Python script or checking the versions:
python -c "import streamlit, pandas, numpy, matplotlib, seaborn, sklearn, imblearn, shap; print('All dependencies installed!')"



Usage
Running the Streamlit App

Start the Streamlit App:From the project directory, run the following command to launch the web app:
streamlit run app.py

This will open the app in your default web browser at http://localhost:8501.

Input Parameters:

Sensor Reading: Enter a value between 0 and 100 (default: 50). This represents the sensor measurement from the manufacturing process.
Product Count: Enter a value between 0 and 40 (default: 20). This represents the number of products processed.
Machine Status: Select the machine status (On, Off, or Standby) from the dropdown menu.


View Results:

Prediction: The app will display whether the product is predicted to pass or fail the quality check, along with the probability of passing.
Feature Importance: A table and bar plot showing the importance of each feature.
SHAP Explanation: A bar plot showing the contribution of each feature to the prediction.
Business Insights: Recommendations to improve quality pass rates based on the model’s insights.
Model Performance Metrics: Validation and holdout metrics (Precision, Recall, F1, ROC-AUC) for the SVM model.



Running the Analysis Script
The project also includes a standalone Python script (manufacturing_efficiency_analyzer_debugged.py) that performs the full analysis, including data preprocessing, model training, evaluation, and visualization. To run this script:

Ensure the dataset (dataset.csv) is available in the specified path (or modify the path in the script).
Run the script:python manufacturing_efficiency_analyzer_debugged.py

Note: This script requires a graphical environment to display plots (e.g., Jupyter Notebook with %matplotlib inline, Spyder, or a GUI-enabled IDE).

Project Structure

app.py: The Streamlit app for interactive quality prediction.
manufacturing_efficiency_analyzer_debugged.py: The full analysis script, including data preprocessing, model training, and evaluation.
requirements.txt: List of Python dependencies.
README.md: Project documentation (this file).

Model Details

Algorithm: Support Vector Machine (SVM)
Parameters: C=0.1, kernel='rbf', probability=True
Features:
SensorReading: Numerical (range: ~16–84 based on data)
ProductCount: Numerical (range: ~7–33 based on data)
MachineStatus_Encoded: Categorical (On, Off, Standby encoded as 0, 1, 2)


Target: QualityCheck (binary: True/False)
Performance:
Validation F1 Score (after tuning): 0.672
Holdout F1 Score: 0.615



Data
The project uses a simulated dataset due to the unavailability of the original dataset.csv. The simulated data mimics the characteristics of the original dataset:

SensorReading: Uniform between 16 and 84.
ProductCount: Uniform between 7 and 33.
MachineStatus: Categorical (On, Off, Standby).
QualityCheck: ~60% True, 40% False.

To use the app with real data:

Replace the simulated data in app.py with your dataset by uncommenting the pd.read_csv line and specifying the correct path.
Ensure the dataset has the required columns: SensorReading, ProductCount, MachineStatus, and QualityCheck.

Business Insights
The project provides the following insights and recommendations:

Optimize Sensor Readings: Maintain SensorReading around 50 to align with Cluster 2 conditions, improving the quality pass rate to ~55%.
Reduce Standby Time: Minimize Standby mode to enhance efficiency, as Cluster 0 shows lower quality despite high readings.
Real-Time Monitoring: Implement alerts for SensorReading > 50 to prevent quality issues.
Deploy Predictive Model: Use the SVM model (F1=0.672) for quality prediction and operational adjustments.

Contributing
Contributions are welcome! If you’d like to contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows the project’s style and includes appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, please open an issue on GitHub or contact the project maintainer at your-email@example.com.

Last updated: May 19, 2025
