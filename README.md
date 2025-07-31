**Diabetes Prediction Project**
This project focuses on predicting diabetes using health-related data through machine learning and data analysis techniques. It includes data preprocessing, EDA, feature selection, model comparison, hyperparameter tuning, evaluation metrics, handling imbalanced data with SMOTE, and interpretability using SHAP values.

📂 Project Structure

📁 diabetes-prediction/
│
├── 📄 diabetes_prediction_notebook.ipynb
├── 📄 cleaned_dataset.csv
├── 📄 final_cleaned_dataset.csv
├── 📄 README.md  <-- You are here
└── 📄 requirements.txt (Optional if you want to list dependencies)

📌 Features Covered
Data Cleaning & Preprocessing: Scaling, encoding, handling missing & infinite values

EDA (Exploratory Data Analysis): Pairplots, boxplots, violin plots, heatmaps

Feature Engineering: One-hot encoding, feature importance with Random Forest

Dimensionality Reduction: PCA visualization of clusters

Model Training: Logistic Regression, Decision Tree, Random Forest, SVM

Model Evaluation: Accuracy, Precision, Recall, F1, ROC AUC

Cross-validation & Hyperparameter Tuning: GridSearchCV

Handling Imbalance: SMOTE

Model Explainability: SHAP values for feature contribution insight

🚀 Results
Best Model: Random Forest Classifier

Accuracy: ~97%

Recall (after SMOTE): ~74.6%

Interpretability: SHAP helped identify top health metrics influencing predictions

📊 Visualization Highlights
Pairplot visualizing distribution across diabetes status

Box plots & violin plots for feature analysis

PCA scatter plot clustering

SHAP summary plot for feature importance interpretation

⚙️ Tech Stack
Python

Pandas, NumPy

Scikit-learn

Seaborn & Matplotlib

imbalanced-learn (SMOTE)

SHAP (Model Interpretability)

🔧 Setup

# Clone the repo
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction

# Install dependencies
pip install -r requirements.txt

# Or manually
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn shap
