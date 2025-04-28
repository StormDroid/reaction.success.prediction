# Reaction Success Prediction - Full Project Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# --- Setup: Create necessary folders ---
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# --- Step 1: Load and Prepare the Data ---
df = pd.read_csv('C:/Users/Owner/Downloads/reaction_data_large.csv')

# Feature Engineering
df['Energy_Index'] = df['Temperature'] * df['Pressure']
df['pH_Optimality'] = abs(df['pH'] - 7.0)
df['Catalyst_Effect'] = df['Catalyst'] * df['Concentration']
df['Temp_Over_70'] = (df['Temperature'] > 70).astype(int)

features = ['Temperature', 'pH', 'Concentration', 'Pressure', 
            'Energy_Index', 'pH_Optimality', 'Catalyst_Effect', 'Temp_Over_70']
X = df[features]
y = df['Reaction_Success']

# Scale the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Step 2: Hyperparameter Tuning ---
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Save the best model
with open('models/best_random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")
print(f"✅ Accuracy after tuning: {best_rf.score(X_test, y_test):.4f}")

# --- Step 3: Evaluation of Random Forest ---
y_pred_rf = best_rf.predict(X_test)

# Classification Report
print("\n📄 Classification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Failure", "Success"])
disp_rf.plot(cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.savefig('plots/random_forest_confusion_matrix.png')
plt.show()

# Feature Importance
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/random_forest_feature_importance.png')
plt.show()

# --- Step 4: Gradient Boosting Comparison ---
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)

print("\n📄 Classification Report (Gradient Boosting):\n")
print(classification_report(y_test, y_pred_gbc))

# Confusion Matrix for Gradient Boosting
cm_gbc = confusion_matrix(y_test, y_pred_gbc)
disp_gbc = ConfusionMatrixDisplay(confusion_matrix=cm_gbc, display_labels=["Failure", "Success"])
disp_gbc.plot(cmap='Purples')
plt.title("Gradient Boosting Confusion Matrix")
plt.savefig('plots/gradient_boosting_confusion_matrix.png')
plt.show()
