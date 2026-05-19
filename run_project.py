import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-Learn tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, ConfusionMatrixDisplay, 
                             roc_curve, auc, precision_recall_fscore_support)

# Imbalanced-Learn for SMOTE
from imblearn.over_sampling import SMOTE

# Joblib for model saving
import joblib

import warnings
# Ignore warnings for clean output
warnings.filterwarnings('ignore')

# Create directory to save images if needed
output_dir = '.'

print("=== CELL 2: Load Dataset ===")
# Load the dataset
df = pd.read_csv('player_injuries_balanced.csv')
print(f"Dataset Shape: {df.shape}\n")
print("First 5 Rows:")
print(df.head())
print("\nTarget Column (is_ligament_injury) Value Counts:")
print(df['is_ligament_injury'].value_counts())

print("\n=== CELL 3: Data Preprocessing ===")
# 1. Drop useless columns (Name, Team Name, Season, dates, missed, after, opposition)
cols_to_drop = ['Name', 'Team Name', 'Season', 'Date of Injury', 'Date of return']
for col in df.columns:
    if 'missed' in col.lower() or 'after' in col.lower() or 'opposition' in col.lower():
        cols_to_drop.append(col)

# Keep only columns that exist in the dataframe to avoid errors
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_cols_to_drop)

# 2. Replace all "N.A." strings with NaN
df = df.replace('N.A.', np.nan)

# 3. Clean player rating columns using regex to extract numeric parts
rating_cols = [
    'Match1_before_injury_Player_rating', 
    'Match2_before_injury_Player_rating', 
    'Match3_before_injury_Player_rating'
]
for col in rating_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# 4. Encode Result columns (win=2, draw=1, lose=0)
result_cols = [
    'Match1_before_injury_Result', 
    'Match2_before_injury_Result', 
    'Match3_before_injury_Result'
]
result_mapping = {'win': 2, 'draw': 1, 'lose': 0}
for col in result_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map(result_mapping)

# 5. One-hot encode the Position column
if 'Position' in df.columns:
    df = pd.get_dummies(df, columns=['Position'], drop_first=False, dtype=int)

# 6. Convert all columns except 'Injury' to numeric
for col in df.columns:
    if col != 'Injury':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 7. Fill remaining NaN values with column mean (only for numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print(f"Final Shape after Preprocessing: {df.shape}")
print(f"Total Missing Values Remaining (excluding 'Injury'): {df.drop(columns=['Injury'], errors='ignore').isnull().sum().sum()}")
print("\nFeature Columns Being Used:")
print(list(df.columns))

print("\n=== CELL 4: Exploratory Data Analysis (EDA) ===")
# 1. Bar chart — class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='is_ligament_injury', palette='Set2')
plt.title('Class Distribution: Ligament vs Non-Ligament Injuries')
plt.xlabel('Is Ligament Injury (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()
print("Saved: class_distribution.png")

# 2. Bar chart — top 10 most common injury types
if 'Injury' in df.columns:
    plt.figure(figsize=(10, 6))
    top_injuries = df['Injury'].value_counts().head(10)
    sns.barplot(y=top_injuries.index, x=top_injuries.values, palette='viridis')
    plt.title('Top 10 Most Common Injury Types')
    plt.xlabel('Frequency')
    plt.ylabel('Injury Type')
    plt.tight_layout()
    plt.savefig('top_10_injuries.png')
    plt.close()
    print("Saved: top_10_injuries.png")

# 3. Box plot — Age distribution by injury class
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='is_ligament_injury', y='Age', palette='Set1')
plt.title('Age Distribution by Ligament Injury')
plt.xlabel('Is Ligament Injury (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.close()
print("Saved: age_distribution.png")

# 4. Box plot — FIFA rating distribution by injury class
if 'FIFA rating' in df.columns:
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x='is_ligament_injury', y='FIFA rating', palette='Set3')
    plt.title('FIFA Rating Distribution by Ligament Injury')
    plt.xlabel('Is Ligament Injury (0 = No, 1 = Yes)')
    plt.ylabel('FIFA Rating')
    plt.tight_layout()
    plt.savefig('fifa_rating_distribution.png')
    plt.close()
    print("Saved: fifa_rating_distribution.png")

# 5. Heatmap — correlation matrix of all numeric features
plt.figure(figsize=(12, 10))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()
print("Saved: correlation_matrix.png")

print("\n=== CELL 5: Prepare X and y, Train-Test Split ===")
cols_to_exclude = ['is_ligament_injury', 'Injury']
X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])
y = df['is_ligament_injury']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\n=== CELL 6: Handle Class Imbalance with SMOTE ===")
print("Class Distribution Before SMOTE (y_train):")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class Distribution After SMOTE (y_train_smote):")
print(y_train_smote.value_counts())

print("\n=== CELL 7: Train All Three Models ===")
# 1. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_smote, y_train_smote)
print("Random Forest Model trained successfully")

# 2. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_train_smote, y_train_smote)
print("Decision Tree Model trained successfully")

# 3. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_smote, y_train_smote)
print("Naive Bayes Model trained successfully")

print("\n=== CELL 8: Evaluate All Models and Compare ===")
models = {
    'Random Forest': rf_model,
    'Decision Tree': dt_model,
    'Naive Bayes': nb_model
}

metrics_list = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Evaluation for {name} ---")
    print(f"Accuracy Score: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    metrics_list.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    })

print("\n--- MODEL COMPARISON TABLE ---")
comparison_df = pd.DataFrame(metrics_list)
print(comparison_df.to_string(index=False))

print("\n=== CELL 9: Visualize Results ===")
# 1. RF Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues')
plt.title('Random Forest - Confusion Matrix')
plt.tight_layout()
plt.savefig('cm_random_forest.png')
plt.close()
print("Saved: cm_random_forest.png")

# 2. DT Confusion Matrix
ConfusionMatrixDisplay.from_estimator(dt_model, X_test, y_test, cmap='Oranges')
plt.title('Decision Tree - Confusion Matrix')
plt.tight_layout()
plt.savefig('cm_decision_tree.png')
plt.close()
print("Saved: cm_decision_tree.png")

# 3. NB Confusion Matrix
ConfusionMatrixDisplay.from_estimator(nb_model, X_test, y_test, cmap='Greens')
plt.title('Naive Bayes - Confusion Matrix')
plt.tight_layout()
plt.savefig('cm_naive_bayes.png')
plt.close()
print("Saved: cm_naive_bayes.png")

# 4. Feature Importances for Random Forest (top 15)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]
feature_names = X.columns[indices]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names, palette='magma')
plt.title('Top 15 Feature Importances - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importances_rf.png')
plt.close()
print("Saved: feature_importances_rf.png")

# 5. Bar chart comparing Accuracy
model_names = comparison_df['Model']
accuracies = comparison_df['Accuracy']
plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies, palette='pastel')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.close()
print("Saved: accuracy_comparison.png")

# 6. Bar chart comparing F1-Score
f1_scores = comparison_df['F1-Score']
plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=f1_scores, palette='muted')
plt.title('Model F1-Score Comparison')
plt.ylim(0, 1.1)
plt.ylabel('F1-Score')
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('f1_score_comparison.png')
plt.close()
print("Saved: f1_score_comparison.png")

# 7. ROC Curve for Random Forest
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve_rf.png')
plt.close()
print("Saved: roc_curve_rf.png")

print("\n=== CELL 10: Cross Validation ===")
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"Cross Validation Scores (5 Folds): {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"CV Standard Deviation: {cv_scores.std():.4f}")

print("\n=== CELL 11: Predict on a New Sample ===")
fake_player = X_test.iloc[[0]].copy()
if 'Age' in fake_player.columns:
    fake_player['Age'] = 31
if 'FIFA rating' in fake_player.columns:
    fake_player['FIFA rating'] = 85

rf_prediction = rf_model.predict(fake_player)[0]
rf_probabilities = rf_model.predict_proba(fake_player)[0]
print("--- New Player Prediction (Random Forest) ---")
if rf_prediction == 1:
    print("Prediction: LIGAMENT INJURY (Class 1)")
else:
    print("Prediction: NON-LIGAMENT INJURY (Class 0)")
print(f"Probability of Ligament Injury: {rf_probabilities[1]*100:.2f}%")
print(f"Probability of Non-Ligament Injury: {rf_probabilities[0]*100:.2f}%")

print("\n=== CELL 12: Save the Model ===")
model_filename = 'sports_injury_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"Success! Model successfully saved as: {model_filename}")
