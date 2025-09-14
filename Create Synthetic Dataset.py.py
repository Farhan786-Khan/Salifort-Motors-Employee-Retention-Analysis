# Create synthetic Salifort Motors dataset matching the exact structure described
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset matching Salifort Motors structure
n_samples = 14999

# Create synthetic data based on the pattern described in the research
data = {
    'satisfaction_level': np.random.beta(2, 2, n_samples),  # [0-1]
    'last_evaluation': np.random.uniform(0.36, 1.0, n_samples),  # [0.36-1]
    'number_project': np.random.choice([2, 3, 4, 5, 6, 7], n_samples, p=[0.05, 0.2, 0.3, 0.25, 0.15, 0.05]),
    'average_monthly_hours': np.random.normal(201, 49, n_samples),  # Based on overwork patterns
    'time_spend_company': np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10], n_samples, 
                                         p=[0.15, 0.2, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02]),
    'Work_accident': np.random.choice([0, 1], n_samples, p=[0.855, 0.145]),
    'promotion_last_5years': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
    'Department': np.random.choice(['sales', 'technical', 'support', 'IT', 'product_mng', 
                                  'marketing', 'RandD', 'accounting', 'hr', 'management'], 
                                 n_samples, p=[0.27, 0.2, 0.15, 0.08, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]),
    'salary': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.49, 0.43, 0.08])
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate 'left' variable based on realistic patterns from research
# Higher probability of leaving based on:
# - Low satisfaction
# - High number of projects (especially 7)
# - Very high or very low hours
# - No promotion
# - Lower evaluation scores

leave_prob = np.zeros(n_samples)

for i in range(n_samples):
    prob = 0.1  # Base probability
    
    # Satisfaction level effect (strong negative correlation)
    prob += (1 - df.loc[i, 'satisfaction_level']) * 0.4
    
    # Number of projects effect
    if df.loc[i, 'number_project'] == 7:
        prob += 0.8  # Almost all with 7 projects leave
    elif df.loc[i, 'number_project'] >= 5:
        prob += 0.2
    elif df.loc[i, 'number_project'] <= 2:
        prob += 0.1
    
    # Hours effect (overwork and underwork)
    if df.loc[i, 'average_monthly_hours'] > 250:
        prob += 0.3
    elif df.loc[i, 'average_monthly_hours'] < 150:
        prob += 0.15
    
    # Promotion effect
    if df.loc[i, 'promotion_last_5years'] == 0:
        prob += 0.1
    else:
        prob -= 0.2  # Much less likely to leave if promoted
    
    # Tenure effect
    if df.loc[i, 'time_spend_company'] >= 6:
        prob -= 0.2  # Long tenure employees stay
    elif df.loc[i, 'time_spend_company'] == 4:
        prob += 0.1  # 4-year mark dissatisfaction
    
    # Evaluation effect
    if df.loc[i, 'last_evaluation'] < 0.5:
        prob += 0.15
    
    # Cap probability between 0 and 1
    prob = max(0, min(1, prob))
    leave_prob[i] = prob

# Generate left variable based on probabilities
df['left'] = np.random.binomial(1, leave_prob, n_samples)

# Add some duplicates to match the dataset description
duplicate_indices = np.random.choice(df.index, size=3008, replace=False)
df_duplicates = df.loc[duplicate_indices].copy()
df = pd.concat([df, df_duplicates], ignore_index=True)

# Ensure we have exactly 15000 rows
df = df.iloc[:15000].reset_index(drop=True)

# Clean up column names to match the dataset
df.columns = ['satisfaction_level', 'last_evaluation', 'number_project', 
              'average_montly_hours', 'time_spend_company', 'Work_accident', 
              'promotion_last_5years', 'Department', 'salary', 'left']

print("Salifort Motors Dataset Created Successfully!")
print(f"Dataset shape: {df.shape}")
print(f"\nDataset info:")
print(df.info())
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget variable distribution:")
print(df['left'].value_counts(normalize=True))