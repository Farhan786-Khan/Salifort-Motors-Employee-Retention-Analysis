# SALIFORT MOTORS EMPLOYEE RETENTION ANALYSIS
# Using PACE Methodology: Plan, Analyze, Construct, Execute

# ====================================
# PACE STAGE 1: PLAN
# ====================================

# Load the dataset and start initial exploration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('default')
sns.set_palette("husl")

print("="*60)
print("SALIFORT MOTORS EMPLOYEE RETENTION ANALYSIS")
print("Using PACE Methodology")
print("="*60)

# Load dataset
df0 = pd.read_csv('salifort_motors_dataset.csv')

print("\n" + "="*20 + " PACE: PLAN STAGE " + "="*20)
print("\nBUSINESS PROBLEM:")
print("- Salifort Motors experiencing high employee turnover")
print("- HR department wants data-driven insights to improve retention")
print("- Goal: Predict which employees will leave and identify key factors")
print("- Stakeholders: HR department, Senior leadership, Department managers")

print(f"\nDATASET OVERVIEW:")
print(f"Shape: {df0.shape}")
print(f"Columns: {list(df0.columns)}")

# Basic dataset information
print("\nDATASET INFO:")
df0.info()
print("\nFIRST 5 ROWS:")
print(df0.head())

print("\nDESCRIPTIVE STATISTICS:")
print(df0.describe())
