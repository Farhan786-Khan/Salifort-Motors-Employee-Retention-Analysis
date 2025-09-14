# ====================================
# PACE STAGE 3: CONSTRUCT
# ====================================

print("\n" + "="*20 + " PACE: CONSTRUCT STAGE " + "="*20)
print("MACHINE LEARNING MODEL DEVELOPMENT")

# Prepare data for modeling
print("\nDATA PREPROCESSING FOR MODELING:")

# Create a copy for modeling
df_model = df1.copy()

# Handle categorical variables
# Encode salary as ordinal
salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
df_model['salary_encoded'] = df_model['salary'].map(salary_mapping)

# One-hot encode department
dept_dummies = pd.get_dummies(df_model['department'], prefix='dept', drop_first=True)
df_model = pd.concat([df_model, dept_dummies], axis=1)

# Remove outliers in tenure for logistic regression
df_logreg = df_model[(df_model['tenure'] >= 1.5) & (df_model['tenure'] <= 5.5)]
print(f"Dataset shape after outlier removal: {df_logreg.shape}")

# Feature selection
feature_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 
                'avg_monthly_hours', 'tenure', 'work_accident', 'promotion_last_5y', 
                'salary_encoded'] + [col for col in df_logreg.columns if col.startswith('dept_')]

X = df_logreg[feature_cols]
y = df_logreg['left']

print(f"Features selected: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=42, stratify=y)

print(f"\nTRAIN/TEST SPLIT:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training target distribution: {y_train.value_counts(normalize=True).round(3).to_dict()}")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nMODEL DEVELOPMENT:")
print("Building multiple models for comparison:")
print("1. Logistic Regression")
print("2. Decision Tree")
print("3. Random Forest")
print("4. Gradient Boosting")

# MODEL 1: LOGISTIC REGRESSION
print("\n1. LOGISTIC REGRESSION MODEL:")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predictions
log_pred = log_reg.predict(X_test_scaled)
log_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
log_accuracy = accuracy_score(y_test, log_pred)
log_precision = precision_score(y_test, log_pred)
log_recall = recall_score(y_test, log_pred)
log_f1 = f1_score(y_test, log_pred)
log_auc = roc_auc_score(y_test, log_pred_proba)

print(f"Accuracy: {log_accuracy:.3f}")
print(f"Precision: {log_precision:.3f}")
print(f"Recall: {log_recall:.3f}")
print(f"F1-Score: {log_f1:.3f}")
print(f"AUC-ROC: {log_auc:.3f}")

# MODEL 2: DECISION TREE
print("\n2. DECISION TREE MODEL:")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_pred_proba = dt_model.predict_proba(X_test)[:, 1]

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_pred_proba)

print(f"Accuracy: {dt_accuracy:.3f}")
print(f"Precision: {dt_precision:.3f}")
print(f"Recall: {dt_recall:.3f}")
print(f"F1-Score: {dt_f1:.3f}")
print(f"AUC-ROC: {dt_auc:.3f}")

# MODEL 3: RANDOM FOREST
print("\n3. RANDOM FOREST MODEL:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f"Accuracy: {rf_accuracy:.3f}")
print(f"Precision: {rf_precision:.3f}")
print(f"Recall: {rf_recall:.3f}")
print(f"F1-Score: {rf_f1:.3f}")
print(f"AUC-ROC: {rf_auc:.3f}")

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES (Random Forest):")
print(feature_importance.head(10).to_string(index=False))

# MODEL 4: GRADIENT BOOSTING
print("\n4. GRADIENT BOOSTING MODEL:")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)
gb_auc = roc_auc_score(y_test, gb_pred_proba)

print(f"Accuracy: {gb_accuracy:.3f}")
print(f"Precision: {gb_precision:.3f}")
print(f"Recall: {gb_recall:.3f}")
print(f"F1-Score: {gb_f1:.3f}")
print(f"AUC-ROC: {gb_auc:.3f}")

# MODEL COMPARISON SUMMARY
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [log_accuracy, dt_accuracy, rf_accuracy, gb_accuracy],
    'Precision': [log_precision, dt_precision, rf_precision, gb_precision],
    'Recall': [log_recall, dt_recall, rf_recall, gb_recall],
    'F1-Score': [log_f1, dt_f1, rf_f1, gb_f1],
    'AUC-ROC': [log_auc, dt_auc, rf_auc, gb_auc]
}).round(3)

print(comparison_df.to_string(index=False))

# Identify best model
best_model_idx = comparison_df['F1-Score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_f1_score = comparison_df.loc[best_model_idx, 'F1-Score']

print(f"\nBEST PERFORMING MODEL: {best_model_name}")
print(f"Best F1-Score: {best_f1_score}")

# Save comparison results
comparison_df.to_csv('model_comparison_results.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print(f"\nResults saved to CSV files for further analysis")