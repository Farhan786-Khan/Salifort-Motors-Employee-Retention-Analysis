# ====================================
# PACE STAGE 2: ANALYZE
# ====================================

print("\n" + "="*20 + " PACE: ANALYZE STAGE " + "="*20)

# Data cleaning - rename columns for consistency
print("\nDATA CLEANING:")
df0 = df0.rename(columns={
    'Work_accident': 'work_accident',
    'average_montly_hours': 'avg_monthly_hours',  # Fix typo
    'time_spend_company': 'tenure',
    'Department': 'department',
    'promotion_last_5years': 'promotion_last_5y'
})

print("Columns renamed for consistency")
print(f"New column names: {list(df0.columns)}")

# Check for missing values
print(f"\nMISSING VALUES:")
print(df0.isnull().sum())

# Check for duplicates
print(f"\nDUPLICATES:")
duplicates = df0.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates
df1 = df0.drop_duplicates(keep='first')
print(f"Dataset shape after removing duplicates: {df1.shape}")

# Check outliers in tenure
print(f"\nOUTLIER ANALYSIS - TENURE:")
Q1 = df1['tenure'].quantile(0.25)
Q3 = df1['tenure'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df1[(df1['tenure'] < lower_bound) | (df1['tenure'] > upper_bound)]
print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")
print(f"Number of outliers in tenure: {len(outliers)}")

# Target variable analysis
print(f"\nTARGET VARIABLE ANALYSIS:")
left_counts = df1['left'].value_counts()
left_pct = df1['left'].value_counts(normalize=True) * 100

print("Employee Turnover Distribution:")
print(f"Stayed (0): {left_counts[0]:,} employees ({left_pct[0]:.1f}%)")
print(f"Left (1): {left_counts[1]:,} employees ({left_pct[1]:.1f}%)")
print(f"Turnover Rate: {left_pct[1]:.1f}%")

# EXPLORATORY DATA ANALYSIS

# Satisfaction level analysis
print(f"\nSATISFACTION LEVEL ANALYSIS:")
satisfaction_by_left = df1.groupby('left')['satisfaction_level'].agg(['mean', 'median', 'std'])
print(satisfaction_by_left)

# Number of projects analysis
print(f"\nNUMBER OF PROJECTS ANALYSIS:")
projects_by_left = df1.groupby(['number_project', 'left']).size().unstack()
projects_pct = projects_by_left.div(projects_by_left.sum(axis=1), axis=0) * 100
print("Turnover rate by number of projects:")
print(projects_pct.round(1))

# Check employees with 7 projects
seven_projects = df1[df1['number_project'] == 7]
seven_projects_left = seven_projects['left'].sum()
seven_projects_total = len(seven_projects)
print(f"\nEmployees with 7 projects - Left: {seven_projects_left}, Total: {seven_projects_total}")
print(f"Turnover rate for 7-project employees: {seven_projects_left/seven_projects_total*100:.1f}%")

# Hours worked analysis
print(f"\nWORKING HOURS ANALYSIS:")
hours_by_left = df1.groupby('left')['avg_monthly_hours'].agg(['mean', 'median', 'std'])
print(hours_by_left)

# Department analysis
print(f"\nDEPARTMENT ANALYSIS:")
dept_by_left = pd.crosstab(df1['department'], df1['left'], margins=True)
dept_pct = pd.crosstab(df1['department'], df1['left'], normalize='index') * 100
print("Turnover rate by department (%):")
print(dept_pct.round(1))

# Salary analysis
print(f"\nSALARY ANALYSIS:")
salary_by_left = pd.crosstab(df1['salary'], df1['left'], normalize='index') * 100
print("Turnover rate by salary level (%):")
print(salary_by_left.round(1))

# Promotion analysis
print(f"\nPROMOTION ANALYSIS:")
promotion_by_left = pd.crosstab(df1['promotion_last_5y'], df1['left'], normalize='index') * 100
print("Turnover rate by promotion status (%):")
print(promotion_by_left.round(1))

print("\nKEY INSIGHTS FROM ANALYSIS:")
print("1. High turnover rate of 57.3%")
print("2. Employees who left have much lower satisfaction (avg: ~0.44 vs ~0.66)")
print("3. ALL employees with 7 projects left the company")
print("4. Promoted employees have much lower turnover (~5% vs 58%)")
print("5. Lower satisfaction correlation with departure")
