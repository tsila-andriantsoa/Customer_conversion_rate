import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import joblib


# Data importation
df_train = pd.read_csv('data/customer_conversion_training_dataset.csv')
df_test = pd.read_csv('data/customer_conversion_testing_dataset.csv')
print("Data importation step done !")

## Rename columns
df_train.columns = ['LeadID', 'Age', 'Gender', 'Location', 'LeadSource',
       'TimeSpentMinutes', 'PagesViewed', 'LeadStatus', 'EmailSent',
       'DeviceType', 'ReferralSource', 'FormSubmissions', 'Downloads',
       'CTR_ProductPage', 'ResponseTimeHours', 'FollowUpEmails',
       'SocialMediaEngagement', 'PaymentHistory', 'Conversion']

df_test.columns = ['LeadID', 'Age', 'Gender', 'Location', 'LeadSource',
       'TimeSpentMinutes', 'PagesViewed', 'LeadStatus', 'EmailSent',
       'DeviceType', 'ReferralSource', 'FormSubmissions', 'Downloads',
       'CTR_ProductPage', 'ResponseTimeHours', 'FollowUpEmails',
       'SocialMediaEngagement', 'PaymentHistory', 'Conversion']

# Data preparation
df_full_train, df_validation = train_test_split(df_train, test_size = 0.2, random_state = 42)

## Remove duplicated rows
df_full_train.drop_duplicates(inplace = True)
df_full_train.reset_index(drop = True, inplace = True)

# ## Check null values
# df_full_train.isnull().sum()

# ## Drop unused columns
# df_full_train.drop(columns = ['LeadID'], inplace = True)

# ## Remove outliers
# numerical_columns = ['Age',
#  'TimeSpentMinutes',
#  'PagesViewed',
#  'EmailSent',
#  'FormSubmissions',
#  'Downloads',
#  'ResponseTimeHours',
#  'FollowUpEmails',
#  'SocialMediaEngagement']

# index_outlier = []
# for num in numerical_columns:
#     q99 = np.quantile(df_full_train[num], .99)
#     outlier = df_full_train[df_full_train[num] > q99].index.tolist()
#     index_outlier.extend(outlier)
# index_outlier = list(set(index_outlier))

# df_full_train_cleaned = df_full_train.drop(index = index_outlier,)
# df_full_train_cleaned.reset_index(drop = True, inplace = True)
# print("Data preparation step done !")

# # Model training

# ## Get selected features for model
# top_numerical_columns = ['PagesViewed', 'Age', 'EmailSent', 'TimeSpentMinutes', 'FollowUpEmails', 'SocialMediaEngagement', 'FormSubmissions', 'Downloads', 'ResponseTimeHours',] 
# top_categorical_columns = ['Location', 'LeadStatus',]

# df_full_train_cleaned_selected = pd.concat([df_full_train_cleaned[top_numerical_columns], df_full_train_cleaned[top_categorical_columns], df_full_train_cleaned[['Conversion']]], axis = 1)
# df_full_train_cleaned_selected.head()

# ## Get prepared dataset
# X_train_new = df_full_train_cleaned_selected.drop(columns = ['Conversion'])
# y_train_new = df_full_train_cleaned_selected['Conversion']

# ## Define preprocessing for numerical data
# numerical_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# ## Define preprocessing for categorical data
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(drop = 'first', handle_unknown='ignore'))
# ])

# ## Combine preprocessors in a ColumnTransformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, top_numerical_columns),
#         ('cat', categorical_transformer, top_categorical_columns)
#     ])

# ## Define the full pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])

# ## Fit pipeline
# pipeline.fit(X_train_new, y_train_new)

# ## Perform cross-validation
# cv_scores = cross_val_score(pipeline, X_train_new, y_train_new, cv=5, scoring='roc_auc')
# print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ## Define parameter grid
# param_grid = {
#     'classifier__n_estimators': [50, 100, 200],
#     'classifier__max_depth': [None, 10, 20],
#     'classifier__min_samples_split': [2, 5]
# }

# ## Run grid search to find best parameters
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid_search.fit(X_train_new, y_train_new)

# ## Best parameters and score
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best CV Score: {grid_search.best_score_:.4f}")

# ## Save pipeline to a file
# with open('model/best_pipeline.pkl', 'wb') as f:
#     joblib.dump(grid_search.best_estimator_, f)

# print("Model training step done !")   

# # Model evaluation    

## Load pipeline
with open('model/best_pipeline.pkl', 'rb') as f:
    loaded_pipeline = joblib.load(f)
    
## Evaluate model with validation dataset
X_validation = df_validation.drop(columns = ['Conversion'])
y_validation = df_validation['Conversion']

## Predict
y_validation_pred = loaded_pipeline.predict(X_validation)

## ROC_AUC for validation dataset
roc_auc_df_validation = roc_auc_score(y_validation, y_validation_pred)
print(f"ROC AUC with validation dataset: {roc_auc_df_validation:.4f}")

## Evaluate model with test dataset
X_test = df_test.drop(columns = ['Conversion'])
y_test = df_test['Conversion']

## Predict
y_test_pred = loaded_pipeline.predict(X_test)

## ROC_AUC for validation dataset
roc_auc_df_test = roc_auc_score(y_test, y_test_pred)
print(f"ROC AUC with test dataset: {roc_auc_df_test:.4f}")

print("Model evaluation step done !")   