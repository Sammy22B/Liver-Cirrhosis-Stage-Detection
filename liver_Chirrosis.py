
#%%
import pandas as pd
import numpy as np
 # Load dataset
liver_df = pd.read_csv('D:\zoology download\Projects-20240722T093004Z-001\Projects\liver_cirrhosis_stage\liver_cirrhosis_stage\liver_cirrhosis.csv')
print(liver_df.shape)
liver_df.head()
liver_df.info()
 # Print number of rows and columns
 # Display first few records
 # Data types and non-null counts
 #%%
print(liver_df.isna().sum())
# %%
liver_df.describe(include='all').T

# %%
print(liver_df['Stage'].value_counts())
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
cat_col= liver_df.select_dtypes(include='object').columns
#convert the categorical values into numerical values
# %%
for col in cat_col:
    print(col)
    print((liver_df[col].unique()), list(range(liver_df[col].nunique())))
    liver_df[col].replace((liver_df[col].unique()), range(liver_df[col].nunique()), inplace=True)
    print('*'*90)
    print()

# %%
# Drop irrelevant features and separate target
X = liver_df.drop(['N_Days','Status','Stage'], axis=1)
y = liver_df['Stage']
# %%
#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#%%
print(liver_df.columns)
#%%
#Scale numerical features
num_cols = ['Age','Bilirubin','Cholesterol','Albumin','Copper',
 'Alk_Phos','SGOT','Tryglicerides','Platelets','Prothrombin']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
# %%
#Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
# Stage distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Class Distribution of Cirrhosis Stages")
plt.xlabel("Stage")
plt.ylabel("Count")
plt.show()
# %%
# Histograms of some features
X_train[num_cols].hist(bins=30, figsize=(12,10))
plt.tight_layout()
plt.show()
# %%
# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(X_train[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
# %%
#Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
# Train logistic regression
lr = LogisticRegression(max_iter=1000, multi_class='multinomial')
lr.fit(X_train, y_train)
# Predictions and evaluation
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr, digits=4))
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# %%
print(f'Logistic Regression Score: {accuracy_score(y_test, y_pred_lr)}')
# %%
from sklearn.ensemble import RandomForestClassifier
# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# Predictions and evaluation
y_pred_rf = rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("Random Forest Classifier's Accuracy: ", accuracy_score(y_test, y_pred_rf))
# %%
import numpy as np
# Feature importance from Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train.columns
# Plot top 10 feature importances
plt.figure(figsize=(6,6))
plt.title("Feature Importances (Random Forest)")
sns.barplot(y=feature_names[indices][:10], x=importances[indices][:10],
 orient='h')
plt.xlabel("Importance")
plt.show() # The bar chart shows which features the Random Forest found most useful for classification.
# %%
from sklearn.svm import SVC
# Train SVM with RBF kernel
svm = SVC(kernel='rbf', probability=True) # probability=True to compute ROC later
svm.fit(X_train, y_train)
# Predictions and evaluation
y_pred_svm = svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, digits=4))
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("SVM's Accuracy: ", accuracy_score(y_test, y_pred_svm))
# %%
#Now we refine models via hyperparameter tunning
from sklearn.model_selection import GridSearchCV
# Example: hyperparameter tuning for Random Forest
param_grid_rf = {
 'n_estimators': [50, 100, 200],
 'max_depth': [5, 10, None],
 'min_samples_split': [2, 5]
 }
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)
print("Best Random Forest parameters:", grid_rf.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(grid_rf.best_score_))
best_rf = grid_rf.best_estimator_
# After finding the best parameters, we evaluate the tuned model on the test set:
y_pred_best_rf = best_rf.predict(X_test)
print("Tuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best_rf, digits=4))
# %%
#Model evaluation and comparison
models = {
 'Logistic Regression': y_pred_lr,
 'Random Forest (baseline)': y_pred_rf,
 'Random Forest (tuned)': y_pred_best_rf,
 'SVM': y_pred_svm,
 }
for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
# %%
#Visualize Performance with ROC Curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# Binarize the output labels for ROC
classes = [1, 2, 3]
y_test_bin = label_binarize(y_test, classes=classes)
# Use probabilities from one model, e.g., Random Forest tuned
y_score = best_rf.predict_proba(X_test)
fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
 # Plot ROC curves for each class
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color,
    label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.title('One-vs-Rest ROC Curves (Random Forest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
# %%
