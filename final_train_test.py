import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.ensemble import VotingClassifier

# Load the dataset
merged_transaction_dataset = pd.read_csv("merged_transaction_dataset.csv")

# Preprocessing
merged_transaction_dataset.drop(columns=['merchant'], inplace=True)
merged_transaction_dataset['transaction_date_time'] = pd.to_datetime(merged_transaction_dataset['transaction_date_time'], format='%d-%m-%Y %H:%M')
merged_transaction_dataset = pd.get_dummies(merged_transaction_dataset, columns=['transaction_mode'])
label_encoder = LabelEncoder()
merged_transaction_dataset['label'] = label_encoder.fit_transform(merged_transaction_dataset['label'])
merged_transaction_dataset['transaction_year'] = merged_transaction_dataset['transaction_date_time'].dt.year
merged_transaction_dataset['transaction_month'] = merged_transaction_dataset['transaction_date_time'].dt.month
merged_transaction_dataset['transaction_day'] = merged_transaction_dataset['transaction_date_time'].dt.day
merged_transaction_dataset['transaction_hour'] = merged_transaction_dataset['transaction_date_time'].dt.hour
merged_transaction_dataset['transaction_minute'] = merged_transaction_dataset['transaction_date_time'].dt.minute
merged_transaction_dataset.drop(columns=['transaction_date_time'], inplace=True)

# Feature and label extraction
X = merged_transaction_dataset[['transaction_year', 'transaction_month', 'transaction_day', 'transaction_hour', 'transaction_minute', 'transferred_amount', 'amount_before_transaction', 'amount_after_transaction', 'transaction_mode_Online']]
y = merged_transaction_dataset['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling class imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
svm_clf = SVC(kernel='rbf')
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Initialize SVM classifier with best parameters
best_svm_clf = grid_search.best_estimator_

# Predict labels for the test set
y_pred_svm = best_svm_clf.predict(X_test_scaled)

# Evaluate the SVM classifier
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Classifier Accuracy:", accuracy_svm)
print("Classification Report for SVM Classifier:")
print(classification_report(y_test, y_pred_svm))

# Ensemble Learning
voting_clf = VotingClassifier([('svm', best_svm_clf)], voting='hard')
voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_test_scaled)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print("Voting Classifier Accuracy:", accuracy_voting)
print("Classification Report for Voting Classifier:")
print(classification_report(y_test, y_pred_voting))




# Save the trained SVM model
joblib.dump(best_svm_clf, 'svm_model.pkl')

# Save the Voting Classifier
joblib.dump(voting_clf, 'voting_classifier.pkl')
