import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
merged_transaction_dataset = pd.read_csv("merged_transaction_dataset.csv")

# Preprocessing
# Drop irrelevant columns
merged_transaction_dataset.drop(columns=['merchant'], inplace=True)

# Extract features from 'transaction_date_time'
merged_transaction_dataset['transaction_date_time'] = pd.to_datetime(merged_transaction_dataset['transaction_date_time'], format='%d-%m-%Y %H:%M')

# Convert categorical variables to dummy variables
merged_transaction_dataset = pd.get_dummies(merged_transaction_dataset, columns=['transaction_mode'])

# Encode labels
label_encoder = LabelEncoder()
merged_transaction_dataset['label'] = label_encoder.fit_transform(merged_transaction_dataset['label'])

# Split features and labels
# Extract date and time components
merged_transaction_dataset['transaction_year'] = merged_transaction_dataset['transaction_date_time'].dt.year
merged_transaction_dataset['transaction_month'] = merged_transaction_dataset['transaction_date_time'].dt.month
merged_transaction_dataset['transaction_day'] = merged_transaction_dataset['transaction_date_time'].dt.day
merged_transaction_dataset['transaction_hour'] = merged_transaction_dataset['transaction_date_time'].dt.hour
merged_transaction_dataset['transaction_minute'] = merged_transaction_dataset['transaction_date_time'].dt.minute

# Drop the original datetime column
merged_transaction_dataset.drop(columns=['transaction_date_time'], inplace=True)

# Split features and labels
X = merged_transaction_dataset[['transaction_year', 'transaction_month', 'transaction_day', 'transaction_hour', 'transaction_minute', 'transferred_amount', 'amount_before_transaction', 'amount_after_transaction', 'transaction_mode_Online']]
y = merged_transaction_dataset['label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize base classifiers
svm_clf = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
gb_clf = GradientBoostingClassifier()
ada_clf = AdaBoostClassifier()
et_clf = ExtraTreesClassifier(n_estimators=100)

# Create the Voting Classifier with diverse base classifiers
voting_clf = VotingClassifier(estimators=[('svm', svm_clf), ('gradient_boosting', gb_clf), ('adaboost', ada_clf), ('extra_trees', et_clf)], voting='soft')

# Train the Voting Classifier
voting_clf.fit(X_train_scaled, y_train)

# Predict labels for the test set
y_pred_voting = voting_clf.predict(X_test_scaled)

# Evaluate the Voting Classifier
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print("Voting Classifier Accuracy:", accuracy_voting)

# Print classification report
print("Classification Report for Voting Classifier:")
print(classification_report(y_test, y_pred_voting))
