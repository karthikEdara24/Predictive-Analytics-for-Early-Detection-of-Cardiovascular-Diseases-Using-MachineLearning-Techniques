import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load and preprocess data
data = pd.read_csv(r'D:\Junk\Massion\CardioVascularPrediction\app\model\heart (3).csv')
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
data['ChestPainType'] = data['ChestPainType'].map({'ASY':3, 'ATA':2, 'NAP':1, 'TA':0})
data['RestingECG'] = data['RestingECG'].map({'LVH':2, 'Normal':1, 'ST':0})
data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y':1, 'N':0})
data['ST_Slope'] = data['ST_Slope'].map({'Down':2, 'Flat':1, 'Up':0})

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Feature selection using RFE
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=8)
rfe.fit(X_train, y_train)

X_train = rfe.transform(X_train)
X_test = rfe.transform(X_test)

# Define models
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(probability=True, random_state=42)
lr = LogisticRegression(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Hyperparameter tuning (simplified)
best_rf = rf.fit(X_train, y_train)
best_xgb = xgb.fit(X_train, y_train)
best_svm = svm.fit(X_train, y_train)
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf),
    ('knn', knn),
    ('svm', best_svm),
    ('lr', lr),
    ('xgb', best_xgb)
], voting='soft')

voting_clf.fit(X_train, y_train)

# Predict on test set
y_pred = voting_clf.predict(X_test)

# Print metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (Weighted):", precision_score(y_test, y_pred, average='weighted'))
print("Recall (Weighted):", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score (Weighted):", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


y_proba = voting_clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'Voting Classifier (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# Save model and preprocessors
with open('app/model/model.pkl', 'wb') as f:
    pickle.dump({'model': voting_clf, 'rfe': rfe, 'scaler': scaler}, f)
