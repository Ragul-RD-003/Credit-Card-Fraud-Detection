# Guarding Transactions with an AI-Powered Credit System

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Load Dataset
df = pd.read_csv("ai_powered_credit_guard_dataset.csv")

# Display the first few rows of the dataset
print("Dataset loaded successfully.")
print(df.head())

# 3. Check for non-numeric columns
print("\nData types:")
print(df.dtypes)

# 4. Preprocess Data
# Drop 'Time' column as it's unlikely to be helpful for prediction
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# 5. Normalize only numeric columns (features V1 to V28, Amount)
scaler = StandardScaler()
X_scaled = X.copy()  # Make a copy of X to scale the data

# Normalize all columns except 'Class' and 'Time' (already excluded)
X_scaled = scaler.fit_transform(X_scaled)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 6. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\nData split complete. Training and testing sets prepared.")

# 7. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training complete.")

# 8. Evaluate the Model
y_pred = model.predict(X_test)

# Print confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. Example Prediction
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)

print("\nSample Transaction Prediction: ", "Fraud" if prediction[0] == 1 else "Legit")

# 10. Conclusion
print("\nAI-powered fraud detection system completed successfully.")
