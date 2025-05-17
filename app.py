import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle

# Step 1: Load data
data = pd.read_csv("creditcard.csv")

# Step 2: Prepare features and labels
X = data.drop("Class", axis=1)
y = data["Class"]

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate
print(classification_report(y_test, y_pred))

# Step 7: Save model and features
model_bundle = {
    "model": model,
    "features": list(X.columns)
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

