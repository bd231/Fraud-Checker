import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

# 1. Load your data
data = pd.read_csv("creditcard.csv")

# 2. Keep only the user-friendly columns
df = data[["Time", "Amount"]]

# 3. Train an Isolation Forest
iso = IsolationForest(contamination=0.002, random_state=42)
iso.fit(df)

# 4. Save it
with open("iso_model.pkl", "wb") as f:
    pickle.dump(iso, f)
