import pandas as pd

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Get the average values for each column
means = data.mean()

# Save only V1 to V28
v_means = means[[f"V{i}" for i in range(1, 29)]]

# Save to a CSV so you can copy/paste later
v_means.to_csv("v_feature_means.csv")
