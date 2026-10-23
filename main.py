
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Step 1: Generate normal data
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=5, size=300)

# Step 2: Add unusual values (anomalies)
anomalies = np.array([10, 15, 85, 90, 100])
all_data = np.concatenate([normal_data, anomalies])

# Step 3: Create a DataFrame
df = pd.DataFrame({"Value": all_data})

# Step 4: Train the anomaly detection model
model = IsolationForest(contamination=0.02, random_state=42)
df["Prediction"] = model.fit_predict(df[["Value"]])

# Step 5: Label the results
df["Status"] = df["Prediction"].map({
    1: "Normal",
    -1: "Anomaly"
})

# Step 6: Print the results
print("ANOMALY DETECTION RESULTS")
print("=========================")
print(df["Status"].value_counts())
print("\nDetected anomalies:")
print(df[df["Status"] == "Anomaly"].to_string(index=False))

# Step 7: Draw a graph
normal = df[df["Status"] == "Normal"]
abnormal = df[df["Status"] == "Anomaly"]

plt.figure(figsize=(10, 5))
plt.scatter(normal.index, normal["Value"], label="Normal", alpha=0.7)
plt.scatter(abnormal.index, abnormal["Value"], label="Anomaly",
            marker="x", s=80)

plt.title("Anomaly Detection Using Isolation Forest")
plt.xlabel("Data Point")
plt.ylabel("Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Step 8: Save and display the graph
plt.savefig("anomaly_detection.png")
print("\nGraph saved as anomaly_detection.png")
plt.show()
