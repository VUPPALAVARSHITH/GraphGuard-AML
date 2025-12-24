from google.colab import drive
drive.mount('/content/drive')

RAW_PATH = "/content/drive/MyDrive/GraphGuard/raw_data/LI-Small_Trans.csv"
PROCESSED_DIR = "/content/drive/MyDrive/GraphGuard/processed_data"

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv(RAW_PATH)

print("Raw shape:", df.shape)
df.head()

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"])

# Sort events by time (CRITICAL for TGN)
df = df.sort_values("Timestamp").reset_index(drop=True)

df["From_Node"] = df["Account"].astype(str) + "_" + df["From Bank"].astype(str)
df["To_Node"]   = df["Account.1"].astype(str) + "_" + df["To Bank"].astype(str)

node_encoder = LabelEncoder()
all_nodes = pd.concat([df["From_Node"], df["To_Node"]]).unique()
node_encoder.fit(all_nodes)

df["src"] = node_encoder.transform(df["From_Node"])
df["dst"] = node_encoder.transform(df["To_Node"])

edge_index = np.vstack([
    df["src"].values,
    df["dst"].values
])

# Monetary values
df["amount_paid"] = df["Amount Paid"].fillna(0.0)
df["amount_received"] = df["Amount Received"].fillna(0.0)

# Payment format encoding
pay_encoder = LabelEncoder()
df["payment_format_id"] = pay_encoder.fit_transform(
    df["Payment Format"].fillna("UNK")
)

edge_features = np.vstack([
    df["amount_paid"].values,
    df["amount_received"].values,
    df["payment_format_id"].values
]).T

timestamps = df["Timestamp"].astype(np.int64) // 10**9

labels = df["Is Laundering"].fillna(0).astype(int).values

os.makedirs(PROCESSED_DIR, exist_ok=True)

np.save(f"{PROCESSED_DIR}/edge_index.npy", edge_index)
np.save(f"{PROCESSED_DIR}/edge_features.npy", edge_features)
np.save(f"{PROCESSED_DIR}/timestamps.npy", timestamps)
np.save(f"{PROCESSED_DIR}/labels.npy", labels)

pd.DataFrame({"node": node_encoder.classes_}) \
  .to_csv(f"{PROCESSED_DIR}/node_mapping.csv", index=False)

print("✅ GAT + TGN preprocessing completed")
print("Nodes:", len(node_encoder.classes_))
print("Edges:", edge_index.shape[1])
print("Laundering cases:", labels.sum())