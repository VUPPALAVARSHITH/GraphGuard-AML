# GraphGuard – Explainable GNN-based AML Risk Analysis

GraphGuard is an **Anti-Money Laundering (AML) risk analysis system** that uses
**Graph Neural Networks (GNNs)** to identify and explain suspicious accounts
in large-scale financial transaction networks.

This project focuses on **account-level risk prioritization** and
**analyst-friendly explainability**, rather than only transaction-level prediction.

---

## 🔍 Problem Statement
Traditional AML systems rely heavily on rule-based methods, which:
- Fail to capture complex money flow patterns
- Generate a large number of false positives
- Do not scale well to large transaction networks

GraphGuard addresses these issues by modeling transactions as a **graph** and
learning suspicious patterns using **Graph Neural Networks**.

---

## 📊 Dataset
- **IBM AMLSim Dataset**
- Subset used: `LI-Small`
- ~6.9 million transactions
- ~3,500 laundering cases
- Extreme class imbalance (~0.05%)

> Due to size constraints, raw datasets are **not included** in this repository.

---

## 🧠 Methodology

### Graph Construction
- **Nodes**: Account + Bank combinations
- **Edges**: Financial transactions
- **Edge Features**: Transaction amounts, payment format
- **Labels**: Transaction-level laundering indicators

### Model Architecture
- Node Embeddings
- **GraphSAGE** layers for neighborhood aggregation
- **GAT** layer for attention-based refinement
- Edge-level MLP for transaction risk prediction

---

## 📈 Evaluation Strategy
- Static full-graph evaluation (no train-test split)
- Metrics used:
  - ROC-AUC
  - Precision@K
  - Recall@K
  - **Account-level Recall@K (primary AML metric)**

Account risk is computed using:
- Maximum transaction risk (alert flag)
- Top-K mean transaction risk (ranking score)

---

## 🖥️ Dashboard (Streamlit)

The Streamlit dashboard provides:

### Tab 1 – Overview
- Bank-wise filtering
- Alert-level filtering
- Risk distribution
- Top risky accounts

### Tab 2 – Account Explorer
- Search-based account selection
- Entity-level context
- Color-coded GNN risk explanation

### Tab 3 – Transaction Drill-down
- Top risky transactions
- Sender/receiver bank details
- On-demand transaction network visualization

---

## 🧪 Project Structure

GraphGuard-AML/
├── preprocessing/
│ ├── build_graph.py
│ └── generate_account_to_tx.py
├── model/
│ ├── model_architecture.py
│ ├── train_model.py
│ └── evaluate.py
├── dashboard/
│ ├── app.py
│ └── requirements.txt
├── data/
│ └── README.md
├── .gitignore
└── README.md


---

## 🚀 How to Run the Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py


Note: Preprocessed data files are required and are not included in the repo.

🔮 Future Work

Temporal Graph Neural Networks (TGN)

Streaming AML detection

Cross-bank laundering pattern analysis

👤 Author

Vuppala Varshith

📌 Disclaimer

This project is for academic and research purposes only.


---

