import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="GraphGuard – AML Risk Dashboard",
    layout="wide"
)

DATA_PATH = "./data/dashboard_account_risk_demo.csv"
NODE_MAP_PATH = "./data/node_mapping.csv"
EDGE_INDEX_PATH = "./data/edge_index.npy"
TX_RISK_PATH = "./data/transaction_risk_scores.npy"
ACCOUNT_TX_PATH = "./data/account_to_tx.npy"

# =============================
# SESSION STATE
# =============================
if "selected_account" not in st.session_state:
    st.session_state.selected_account = None

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH)

account_to_bank = dict(
    zip(df["Account Number"].astype(str), df["Bank Name"])
)

@st.cache_data
def load_node_map():
    nm = pd.read_csv(NODE_MAP_PATH, dtype=str)
    nm.columns = ["node"]
    nm["node_id"] = nm.index
    nm[["Account Number", "Bank ID"]] = nm["node"].str.split("_", expand=True)
    return nm

@st.cache_data
def load_edge_index():
    return np.load(EDGE_INDEX_PATH)

@st.cache_data
def load_tx_scores():
    return np.load(TX_RISK_PATH)

@st.cache_data
def load_account_to_tx():
    return np.load(ACCOUNT_TX_PATH, allow_pickle=True).item()

node_map = load_node_map()
edge_index = load_edge_index()
tx_scores = load_tx_scores()
account_to_tx = load_account_to_tx()

src_nodes = edge_index[0]
dst_nodes = edge_index[1]

# =============================
# HEADER
# =============================
st.title("🔍 GraphGuard – AML Risk Analysis Dashboard")
st.caption("GNN-based Account Risk Detection & Explainability")

# =============================
# TABS
# =============================
tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "👤 Account Explorer",
    "💸 Transaction Drill-down"
])

# =====================================================
# TAB 1: OVERVIEW + FILTERS
# =====================================================
with tab1:
    st.subheader("System Overview")

    colf1, colf2 = st.columns(2)

    with colf1:
        banks = ["All Banks"] + sorted(df["Bank Name"].unique())
        selected_bank = st.selectbox("Filter by Bank", banks)

    with colf2:
        selected_alerts = st.multiselect(
            "Filter by Alert Level",
            ["LOW", "MEDIUM", "HIGH"],
            default=["HIGH"]
        )

    filtered_df = df.copy()

    if selected_bank != "All Banks":
        filtered_df = filtered_df[filtered_df["Bank Name"] == selected_bank]

    if selected_alerts:
        filtered_df = filtered_df[
            filtered_df["alert_level"].isin(selected_alerts)
        ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Accounts", len(filtered_df))
    c2.metric("High Risk", (filtered_df["alert_level"] == "HIGH").sum())
    c3.metric(
        "High Risk %",
        f"{(filtered_df['alert_level'] == 'HIGH').mean() * 100:.2f}%"
        if len(filtered_df) else "0%"
    )

    st.divider()
    st.dataframe(
        filtered_df[
            ["rank", "Account Number", "Bank Name", "risk_score", "alert_level"]
        ].sort_values("rank").head(20),
        use_container_width=True
    )

# =====================================================
# TAB 2: ACCOUNT EXPLORER (TOGGLE + EXPLANATION)
# =====================================================
with tab2:
    st.subheader("Account Explorer")

    if filtered_df.empty:
        st.info("No accounts available with current filters.")
        st.stop()

    show_all = st.checkbox(
        "Show all accounts (may be slow)",
        value=False
    )

    if show_all:
        search_df = filtered_df
    else:
        search_df = filtered_df.sort_values(
            "risk_score", ascending=False
        ).head(500)

    search_df["label"] = (
        search_df["Account Number"].astype(str)
        + " | "
        + search_df["Bank Name"]
        + " | "
        + search_df["Entity Name"].fillna("Unknown")
    )

    selected_label = st.selectbox(
        "Search Account (Account / Bank / Entity)",
        search_df["label"].tolist()
    )

    acc = selected_label.split(" | ")[0]
    st.session_state.selected_account = acc

    acc_row = search_df[
        search_df["Account Number"].astype(str) == acc
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Bank", acc_row["Bank Name"])
    c2.metric("Risk Score", f"{acc_row['risk_score']:.4f}")
    c3.metric("Alert Level", acc_row["alert_level"])

    st.markdown("### Entity Information")
    st.write("**Entity Name:**", acc_row["Entity Name"])

    # =============================
    # COLOR-CODED RISK EXPLANATION
    # =============================
    st.divider()
    st.markdown("### 🔍 Risk Explanation")

    if acc_row["alert_level"] == "HIGH":
        st.markdown(
            """
            <div style="
                background-color:#ffcccc;
                padding:15px;
                border-radius:8px;
                border-left:6px solid #ff0000;
                color:#000000;
                line-height:1.6;
                font-size:15px;
            ">
            <b>HIGH RISK (GNN Alert)</b><br><br>
            This account is flagged as <b>HIGH RISK</b> by the 
            <b>Graph Neural Network</b> due to repeated involvement in 
            high-risk transactions and suspicious network interactions.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif acc_row["alert_level"] == "MEDIUM":
        st.markdown(
            """
            <div style="
                background-color:#ffe5b4;
                padding:15px;
                border-radius:8px;
                border-left:6px solid #ff9800;
                color:#000000;
                line-height:1.6;
                font-size:15px;
            ">
            <b>MODERATE RISK (GNN Alert)</b><br><br>
            This account exhibits <b>moderate-risk patterns</b> learned by the 
            <b>Graph Neural Network</b>, indicating potentially unusual 
            transaction behavior that may require monitoring.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div style="
                background-color:#d4edda;
                padding:15px;
                border-radius:8px;
                border-left:6px solid #28a745;
                color:#000000;
                line-height:1.6;
                font-size:15px;
            ">
            <b>LOW RISK (GNN Assessment)</b><br><br>
            This account is currently assessed as <b>LOW RISK</b> by the 
            <b>Graph Neural Network</b>, with no strong suspicious patterns detected.
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# TAB 3: TRANSACTION DRILL-DOWN + GRAPH
# =====================================================
with tab3:
    st.subheader("Transaction Drill-down")

    acc = st.session_state.selected_account
    if acc is None:
        st.info("Please select an account in the Account Explorer tab.")
        st.stop()

    acc_node_id = int(
        node_map[node_map["Account Number"] == acc].iloc[0]["node_id"]
    )

    tx_ids = account_to_tx.get(acc_node_id, [])
    if not tx_ids:
        st.info("No transactions found for this account.")
        st.stop()

    tx_ids = np.array(tx_ids)
    top_tx = tx_ids[np.argsort(tx_scores[tx_ids])[-5:]][::-1]

    tx_table = []
    for tx in top_tx:
        src_acc = node_map.loc[src_nodes[tx], "Account Number"]
        dst_acc = node_map.loc[dst_nodes[tx], "Account Number"]

        tx_table.append({
            "Transaction ID": tx,
            "From Account": src_acc,
            "From Bank": account_to_bank.get(src_acc, "Unknown"),
            "To Account": dst_acc,
            "To Bank": account_to_bank.get(dst_acc, "Unknown"),
            "Risk Score": round(tx_scores[tx], 4)
        })

    tx_df = pd.DataFrame(tx_table)
    st.dataframe(tx_df, use_container_width=True)

    # =============================
    # GRAPH (ON-DEMAND)
    # =============================
    st.divider()
    show_graph = st.checkbox(
        "Show Transaction Network (Explainability Graph)",
        value=False
    )

    if show_graph:
        edges = list(zip(tx_df["From Account"], tx_df["To Account"]))

        if edges:
            G = nx.Graph()
            G.add_edges_from(edges)

            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            pos = nx.spring_layout(G, seed=42, k=0.8)

            nx.draw_networkx_nodes(
                G, pos,
                node_size=600,
                node_color="lightcoral",
                ax=ax
            )
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[acc],
                node_size=900,
                node_color="red",
                ax=ax
            )
            nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

            ax.set_title("Local Transaction Network (Top-Risk Interactions)", fontsize=10)
            ax.axis("off")

            st.pyplot(fig, use_container_width=False)
        else:
            st.info("Not enough data to visualize the transaction network.")
