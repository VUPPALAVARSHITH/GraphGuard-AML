import streamlit as st
import pandas as pd
import os


# =====================================================
# DEPLOYMENT MODE
# =====================================================
# True  → Public Streamlit demo (NO .npy, NO graphs)
# False → Local research mode (heavy graph artifacts)
PUBLIC_DEMO_MODE = True

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="GraphGuard – AML Risk Dashboard",
    layout="wide"
)

# =====================================================
# PATHS (RELATIVE TO dashboard/app.py)
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dashboard_account_risk_demo.csv")

st.write("BASE_DIR:", BASE_DIR)
st.write("Looking for file at:", DATA_PATH)
st.write("File exists:", os.path.exists(DATA_PATH))

# =====================================================
# SAFE DATA LOADING
# =====================================================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ Dataset not found at resolved path")
        st.stop()
    return pd.read_csv(DATA_PATH)


df = load_data()

# =====================================================
# SESSION STATE
# =====================================================
if "selected_account" not in st.session_state:
    st.session_state.selected_account = None

# =====================================================
# HEADER
# =====================================================
st.title("🔍 GraphGuard – AML Risk Analysis Dashboard")
st.caption(
    "Explainable GNN-based Account Risk Prioritization "
    "(Public Demonstration Dashboard)"
)

# =====================================================
# TABS
# =====================================================
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
        banks = ["All Banks"] + sorted(df["Bank Name"].dropna().unique())
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
    c2.metric(
        "High Risk Accounts",
        (filtered_df["alert_level"] == "HIGH").sum()
    )
    c3.metric(
        "High Risk %",
        f"{(filtered_df['alert_level'] == 'HIGH').mean() * 100:.2f}%"
        if len(filtered_df) else "0%"
    )

    st.divider()

    st.subheader("Top Risky Accounts")
    st.dataframe(
        filtered_df[
            ["rank", "Account Number", "Bank Name", "risk_score", "alert_level"]
        ]
        .sort_values("rank")
        .head(20),
        width="stretch"
    )

# =====================================================
# TAB 2: ACCOUNT EXPLORER + EXPLANATION
# =====================================================
with tab2:
    st.subheader("Account Explorer")

    if filtered_df.empty:
        st.info("No accounts available with the selected filters.")
        st.stop()

    show_all = st.checkbox(
        "Show all accounts (may be slower)",
        value=False
    )

    if show_all:
        search_df = filtered_df
    else:
        search_df = filtered_df.sort_values(
            "risk_score", ascending=False
        ).head(500)

    search_df = search_df.copy()
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

    # =================================================
    # COLOR-CODED RISK EXPLANATION
    # =================================================
    st.divider()
    st.markdown("### 🔍 Risk Explanation")

    if acc_row["alert_level"] == "HIGH":
        bg, border, title = "#ffcccc", "#ff0000", "HIGH RISK (GNN Alert)"
        text = (
            "This account is flagged as HIGH RISK by the Graph Neural Network "
            "due to repeated involvement in suspicious transaction patterns "
            "and risky network interactions."
        )
    elif acc_row["alert_level"] == "MEDIUM":
        bg, border, title = "#ffe5b4", "#ff9800", "MODERATE RISK (GNN Alert)"
        text = (
            "This account shows moderately suspicious behavior learned by "
            "the GNN model and may require monitoring or further review."
        )
    else:
        bg, border, title = "#d4edda", "#28a745", "LOW RISK (GNN Assessment)"
        text = (
            "This account does not currently exhibit strong laundering "
            "patterns based on the GNN’s learned representations."
        )

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            padding:18px;
            border-radius:8px;
            border-left:6px solid {border};
            color:#000000;
            line-height:1.6;
            font-size:15px;
        ">
        <b>{title}</b><br><br>
        {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# TAB 3: TRANSACTION DRILL-DOWN (EXPLANATION ONLY)
# =====================================================
with tab3:
    st.subheader("Transaction Drill-down")

    st.markdown(
        """
        <div style="
            background-color:#f0f4f8;
            padding:25px;
            border-radius:10px;
            border-left:8px solid #1f77b4;
            color:#000000;
            line-height:1.8;
            font-size:16px;
        ">
        <b>ℹ️ Why is transaction-level graph explainability not shown here?</b><br><br>

        Transaction-level graph explainability requires access to the full
        transaction network, learned node embeddings, and large graph artifacts,
        which are computationally intensive to load and analyze in an online
        environment.<br><br>

        In <b>GraphGuard</b>, such fine-grained explainability is available in the
        <b>local research environment</b>, where the complete transaction graph is
        stored and used for model development, validation, and analysis.<br><br>

        The <b>publicly deployed dashboard</b> focuses on
        <b>account-level risk prioritization</b> using
        <b>precomputed GNN risk scores</b>. This design reflects real-world AML
        analyst workflows, where scalability, responsiveness, and prioritization
        are critical.<br><br>

        This separation ensures <b>reliability</b>, <b>fast interaction</b>, and
        <b>practical usability</b>, while preserving full explainability in the
        research setting.
        </div>
        """,
        unsafe_allow_html=True
    )
