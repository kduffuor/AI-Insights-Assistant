import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from dotenv import load_dotenv


# --- Load Environment Variables ---
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# --- Configuration ---
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# --- Page Setup ---
st.set_page_config(
    page_title="AI Customer Insights Assistant",
    page_icon="📈",
    layout="wide"
)

# --- UI Setup ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1.5rem;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_URL)
    return df

df = load_data()

# --- Derived Metrics ---
total_customers = len(df)
churn_rate = round(df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100, 2)
avg_tenure = round(df["tenure"].mean(), 2)
avg_monthly = round(df["MonthlyCharges"].mean(), 2)

# --- Build Dataset Summary for LLM Context ---
def build_summary(df):
    churn_count = df["Churn"].value_counts().to_dict()
    contract_dist = df["Contract"].value_counts().to_dict()
    churn_by_contract = df.groupby("Contract")["Churn"].apply(
        lambda x: round((x == "Yes").mean() * 100, 2)
    ).to_dict()
    churn_by_internet = df.groupby("InternetService")["Churn"].apply(
        lambda x: round((x == "Yes").mean() * 100, 2)
    ).to_dict()

    summary = f"""
    Dataset Overview:
    - Total customers: {total_customers}
    - Churn distribution: {churn_count}
    - Churn rate: {churn_rate}%
    - Average customer tenure: {avg_tenure} months
    - Average monthly charges: ${avg_monthly}
    - Contract types: {contract_dist}
    - Churn rate by contract type: {churn_by_contract}
    - Churn rate by internet service type: {churn_by_internet}
    """
    return summary

# --- Query HuggingFace LLM ---
def query_llm(user_question, dataset_summary):
    prompt = f"""You are a professional data analyst assistant specializing in customer churn analysis.

You have been given the following dataset summary:
{dataset_summary}

Answer the following business question clearly and concisely, as if presenting to a non-technical executive audience:

Question: {user_question}"""

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.5
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"API Error {response.status_code}: {response.text}"

# --- App Header ---
st.title("AI Customer Insights Assistant")
st.caption("Churn Analysis powered by Natural Language")

st.divider()

# --- Metrics Banner ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Customers", f"{total_customers:,}")
m2.metric("Churn Rate", f"{churn_rate}%")
m3.metric("Avg. Tenure", f"{avg_tenure} months")
m4.metric("Avg. Monthly Charges", f"${avg_monthly}")

st.divider()

# --- Row 1 Charts: Churn Distribution + Churn by Contract ---
chart1, chart2 = st.columns(2)

with chart1:
    # st.subheader("Overall Churn Distribution")
    st.markdown("**How balanced is churn across all customers?**")

    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Count"]
    fig1 = px.pie(
        churn_counts,
        names="Churn",
        values="Count",
        color_discrete_sequence=["#00C4B4", "#FF6B6B"],
        hole=0.4
    )
    fig1.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig1, use_container_width=True)
    st.info("Insight: Churn is significant enough to require segmentation analysis.")

with chart2:
    # st.subheader("Churn Rate by Contract Type")
    st.markdown("**Does contract type influence customer retention?**")

    churn_by_contract = df.groupby("Contract")["Churn"].apply(
        lambda x: round((x == "Yes").mean() * 100, 2)
    ).reset_index()
    churn_by_contract.columns = ["Contract", "Churn Rate (%)"]
    fig2 = px.bar(
        churn_by_contract,
        x="Contract",
        y="Churn Rate (%)",
        color="Contract",
        color_discrete_sequence=["#FF6B6B", "#FFD93D", "#00C4B4"],
        text="Churn Rate (%)"
    )
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    fig2.update_layout(showlegend=False, height=400, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.success("Conclusion: Month-to-month contracts drive the highest churn risk.")

st.divider()

# --- Row 2 Charts: Churn by Internet Service + Tenure Distribution ---
chart3, chart4 = st.columns(2)

with chart3:
    # st.subheader("Churn Rate by Internet Service Type")
    st.markdown("**Does service type affect churn behavior?**")

    churn_by_internet = df.groupby("InternetService")["Churn"].apply(
        lambda x: round((x == "Yes").mean() * 100, 2)
    ).reset_index()
    churn_by_internet.columns = ["Internet Service", "Churn Rate (%)"]
    fig3 = px.bar(
        churn_by_internet,
        x="Internet Service",
        y="Churn Rate (%)",
        color="Internet Service",
        color_discrete_sequence=["#FF6B6B", "#FFD93D", "#00C4B4"],
        text="Churn Rate (%)"
    )
    fig3.update_traces(texttemplate="%{text}%", textposition="outside")
    fig3.update_layout(showlegend=False, height=400, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig3, use_container_width=True)
    st.success("Conclusion: Fiber optic users show higher churn sensitivity.")

with chart4:
    # st.subheader("Tenure Distribution: Churned vs Retained")
    st.markdown("**At what stage of tenure do customers leave?**")

    tenure_df = df.groupby(["tenure", "Churn"]).size().reset_index(name="Count")
    fig4 = px.line(
        tenure_df,
        x="tenure",
        y="Count",
        color="Churn",
        color_discrete_map={"Yes": "#FF6B6B", "No": "#00C4B4"},
        labels={"tenure": "Tenure (Months)", "Count": "Number of Customers"}
    )
    fig4.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig4, use_container_width=True)
    st.warning("Insight: Most churn happens within the first 0–10 months.")

# st.divider()

# --- Dataset Summary Panel ---
with st.expander("Dataset Summary", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    with col_b:
        st.subheader("Basic Statistics")
        st.dataframe(
            df[["tenure", "MonthlyCharges", "SeniorCitizen"]].describe(),
            use_container_width=True
        )

# st.divider()

# --- AI Question Interface ---
st.subheader("Ask a Business Question")
st.write("Type a plain English question about the dataset and the AI will generate an insight.")

# --- Suggested Questions ---
st.caption("Suggested questions — click to use:")

suggested = [
    "Why are customers churning?",
    "Which customer segment is at highest risk of churn?",
    "How does internet service type affect churn?"
]

col_s1, col_s2, col_s3 = st.columns(3)

if col_s1.button(suggested[0], use_container_width=True):
    st.session_state["question"] = suggested[0]

if col_s2.button(suggested[1], use_container_width=True):
    st.session_state["question"] = suggested[1]

if col_s3.button(suggested[2], use_container_width=True):
    st.session_state["question"] = suggested[2]

user_question = st.text_input(
    label="Your Question",
    placeholder="e.g. Why are customers churning?",
    value=st.session_state.get("question", "")
)

if st.button("Generate Insight"):
    if not user_question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Analyzing data and generating insight..."):
            summary = build_summary(df)
            answer = query_llm(user_question, summary)

        st.subheader("AI Insight")
        st.write(answer)

st.divider()

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 13px;'>
        AI Customer Insights Assistant<br>
        Built with Streamlit • Python • Plotly • HuggingFace LLMs<br>
        <br>
        🔗 <a href='https://github.com/kduffuor' target='_blank'>GitHub</a> |
        <a href='https://linkedin.com/in/kduffuor' target='_blank'>LinkedIn</a><br>
        <br>
        © 2026 — All insights generated for educational purposes
    </div>
    """,
    unsafe_allow_html=True
)