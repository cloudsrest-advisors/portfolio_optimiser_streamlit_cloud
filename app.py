import streamlit as st
import pandas as pd
import numpy as np
import psycopg
import plotly.express as px
import ast
import cvxpy as cp
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import requests


# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", initial_sidebar_state="expanded")


# -------------------------------
# Database
# -------------------------------
DB_CONFIG = {
    "dbname": st.secrets["DB_NAME"],
    "user": st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"],
    "host": st.secrets["DB_HOST"],
    "port": int(st.secrets.get("DB_PORT", 5432)),
    "sslmode":  "require",           # ← important for cloud hosting
}

THEME_API_URL = st.secrets["THEME_API_URL"]

@st.cache_data
def load_data():
    # psycopg v3 connection (DB-API compatible)
    with psycopg.connect(**DB_CONFIG) as conn:
        portfolio_df = pd.read_sql("SELECT * FROM sample_portfolio", conn)
        price_df     = pd.read_sql("SELECT * FROM price_data", conn)
    return portfolio_df, price_df

portfolio_df, price_df = load_data()
price_df["date"] = pd.to_datetime(price_df["date"])

# -------------------------------
# Returns & Covariance
# -------------------------------
price_pivot = (
    price_df.pivot(index="date", columns="id", values="close")
    .sort_index()
)
returns = price_pivot.pct_change().dropna()
cov_matrix = returns.cov()  # (ids x ids)


# -------------------------------
# Sidebar Preferences
# -------------------------------
st.sidebar.header("🔧 Optimization Preferences")
conviction_wt = st.sidebar.slider("Conviction Score Weight", 0.0, 1.0, 0.3)
value_wt = st.sidebar.slider("Value Score Weight", 0.0, 1.0, 0.3)
estimate_wt = st.sidebar.slider("Estimate Score Weight", 0.0, 1.0, 0.3)
narrative_wt = st.sidebar.slider("Narrative Diversity Weight", 0.0, 1.0, 0.1)

st.sidebar.header("🧠 AI Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose an AI model:",
    [
        "Claude 3 Sonnet (v1.0)",
        "Claude Sonnet 4",
        "Claude Opus 4",   # <-- make label match the key below
        "Titan Text G1 - Premier",
    ],
)


model_map = {
    "Claude 3 Sonnet (v1.0)": "anthropic.claude-3-sonnet-20240229-v1:0",   # this one can stay as raw model
    "Claude Sonnet 4":        "us.anthropic.claude-sonnet-4-20250514-v1:0", # <-- profile ID
    "Claude Opus 4":          "us.anthropic.claude-opus-4-20250514-v1:0",   # <-- profile ID (verify date)
    "Titan Text G1 - Premier":"amazon.titan-text-premier-v1:0"
    }

model_id = model_map[model_choice]


# -------------------------------
# Header / Universe
# -------------------------------
st.markdown("## 📈 **AI Portfolio Optimiser**")
st.markdown("<br><br>", unsafe_allow_html=True)

st.subheader("📋 Full Stock Universe")
universe_cols = [
    "name",
    "yahoo_ticker",
    "conviction_score",
    "value_score",
    "estimate_score",
    "investment_case",
]
existing_cols = [c for c in universe_cols if c in portfolio_df.columns]
st.dataframe(portfolio_df[existing_cols])

st.markdown("<br><br>", unsafe_allow_html=True)


# -------------------------------
# Embeddings & Scoring
# -------------------------------
portfolio_df = portfolio_df.set_index("id", drop=False)  # keep id for lookup

def parse_embedding(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return parsed if isinstance(parsed, list) else None
        except Exception:
            return None
    return None

portfolio_df["investment_case_embedding"] = portfolio_df.get("investment_case_embedding", None)
if "investment_case_embedding" in portfolio_df.columns:
    portfolio_df["investment_case_embedding"] = portfolio_df["investment_case_embedding"].apply(parse_embedding)
    # Keep only rows with embeddings
    portfolio_df = portfolio_df.dropna(subset=["investment_case_embedding"])

# Base factor score
base_scores = (
    conviction_wt * portfolio_df.get("conviction_score", pd.Series(0, index=portfolio_df.index)).fillna(0) +
    value_wt * portfolio_df.get("value_score", pd.Series(0, index=portfolio_df.index)).fillna(0) +
    estimate_wt * portfolio_df.get("estimate_score", pd.Series(0, index=portfolio_df.index)).fillna(0)
)

# Narrative Diversity (mean cosine distance to others)
try:
    embeddings = np.vstack(portfolio_df["investment_case_embedding"].tolist())
    pairwise_dist = cosine_distances(embeddings)
    diversity = pairwise_dist.mean(axis=1)
    # Scale 0–100
    dmin, dmax = diversity.min(), diversity.max()
    if dmax > dmin:
        diversity_scaled = 100 * (diversity - dmin) / (dmax - dmin)
    else:
        diversity_scaled = np.zeros_like(diversity)
    diversity_series = pd.Series(diversity_scaled, index=portfolio_df.index)
    scores = base_scores + narrative_wt * diversity_series
except Exception as e:
    st.warning(f"Narrative diversity calculation failed: {e}")
    diversity_series = None
    scores = base_scores


# -------------------------------
# Select Top N & Build Optimization
# -------------------------------
top_n = 20
top_ids = scores.sort_values(ascending=False).head(top_n).index

# Align to covariance (drop ids without returns)
top_ids = [i for i in top_ids if i in cov_matrix.columns]
if len(top_ids) < 2:
    st.error("Not enough instruments with both scores and return history to optimize.")
    st.stop()

score_series = scores.loc[top_ids]      # composite score per name
cov_sub = cov_matrix.loc[top_ids, top_ids]

mu = score_series.values                # keep the name 'mu' if you like, but it's a score
Sigma = cov_sub.values

# Optimization problem: maximize mu^T w - lambda * w^T Σ w
w = cp.Variable(len(top_ids))
risk_aversion = 0.1

objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))

min_weight = 0.005  # 0.5%
max_weight = 0.07   # 10%

constraints = [
    cp.sum(w) == 1,
    w >= min_weight,
    w <= max_weight,
]

problem = cp.Problem(objective, constraints)
opt_val = None
try:
    opt_val = problem.solve(solver=cp.ECOS)
except Exception:
    try:
        opt_val = problem.solve(solver=cp.SCS)
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()

if w.value is None:
    # Fall back to equal weights if solver didn't return a solution
    st.warning("Optimization returned no solution. Falling back to equal weights.")
    w_val = np.ones(len(top_ids)) / len(top_ids)
else:
    w_val = w.value

weights = pd.Series(w_val, index=top_ids)


# -------------------------------
# Optimized Portfolio Table & Chart
# -------------------------------
optimized_df = portfolio_df.loc[top_ids].copy()
optimized_df["Optimized Weight"] = weights
optimized_df["Total Score"] = scores.loc[top_ids]
if diversity_series is not None:
    optimized_df["Narrative Diversity"] = diversity_series.loc[top_ids]

optimized_df["Optimized Weight (%)"] = (optimized_df["Optimized Weight"] * 100).round(1).astype(str) + "%"

st.subheader("📈 Optimized Portfolio")
display_cols = ["name", "yahoo_ticker", "Optimized Weight (%)", "Total Score"]
if "Narrative Diversity" in optimized_df.columns:
    display_cols.append("Narrative Diversity")
st.dataframe(optimized_df.sort_values(by="Optimized Weight", ascending=False)[display_cols])

# Bar chart of weights
bar_chart_df = optimized_df.sort_values(by="Optimized Weight", ascending=False).copy()
bar_chart_df["Stock Label"] = bar_chart_df["name"] + " (" + bar_chart_df["yahoo_ticker"] + ")"

fig_bar = px.bar(
    bar_chart_df,
    x="Stock Label",
    y="Optimized Weight",
    title="🔍 Optimized Portfolio Weights",
    labels={"Optimized Weight": "Weight"},
)
fig_bar.update_layout(
    xaxis_title="Stock",
    yaxis_title="Weight",
    xaxis_tickangle=-45,
    yaxis_tickformat=".1%",
    yaxis=dict(range=[0, 0.10]),  # 0% to 10%
    height=500,
)
st.plotly_chart(fig_bar, use_container_width=True)


# -------------------------------
# Portfolio Summary Stats
# -------------------------------
portfolio_variance = float(weights.values.T @ Sigma @ weights.values)
weighted_score     = float(weights.values @ mu)
risk_term          = float(risk_aversion * portfolio_variance)
objective_value    = float(weighted_score - risk_term)
effective_n = float(1 / np.sum(weights.values ** 2))
top_stock = weights.idxmax()
top_weight = float(weights.max())
avg_diversity = None
if diversity_series is not None:
    avg_diversity = float(diversity_series.loc[top_ids].dot(weights))

st.subheader("📊 Portfolio Summary Stats")
portfolio_stats_html = f"""
<div style='font-size:40px;'>
    <p>📈 <strong>Weighted Score (μ·w):</strong> {weighted_score:.2f}</p>
    <p>⚖️ <strong>Risk Term (λ·wᵀΣw) with λ={risk_aversion}:</strong> {risk_term:.4f}</p>
    <p>🎯 <strong>Total Objective (Score − Risk):</strong> {objective_value:.2f}</p>
    <p>📉 <strong>Portfolio Variance (wᵀΣw):</strong> {portfolio_variance:.4f}</p>
    <p>📊 <strong>Effective Number of Stocks:</strong> {effective_n:.2f}</p>
    <p>🏆 <strong>Top Stock Holding:</strong> {portfolio_df.loc[top_stock, 'name']} ({top_weight*100:.1f}%)</p>
    {"<p>🧠 <strong>Average Narrative Diversity:</strong> {:.2f}</p>".format(avg_diversity) if avg_diversity is not None else ""}
</div>
"""

st.markdown(portfolio_stats_html, unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


# -------------------------------
# Narrative Clustering (PCA + KMeans)
# -------------------------------
st.subheader("🧠 Narrative Clustering")
try:
    valid_embeddings = portfolio_df["investment_case_embedding"].dropna()
    embeddings = np.vstack(valid_embeddings.tolist())
    filtered_df = portfolio_df.loc[valid_embeddings.index]

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    # KMeans in 2D
    kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto").fit(coords)
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Build vis df
    vis_df = pd.DataFrame(coords, columns=["x", "y"])
    vis_df["Company"] = filtered_df["name"].values
    vis_df["Ticker"] = filtered_df["yahoo_ticker"].values
    # Make cluster legend 1..5 and categorical
    vis_df["Cluster"] = (cluster_labels.astype(int) + 1).astype(str)

    fig = px.scatter(
        vis_df,
        x="x",
        y="y",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,  # distinct colors
        hover_data=["Company", "Ticker"],
        title="Narrative Clustering (5 Clusters)",
    )
    fig.update_traces(marker=dict(size=12))

    # Centroids (note: they are in PC coordinates, same space)
    fig.add_scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode="markers+text",
        marker=dict(size=18, color="black", symbol="x"),
        text=[f"C{i+1}" for i in range(len(centroids))],
        textposition="top center",
        name="Centroids",
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.info(f"Narrative visualization not available: {e}")


# -------------------------------
# Theme Detection (Lambda)
# -------------------------------
st.subheader("📝 Cluster Theme Summaries (AI Generated)")

def get_theme_summary(top_stocks: list[dict], model_id: str, timeout: int = 20) -> str:
    """
    Call your Lambda theme-detection endpoint and return a short summary string.
    """
    api_url = "https://m2hdhbj7rc.execute-api.us-east-1.amazonaws.com/theme-detection"
    payload = {"top_stocks": top_stocks, "model_id": model_id}

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Theme API request failed: {e}") from e

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"Theme API returned non-JSON: {resp.text[:200]}")

    summary = (data or {}).get("theme_summary") or ""
    return summary.strip() if summary else "No summary returned."


# Generate summaries per cluster
try:
    # Quick lookup table by name to avoid repeated boolean filters
    # (assumes 'name' exists in filtered_df from the PCA section)
    if "filtered_df" in locals():
        name_indexed = filtered_df.set_index("name", drop=False)

        # Sort clusters numerically even if stored as strings
        cluster_ids = sorted(vis_df["Cluster"].unique(), key=lambda x: int(x))

        for cluster_id in cluster_ids:
            cluster_df = vis_df[vis_df["Cluster"] == cluster_id]

            # Build payload for Lambda
            cluster_stock_data = []
            for _, row in cluster_df.iterrows():
                company = row["Company"]
                if company in name_indexed.index:
                    stock_row = name_indexed.loc[company]
                    cluster_stock_data.append({
                        "name": stock_row.get("name", "Unknown"),
                        "sector": stock_row.get("sector", "Unknown"),
                        "investment_case": stock_row.get("investment_case", ""),
                    })

            if not cluster_stock_data:
                st.warning(f"No stocks found for Cluster {cluster_id}.")
                continue

            # Optional: list the names for readability
            names_line = ", ".join([s["name"] for s in cluster_stock_data])
            st.caption(f"**Cluster {cluster_id} names:** {names_line}")

            with st.spinner(f"Generating theme for Cluster {cluster_id}…"):
                summary = get_theme_summary(cluster_stock_data, model_id)

            st.markdown(f"**Cluster {cluster_id} — Theme & Risks:** {summary}")
    else:
        st.info("No embeddings available to build cluster summaries.")
except Exception as e:
    st.info(f"Narrative theme summaries not available: {e}")