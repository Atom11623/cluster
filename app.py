import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("rfm_data.csv")

df = load_data()

# Drop rows with missing CustomerID (you can't segment anonymous customers)
df.dropna(subset=['CustomerID'], inplace=True)

# ---- Check column names ----
st.write(df.columns)  # This will show you the column names

# ---- Sidebar ----
st.sidebar.title("📊 Customer Segmentation")
st.sidebar.markdown("Built by *Ibrahim Ali*")
# st.sidebar.image("assets/logo.png", use_container_width=True)  # Removed logo

# ---- Header ----
st.title("🧠 Customer Segmentation Dashboard")
st.markdown("### Using RFM + KMeans Clustering")
st.markdown("This dashboard helps you explore customer segments based on behavioral patterns.")

# ---- KPIs ----
# Ensure 'CustomerID' exists before using it
if 'CustomerID' in df.columns:
    total_customers = df['CustomerID'].nunique()
else:
    total_customers = "Column 'CustomerID' not found"
    
if 'Cluster' in df.columns:
    num_segments = df['Cluster'].nunique()
else:
    num_segments = "Column 'Cluster' not found"

avg_recency = df['Recency'].mean() if 'Recency' in df.columns else "N/A"
avg_frequency = df['Frequency'].mean() if 'Frequency' in df.columns else "N/A"
avg_monetary = df['Monetary'].mean() if 'Monetary' in df.columns else "N/A"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", total_customers)
col2.metric("Segments", num_segments)
col3.metric("Avg. Recency", f"{avg_recency:.0f}" if avg_recency != "N/A" else avg_recency)
col4.metric("Avg. Frequency", f"{avg_frequency:.0f}" if avg_frequency != "N/A" else avg_frequency)

# ---- Visualizations ----
st.subheader("📌 Segment Distribution")
if 'Cluster' in df.columns:
    seg_counts = df['Cluster'].value_counts().sort_index()
    st.bar_chart(seg_counts)

# ---- PCA Visualization ----
st.subheader("🎯 PCA Cluster Visualization")
if 'Recency' in df.columns and 'Frequency' in df.columns and 'Monetary' in df.columns:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[['Recency', 'Frequency', 'Monetary']])
    df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100, ax=ax)
    plt.title("Customer Segments (2D Projection)")
    st.pyplot(fig)

# ---- Segment Statistics ----
st.subheader("📋 Segment Summary Stats")
if 'Cluster' in df.columns:
    seg_stats = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    st.dataframe(seg_stats)

# ---- Country & Products Insights ----
st.subheader("🌍 Top Countries by Transactions")
if "Country" in df.columns:
    country_counts = df['Country'].value_counts().head(10)
    st.bar_chart(country_counts)

st.subheader("📦 Top Products (Optional)")
if "Description" in df.columns:
    product_counts = df['Description'].value_counts().head(10)
    st.write(product_counts)

# ---- Download Results ----
st.subheader("📥 Download Segmented Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")
