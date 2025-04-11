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

# ---- Sidebar ----
st.sidebar.title("📊 Customer Segmentation")
st.sidebar.markdown("Built by *Ibrahim Ali*")
# st.sidebar.image("assets/logo.png", use_container_width=True)  # Removed logo

# ---- Header ----
st.title("🧠 Customer Segmentation Dashboard")
st.markdown("### Using RFM + KMeans Clustering")
st.markdown("This dashboard helps you explore customer segments based on behavioral patterns.")

# ---- KPIs ----
num_segments = df['Cluster'].nunique()
avg_recency = df['Recency'].mean()
avg_frequency = df['Frequency'].mean()
avg_monetary = df['Monetary'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Segments", num_segments)
col2.metric("Avg. Recency", f"{avg_recency:.0f}")
col3.metric("Avg. Frequency", f"{avg_frequency:.0f}")
col4.metric("Avg. Monetary", f"{avg_monetary:.0f}")

# ---- Visualizations ----
st.subheader("📌 Segment Distribution")
seg_counts = df['Cluster'].value_counts().sort_index()
st.bar_chart(seg_counts)

# ---- PCA Visualization ----
st.subheader("🎯 PCA Cluster Visualization")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[['Recency', 'Frequency', 'Monetary']])
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100, ax=ax)
plt.title("Customer Segments (2D Projection)")
st.pyplot(fig)

# ---- Segment Statistics ----
st.subheader("📋 Segment Summary Stats")
seg_stats = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
st.dataframe(seg_stats)

# ---- Download Results ----
st.subheader("📥 Download Segmented Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")
