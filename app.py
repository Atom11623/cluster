import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("rfm_data.csv")

df = load_data()

# ---- Sidebar ----
st.sidebar.title("📊 Customer Segmentation")
st.sidebar.markdown("Built by *Ibrahim Ali*")
st.sidebar.markdown("Adjust the number of clusters and parameters below:")

# Slider for selecting number of clusters
n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=4, step=1)

# Scale the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Map the segments
segment_map = {
    0: 'Loyal Customers',
    1: 'At Risk',
    2: 'Potential Loyalists',
    3: 'Big Spenders',
    4: 'Churned',
    5: 'High Value',
    6: 'Low Frequency',
    7: 'Frequent Shoppers',
    8: 'Engaged',
    9: 'New Customers'
}

df['Segment'] = df['Cluster'].map(segment_map)

# ---- Header ----
st.title("🧠 Customer Segmentation Dashboard")
st.markdown("### Using RFM + KMeans Clustering")
st.markdown("Explore customer segments based on behavioral patterns and adjust the number of clusters interactively.")

# ---- PCA Visualization ----
st.subheader("🎯 PCA Cluster Visualization")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[['Recency', 'Frequency', 'Monetary']])
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

# Plot the PCA projection with segment colors
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Segment', palette='Set2', s=100, ax=ax, alpha=0.8)
plt.title(f'Customer Segments Visualization using PCA - {n_clusters} Clusters', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(title='Segment', loc='best')
plt.tight_layout()

# Show the plot
st.pyplot(fig)

# ---- Segment Statistics ----
st.subheader("📋 Segment Summary Stats")
seg_stats = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
st.dataframe(seg_stats)

# ---- Download Segmented Data ----
st.subheader("📥 Download Segmented Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")
