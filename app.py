import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# -------------------------------
# PAGE TITLE
# -------------------------------

st.title("Customer Segmentation using K-Means")

st.write(
    "This application groups customers based on their "
    "Annual Income and Spending Score using K-Means clustering."
)


# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("Mall_Customers.csv")


# -------------------------------
# DISPLAY DATASET
# -------------------------------

st.subheader("Customer Dataset")

st.dataframe(df)

st.write("Number of customers:", len(df))


# -------------------------------
# SELECT FEATURES
# -------------------------------

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


# -------------------------------
# STANDARDIZE DATA
# -------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -------------------------------
# ELBOW METHOD
# -------------------------------

st.subheader("Elbow Method")

inertia = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)


fig, ax = plt.subplots()

ax.plot(
    range(1, 11),
    inertia,
    marker="o"
)

ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")

st.pyplot(fig)


# -------------------------------
# SELECT NUMBER OF CLUSTERS
# -------------------------------

st.subheader("Choose Number of Clusters")

k = st.slider(
    "Select K",
    min_value=2,
    max_value=10,
    value=5
)


# -------------------------------
# K-MEANS CLUSTERING
# -------------------------------

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)


# -------------------------------
# SILHOUETTE SCORE
# -------------------------------

score = silhouette_score(
    X_scaled,
    df["Cluster"]
)


st.subheader("Model Evaluation")

st.metric(
    "Silhouette Score",
    round(score, 3)
)


# -------------------------------
# CUSTOMER SEGMENTS
# -------------------------------

st.subheader("Customer Segments")

st.dataframe(df)


# -------------------------------
# CLUSTER VISUALIZATION
# -------------------------------

st.subheader("Customer Segmentation")

fig, ax = plt.subplots()

ax.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=60
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segments")

st.pyplot(fig)


# -------------------------------
# CLUSTER SUMMARY
# -------------------------------

st.subheader("Cluster Summary")

summary = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()

st.dataframe(summary.round(2))


# -------------------------------
# DOWNLOAD RESULTS
# -------------------------------

st.subheader("Download Results")

csv = df.to_csv(index=False)

st.download_button(
    label="Download Customer Segments CSV",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)