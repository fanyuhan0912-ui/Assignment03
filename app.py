# BEGIN AI-assisted code
# Tool: ChatGPT
# Prompt:
# "Create a complete Streamlit app for exploring similarities between
# Vancouver neighbourhoods based on business-type composition."

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Vancouver Business Similarity Explorer",
    page_icon="🏙️",
    layout="wide"
)


# ======================================================
# Load data
# ======================================================

@st.cache_data
def load_data():
    """
    Load the cleaned Vancouver business licence dataset.
    """
    data = pd.read_csv("clean_business_licences.csv")

    required_columns = [
        "licencenumber",
        "businesstype",
        "localarea",
        "latitude",
        "longitude",
        "numberofemployees",
        "feepaid"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return data


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        """
        The file `clean_business_licences.csv` was not found.

        Make sure it is located in the same folder as `app.py`.
        """
    )
    st.stop()

except ValueError as error:
    st.error(str(error))
    st.stop()


# ======================================================
# App title and description
# ======================================================

st.title("Vancouver Business Similarity Explorer")

st.write(
    """
    This application groups Vancouver neighbourhoods according to their
    business-type composition. Neighbourhoods with similar percentages of
    business categories are assigned to the same cluster.
    """
)


# ======================================================
# Dataset overview
# ======================================================

st.header("Dataset Overview")

overview_col1, overview_col2, overview_col3 = st.columns(3)

overview_col1.metric(
    "Business Licences",
    f"{len(df):,}"
)

overview_col2.metric(
    "Neighbourhoods",
    df["localarea"].nunique()
)

overview_col3.metric(
    "Business Types",
    df["businesstype"].nunique()
)

st.subheader("Sample Records")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ======================================================
# Neighbourhood business composition
# ======================================================

st.header("Neighbourhood Business Composition")

# Count businesses in each neighbourhood
area_counts = df["localarea"].value_counts()

# Remove areas with very few businesses
minimum_businesses = 100

valid_areas = area_counts[
    area_counts >= minimum_businesses
].index

area_df = df[
    df["localarea"].isin(valid_areas)
].copy()

# Create neighbourhood × business-type percentage matrix
composition = pd.crosstab(
    area_df["localarea"],
    area_df["businesstype"],
    normalize="index"
) * 100

st.write(
    f"""
    Each row represents one neighbourhood, and each column represents the
    percentage of businesses belonging to a particular business type.

    Neighbourhoods with fewer than **{minimum_businesses} businesses**
    were excluded to reduce unstable percentages caused by very small
    sample sizes.
    """
)

st.dataframe(
    composition.round(2),
    use_container_width=True
)


# ======================================================
# Clustering controls
# ======================================================

st.sidebar.header("Clustering Controls")

maximum_k = min(8, len(composition) - 1)

selected_k = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=maximum_k,
    value=min(4, maximum_k),
    step=1
)

st.sidebar.write(
    """
    Move the slider to re-run K-means and update the
    visualizations.
    """
)


# ======================================================
# K-means clustering
# ======================================================

st.header("Neighbourhood Clustering")

area_kmeans = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)

area_clusters = area_kmeans.fit_predict(composition)

# Keep cluster labels separate from the feature matrix
area_results = composition.copy()
area_results["Cluster"] = area_clusters

st.write(
    f"""
    K-means was applied to the neighbourhood business-composition
    matrix using **K = {selected_k}**.
    """
)

st.dataframe(
    area_results.round(2),
    use_container_width=True
)


# ======================================================
# PCA visualization
# ======================================================

st.header("PCA View of Neighbourhood Similarity")

area_pca = PCA(
    n_components=2,
    random_state=42
)

pca_coordinates = area_pca.fit_transform(composition)

pca_df = pd.DataFrame({
    "Neighbourhood": composition.index,
    "PC1": pca_coordinates[:, 0],
    "PC2": pca_coordinates[:, 1],
    "Cluster": area_clusters.astype(str),
    "Business Count": area_counts.loc[
        composition.index
    ].values
})

pca_figure = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    size="Business Count",
    hover_name="Neighbourhood",
    hover_data={
        "Business Count": True,
        "PC1": ":.3f",
        "PC2": ":.3f"
    },
    title=(
        "Neighbourhood Business Similarity "
        f"— K = {selected_k}"
    )
)

pca_figure.update_traces(
    marker={
        "line": {
            "width": 1
        }
    }
)

st.plotly_chart(
    pca_figure,
    use_container_width=True,
    key="pca_chart"
)

explained_variance = (
    area_pca.explained_variance_ratio_.sum()
)

st.caption(
    f"The first two principal components explain "
    f"{explained_variance:.1%} of the variance in "
    f"neighbourhood business composition."
)


# ======================================================
# Cluster membership
# ======================================================

st.header("Cluster Membership")

membership_df = pd.DataFrame({
    "Neighbourhood": composition.index,
    "Cluster": area_clusters
})

for cluster_number in sorted(
    membership_df["Cluster"].unique()
):
    members = (
        membership_df.loc[
            membership_df["Cluster"] == cluster_number,
            "Neighbourhood"
        ]
        .sort_values()
        .tolist()
    )

    with st.expander(
        (
            f"Cluster {cluster_number} "
            f"— {len(members)} neighbourhoods"
        ),
        expanded=True
    ):
        st.write(", ".join(members))


# ======================================================
# Geographic visualization
# ======================================================

st.header("Geographic View of Neighbourhood Clusters")

# Use mean latitude and longitude as a representative
# centre point for each neighbourhood
area_centroids = (
    area_df
    .groupby("localarea")
    .agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        business_count=("licencenumber", "count")
    )
    .reindex(composition.index)
)

area_centroids["Cluster"] = area_clusters

area_centroids = area_centroids.reset_index()

area_centroids["Cluster"] = (
    area_centroids["Cluster"].astype(str)
)

# Use scatter_map where available.
# Fall back to scatter_mapbox for older Plotly versions.
if hasattr(px, "scatter_map"):
    map_figure = px.scatter_map(
        area_centroids,
        lat="latitude",
        lon="longitude",
        color="Cluster",
        size="business_count",
        hover_name="localarea",
        hover_data={
            "business_count": True,
            "latitude": ":.4f",
            "longitude": ":.4f"
        },
        zoom=10,
        height=650,
        title=(
            "Geographic Distribution of "
            f"Neighbourhood Clusters — K = {selected_k}"
        )
    )

    map_figure.update_layout(
        map_style="open-street-map",
        margin={
            "r": 0,
            "t": 50,
            "l": 0,
            "b": 0
        }
    )

else:
    map_figure = px.scatter_mapbox(
        area_centroids,
        lat="latitude",
        lon="longitude",
        color="Cluster",
        size="business_count",
        hover_name="localarea",
        hover_data={
            "business_count": True,
            "latitude": ":.4f",
            "longitude": ":.4f"
        },
        zoom=10,
        height=650,
        title=(
            "Geographic Distribution of "
            f"Neighbourhood Clusters — K = {selected_k}"
        )
    )

    map_figure.update_layout(
        mapbox_style="open-street-map",
        margin={
            "r": 0,
            "t": 50,
            "l": 0,
            "b": 0
        }
    )

st.plotly_chart(
    map_figure,
    use_container_width=True,
    key="geographic_map"
)

st.markdown(
    """
    The geographic view shows whether neighbourhoods with similar
    business compositions are also physically close to one another.
    Point colour represents cluster membership, while point size
    represents the number of business licences in each neighbourhood.
    """
)


# ======================================================
# Cluster business profiles
# ======================================================

st.header("Cluster Business Profiles")

# Add cluster labels to a separate copy
profile_df = composition.copy()
profile_df["Cluster"] = area_clusters

# Average business-type percentages within each cluster
cluster_profiles = (
    profile_df
    .groupby("Cluster")
    .mean()
)

for cluster_number in sorted(
    cluster_profiles.index
):
    top_types = (
        cluster_profiles
        .loc[cluster_number]
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
    )

    top_types.columns = [
        "Business Type",
        "Average Percentage"
    ]

    top_types["Average Percentage"] = (
        top_types["Average Percentage"].round(2)
    )

    st.subheader(
        f"Cluster {cluster_number}: Top Business Types"
    )

    profile_chart = px.bar(
        top_types,
        x="Average Percentage",
        y="Business Type",
        orientation="h",
        title=(
            "Top Business Types in "
            f"Cluster {cluster_number}"
        )
    )

    profile_chart.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        },
        height=350
    )

    st.plotly_chart(
        profile_chart,
        use_container_width=True,
        key=f"profile_chart_{cluster_number}"
    )


# ======================================================
# Interpretation
# ======================================================

st.header("Interpretation")

st.markdown(
    f"""
    The neighbourhoods were grouped using **K-means clustering**
    based on their business-type composition.

    With **K = {selected_k}**, neighbourhoods with similar business
    structures are assigned to the same cluster.

    The PCA visualization shows the similarity between neighbourhoods
    in a reduced two-dimensional feature space. Neighbourhoods located
    close together in the PCA plot have more similar distributions of
    business types.

    The geographic map shows that business similarity is not determined
    only by physical distance. Some neighbourhoods that are geographically
    separated may still belong to the same cluster because they contain
    similar proportions of business categories.

    The cluster membership lists and business-profile charts help explain
    each grouping by showing the neighbourhoods and dominant business
    types associated with each cluster.
    """
)

# END AI-assisted code