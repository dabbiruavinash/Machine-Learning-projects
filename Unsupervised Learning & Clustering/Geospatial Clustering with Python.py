What is Geospatial Clustering?
At its core, geospatial clustering is just unsupervised learning applied to latitude and longitude data, but the goal is always the same: to find meaningful groupings or patterns in spatial data to make location-based decisions smarter.

Let’s look at some applications of geospatial clustering:

Logistics: When you want to create delivery zones.
Retail: When you want to identify dense areas of customer activity to open a new store.
Urban planning: When you want to detect high-demand zones for public transport.
Crime analysis: When you want to find crime hotspots.
All these use cases have two things in common:

You’re working with location data,
You want to uncover natural groupings.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic

data = pd.read_csv("/content/deliverytime.txt")
data.head()

Now, we will calculate the real-world distance between the pickup point and the delivery location using the geodesic formula:

def calculate_distance(row):
    return geodesic(
        (row['Restaurant_latitude'], row['Restaurant_longitude']),
        (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    ).km

data['Distance_km'] = data.apply(calculate_distance, axis=1)

Here, we defined a function, calculate_distance, that takes a row of the dataset and computes the geographic (real-world) distance in kilometres between the restaurant and delivery coordinates using the geodesic method from the geopy library. We then used .apply() with axis=1 to apply this function row-wise and create a new column Distance_km, containing the distance for each delivery.

Now, let’s visualize all delivery locations across India on an interactive map using Plotly:

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=data['Delivery_location_longitude'],
    lat=data['Delivery_location_latitude'],
    mode='markers',
    marker=dict(color='blue', size=6, opacity=0.7),
    name='Delivery Locations',
    hovertemplate='Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra>Delivery</extra>'
))

fig.update_layout(
    title='📦 Mapping Our Reach — Delivery Locations Across India',
    geo=dict(
        scope='asia',
        showland=True,
        landcolor='rgb(229, 229, 229)',
        showcountries=True,
        countrycolor='rgb(200, 200, 200)',
        showlakes=False,
        lonaxis=dict(range=[68, 98]),  # focus on India
        lataxis=dict(range=[6, 38])
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    showlegend=False)

fig.show()

Performing K-Means Clustering:

from sklearn.cluster import KMeans

X = data[['Delivery_location_latitude', 'Delivery_location_longitude']]
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

fig = go.Figure()

for cluster_label in sorted(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_label]
    fig.add_trace(go.Scattergeo(
        lon=cluster_data['Delivery_location_longitude'],
        lat=cluster_data['Delivery_location_latitude'],
        mode='markers',
        name=f'Cluster {cluster_label}',
        marker=dict(size=6, opacity=0.7),
        hovertemplate='<b>Cluster:</b> %{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>',
        text=[f"{cluster_label}"] * len(cluster_data)
    ))

fig.add_trace(go.Scattergeo(
    lon=centroids[:, 1],
    lat=centroids[:, 0],
    mode='markers',
    name='Centroids',
    marker=dict(size=15, symbol='x', color='red', line=dict(width=2, color='black')),
    hovertemplate='<b>Centroid</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
))

fig.update_layout(
    title=f'📍 Geo-Spatial Clustering of Delivery Locations (k = {k})',
    geo=dict(
        scope='asia',
        showland=True,
        landcolor="rgb(229, 229, 229)",
        showcountries=True,
        countrycolor="rgb(204, 204, 204)",
        lonaxis=dict(range=[68, 98]),
        lataxis=dict(range=[6, 38]),
    ),
    legend_title='Clusters',
    margin=dict(l=0, r=0, t=60, b=0))

fig.show()

Cluster 0 (blue) represents the Central Delivery Zone, covering areas like Maharashtra and Madhya Pradesh, while Cluster 2 (green) forms the Southern Delivery Zone, focused around Tamil Nadu and Karnataka. However, Cluster 1 includes points that lie outside Indian geographic boundaries, indicating outliers or invalid coordinates likely due to GPS errors or data entry issues.

Now, let’s remove the outlier cluster and label valid delivery segments for optimized logistics planning:

filtered_data = data[data['Cluster'] != 1]
filtered_centroids = centroids[[0, 2]]  # Keep only Cluster 0 and 2

# Step 3: Map cluster names
cluster_labels = {
    0: "Central Delivery Zone",
    2: "Southern Delivery Zone"}

filtered_data['Optimized_Zone'] = filtered_data['Cluster'].map(cluster_labels)

Here, we filtered out Cluster 1, which represents outliers located outside India’s geographical boundaries. Then, we renamed the remaining valid clusters (Cluster 0 as “Central Delivery Zone” and Cluster 2 as “Southern Delivery Zone”) to give business context to the spatial segments. This final step transforms raw geospatial clusters into meaningful delivery zones that can be used for route optimization, staffing, and strategic planning.

