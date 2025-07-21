import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import json

# Set page config
st.set_page_config(page_title="Restaurant Analytics Dashboard", layout="wide")


class RestaurantAnalyzer:
    def __init__(self):
        self.data = None

    def load_data(self, file_path_or_dataframe):
        """Load restaurant data from CSV or DataFrame"""
        if isinstance(file_path_or_dataframe, str):
            self.data = pd.read_csv(file_path_or_dataframe)
        else:
            self.data = file_path_or_dataframe.copy()

        # Clean and prepare data
        self.data = self.data.dropna(subset=['lat', 'lon'])

        # Add cuisine extraction if 'name' contains cuisine info
        # You might need to customize this based on your actual data
        self.extract_cuisine_info()

    def extract_cuisine_info(self):
        """Extract cuisine information from restaurant names or add sample cuisines"""
        # This is a placeholder - you'll need to adapt based on your actual data
        cuisines = ['Italian', 'Chinese', 'American', 'Mexican', 'Indian', 'Thai', 'Japanese', 'French', 'Greek',
                    'Pizza']
        np.random.seed(42)
        self.data['cuisine'] = np.random.choice(cuisines, size=len(self.data))

    def create_heatmap(self, filtered_data=None, zoom_level=10):
        """Create interactive heatmap with Folium"""
        data = filtered_data if filtered_data is not None else self.data

        if len(data) == 0:
            return None

        # Calculate center point
        center_lat = data['lat'].mean()
        center_lon = data['lon'].mean()

        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)

        # Add heatmap layer
        heat_data = [[row['lat'], row['lon']] for idx, row in data.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)

        # Add markers for individual restaurants
        for idx, row in data.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                popup=f"{row['name']}<br>Cuisine: {row.get('cuisine', 'Unknown')}",
                color='red',
                fill=True,
                opacity=0.7
            ).add_to(m)

        return m

    def create_density_polygons(self, filtered_data=None, n_clusters=5):
        """Create polygon clusters based on restaurant density"""
        data = filtered_data if filtered_data is not None else self.data

        if len(data) < n_clusters:
            return None

        # Perform K-means clustering
        coords = data[['lat', 'lon']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(coords)
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = clusters

        # Create map
        center_lat = data['lat'].mean()
        center_lon = data['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

        for i in range(n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == i]
            if len(cluster_data) > 0:
                # Create convex hull for cluster (simplified polygon)
                cluster_coords = cluster_data[['lat', 'lon']].values

                # Add circle markers for each cluster
                for idx, row in cluster_data.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=8,
                        popup=f"Cluster {i}<br>{row['name']}<br>Cuisine: {row.get('cuisine', 'Unknown')}",
                        color=colors[i % len(colors)],
                        fill=True,
                        opacity=0.8
                    ).add_to(m)

                # Add cluster center
                center = cluster_coords.mean(axis=0)
                folium.Marker(
                    location=center,
                    popup=f"Cluster {i} Center<br>{len(cluster_data)} restaurants",
                    icon=folium.Icon(color=colors[i % len(colors)], icon='info-sign')
                ).add_to(m)

        return m

    def create_cuisine_analysis(self, filtered_data=None):
        """Create various plots for cuisine analysis"""
        data = filtered_data if filtered_data is not None else self.data

        plots = {}

        # Cuisine distribution pie chart
        cuisine_counts = data['cuisine'].value_counts()
        fig_pie = px.pie(values=cuisine_counts.values, names=cuisine_counts.index,
                         title="Restaurant Distribution by Cuisine")
        plots['pie'] = fig_pie

        # Cuisine bar chart
        fig_bar = px.bar(x=cuisine_counts.index, y=cuisine_counts.values,
                         title="Number of Restaurants by Cuisine Type",
                         labels={'x': 'Cuisine', 'y': 'Count'})
        plots['bar'] = fig_bar

        # Geographic distribution by cuisine
        fig_scatter = px.scatter_mapbox(data, lat="lat", lon="lon", color="cuisine",
                                        hover_name="name", hover_data=["cuisine"],
                                        mapbox_style="open-street-map",
                                        title="Geographic Distribution by Cuisine",
                                        zoom=10, height=600)
        plots['scatter_map'] = fig_scatter

        return plots

    def create_density_analysis(self, filtered_data=None):
        """Create density analysis plots"""
        data = filtered_data if filtered_data is not None else self.data

        plots = {}

        # 2D density plot
        fig_density = px.density_mapbox(data, lat="lat", lon="lon", z=None,
                                        mapbox_style="open-street-map",
                                        title="Restaurant Density Heatmap",
                                        zoom=10, height=600)
        plots['density_map'] = fig_density

        # Histogram of coordinates
        fig_hist = make_subplots(rows=1, cols=2, subplot_titles=('Latitude Distribution', 'Longitude Distribution'))
        fig_hist.add_trace(go.Histogram(x=data['lat'], name='Latitude'), row=1, col=1)
        fig_hist.add_trace(go.Histogram(x=data['lon'], name='Longitude'), row=1, col=2)
        fig_hist.update_layout(title="Geographic Distribution Histograms")
        plots['histogram'] = fig_hist

        return plots


# Streamlit App
def main():
    st.title("🗺️ Restaurant Heatmap & Analytics Dashboard")
    st.markdown("Upload your restaurant data to create interactive heatmaps and analyze cuisine distributions!")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = RestaurantAnalyzer()

    # File upload
    uploaded_file = st.file_uploader("Upload your restaurant CSV file", type=['csv'])

    # Sample data option
    if st.button("Use Sample Data"):
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'original_index': range(100),
            'name': [f'Restaurant {i}' for i in range(100)],
            'zone': np.random.choice(['Zone A', 'Zone B', 'Zone C'], 100),
            'lat': np.random.normal(40.7128, 0.1, 100),  # Around NYC
            'lon': np.random.normal(-74.0060, 0.1, 100),
            'geocode_query_used': ['query'] * 100,
            'geocode_source': ['source'] * 100,
            'geocode_success': [True] * 100
        })
        st.session_state.analyzer.load_data(sample_data)
        st.success("Sample data loaded!")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.analyzer.load_data(data)
        st.success("Data loaded successfully!")

    if st.session_state.analyzer.data is not None:
        data = st.session_state.analyzer.data

        # Sidebar filters
        st.sidebar.header("Filters")

        # Cuisine filter
        cuisines = ['All'] + list(data['cuisine'].unique())
        selected_cuisines = st.sidebar.multiselect("Select Cuisines", cuisines, default=['All'])

        # Zone filter if exists
        if 'zone' in data.columns:
            zones = ['All'] + list(data['zone'].unique())
            selected_zones = st.sidebar.multiselect("Select Zones", zones, default=['All'])

        # Apply filters
        filtered_data = data.copy()
        if 'All' not in selected_cuisines and selected_cuisines:
            filtered_data = filtered_data[filtered_data['cuisine'].isin(selected_cuisines)]

        if 'zone' in data.columns and 'All' not in selected_zones and selected_zones:
            filtered_data = filtered_data[filtered_data['zone'].isin(selected_zones)]

        # Display data info
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Restaurants", len(filtered_data))
        with col2:
            st.metric("Unique Cuisines", filtered_data['cuisine'].nunique())
        with col3:
            if 'zone' in filtered_data.columns:
                st.metric("Unique Zones", filtered_data['zone'].nunique())

        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "📊 Polygons", "🥘 Cuisine Analysis", "📈 Density Analysis"])

        with tab1:
            st.subheader("Restaurant Heatmap")
            zoom_level = st.slider("Map Zoom Level", 5, 15, 10)
            heatmap = st.session_state.analyzer.create_heatmap(filtered_data, zoom_level)
            if heatmap:
                st.components.v1.html(heatmap._repr_html_(), height=600)
            else:
                st.warning("No data to display")

        with tab2:
            st.subheader("Density Polygons (Clusters)")
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            polygon_map = st.session_state.analyzer.create_density_polygons(filtered_data, n_clusters)
            if polygon_map:
                st.components.v1.html(polygon_map._repr_html_(), height=600)
            else:
                st.warning("Not enough data for clustering")

        with tab3:
            st.subheader("Cuisine Analysis")
            plots = st.session_state.analyzer.create_cuisine_analysis(filtered_data)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plots['pie'], use_container_width=True)
            with col2:
                st.plotly_chart(plots['bar'], use_container_width=True)

            st.plotly_chart(plots['scatter_map'], use_container_width=True)

        with tab4:
            st.subheader("Density Analysis")
            plots = st.session_state.analyzer.create_density_analysis(filtered_data)

            st.plotly_chart(plots['density_map'], use_container_width=True)
            st.plotly_chart(plots['histogram'], use_container_width=True)

        # Data table
        with st.expander("View Raw Data"):
            st.dataframe(filtered_data)


if __name__ == "__main__":
    main()