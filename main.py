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
import os

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

        # Handle cuisine column
        self.handle_cuisine_data()

    def handle_cuisine_data(self):
        """Handle cuisine data - use existing column or mark as Unknown"""
        if 'cuisine' not in self.data.columns or self.data['cuisine'].isna().all():
            st.warning("No cuisine column found or all values are null. Setting all cuisines to 'Unknown'.")
            self.data['cuisine'] = 'Unknown'
        else:
            # Clean existing cuisine data
            self.data['cuisine'] = self.data['cuisine'].fillna('Unknown')
            st.info(f"Using existing cuisine column with {self.data['cuisine'].nunique()} unique cuisines.")

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
                center = cluster_data[['lat', 'lon']].values.mean(axis=0)
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


def load_default_data():
    """Try to load default data from repository"""
    try:
        # Try to load from heat_maps folder
        possible_paths = [
            'heat_maps/restaurants.csv',
            'heat_maps/restaurant_data.csv',
            'restaurants.csv',
            'restaurant_data.csv'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return pd.read_csv(path)

        # If no specific file found, look for any CSV in heat_maps folder
        if os.path.exists('heat_maps'):
            csv_files = [f for f in os.listdir('heat_maps') if f.endswith('.csv')]
            if csv_files:
                return pd.read_csv(f'heat_maps/{csv_files[0]}')

        return None
    except Exception as e:
        st.error(f"Error loading default data: {e}")
        return None


def main():
    st.title("🗺️ Restaurant Heatmap & Analytics Dashboard")
    st.markdown("Analyze restaurant data with interactive heatmaps and cuisine distributions!")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = RestaurantAnalyzer()

    # Try to load default data first
    if st.session_state.analyzer.data is None:
        default_data = load_default_data()
        if default_data is not None:
            st.session_state.analyzer.load_data(default_data)
            st.success("Restaurant data loaded successfully!")
        else:
            st.warning("No default data found. Please upload a CSV file.")

    # File upload section
    st.subheader("📁 Upload Custom Data")
    uploaded_file = st.file_uploader("Upload your restaurant CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.analyzer.load_data(data)
        st.success("Custom data uploaded successfully!")

    # Load from URL option
    with st.expander("🌐 Load Data from URL"):
        url = st.text_input("Enter CSV URL (Google Sheets, GitHub raw file, etc.)")
        if url and st.button("Load from URL"):
            try:
                data = pd.read_csv(url)
                st.session_state.analyzer.load_data(data)
                st.success("Data loaded from URL successfully!")
            except Exception as e:
                st.error(f"Error loading from URL: {e}")

    # Instructions for Google Sheets
    with st.expander("📊 Google Sheets Instructions"):
        st.markdown("""
        **To use Google Sheets data:**
        1. Make your Google Sheet public (Share → Anyone with the link can view)
        2. Get the share link
        3. Replace `/edit#gid=0` with `/export?format=csv&gid=0`
        4. Use the modified URL above

        **Example transformation:**
        - Original: `https://docs.google.com/spreadsheets/d/ABC123/edit#gid=0`
        - Use: `https://docs.google.com/spreadsheets/d/ABC123/export?format=csv&gid=0`
        """)

    # Main analysis section
    if st.session_state.analyzer.data is not None:
        data = st.session_state.analyzer.data

        # Display data info
        st.subheader("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Restaurants", len(data))
        with col2:
            st.metric("Unique Cuisines", data['cuisine'].nunique())
        with col3:
            if 'zone' in data.columns:
                st.metric("Unique Zones", data['zone'].nunique())
            else:
                st.metric("Data Points", len(data))
        with col4:
            st.metric("Columns", len(data.columns))

        # Show cuisine distribution summary
        st.subheader("🍽️ Cuisine Distribution Summary")
        cuisine_summary = data['cuisine'].value_counts().head(10)
        st.bar_chart(cuisine_summary)

        # Sidebar filters
        st.sidebar.header("🔍 Filters")

        # Cuisine filter
        cuisines = ['All'] + sorted(list(data['cuisine'].unique()))
        selected_cuisines = st.sidebar.multiselect("Select Cuisines", cuisines, default=['All'])

        # Zone filter if exists
        if 'zone' in data.columns:
            zones = ['All'] + sorted(list(data['zone'].unique()))
            selected_zones = st.sidebar.multiselect("Select Zones", zones, default=['All'])

        # Apply filters
        filtered_data = data.copy()
        if 'All' not in selected_cuisines and selected_cuisines:
            filtered_data = filtered_data[filtered_data['cuisine'].isin(selected_cuisines)]

        if 'zone' in data.columns and 'All' not in selected_zones and selected_zones:
            filtered_data = filtered_data[filtered_data['zone'].isin(selected_zones)]

        # Show filtered data info
        if len(filtered_data) != len(data):
            st.info(f"Filtered data: {len(filtered_data)} restaurants (from {len(data)} total)")

        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "📊 Clusters", "🥘 Cuisine Analysis", "📈 Density Analysis"])

        with tab1:
            st.subheader("Restaurant Heatmap")
            zoom_level = st.slider("Map Zoom Level", 5, 15, 10)
            heatmap = st.session_state.analyzer.create_heatmap(filtered_data, zoom_level)
            if heatmap:
                st.components.v1.html(heatmap._repr_html_(), height=600)
            else:
                st.warning("No data to display")

        with tab2:
            st.subheader("Restaurant Clusters")
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
        with st.expander("📋 View Raw Data"):
            st.dataframe(filtered_data)

        # Download filtered data
        if len(filtered_data) != len(data):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="💾 Download Filtered Data as CSV",
                data=csv,
                file_name='filtered_restaurants.csv',
                mime='text/csv'
            )

    else:
        st.info("👆 Please upload a CSV file or ensure your data file is in the heat_maps folder to get started!")

        # Show expected data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            Your CSV file should contain at least these columns:
            - **name**: Restaurant name
            - **lat**: Latitude (decimal degrees)
            - **lon**: Longitude (decimal degrees)
            - **cuisine**: Cuisine type (optional - will be extracted from names if not provided)

            **Optional columns:**
            - **zone**: Geographic zone or area
            - Any other columns will be preserved and displayed

            **Example:**
            ```
            name,lat,lon,cuisine,zone
            Mario's Pizza,40.7128,-74.0060,Italian,Manhattan
            Golden Dragon,40.7589,-73.9851,Chinese,Manhattan
            ```
            """)


if __name__ == "__main__":
    main()