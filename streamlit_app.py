import base64
import io
import os

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
import networkx as nx
from sklearn.neighbors import BallTree
import numpy as np
import osmnx as ox
import random
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from PIL import Image

# Set page configuration to use the full width of the browser window
st.set_page_config(layout="wide")


# Utility function to create a thumbnail image
def create_thumbnail(image_path, size=(100, 100)):
    img = Image.open(image_path)
    img.thumbnail(size)
    with io.BytesIO() as buffer:
        img.save(buffer, format="JPEG")
        thumbnail_data = buffer.getvalue()
    return thumbnail_data



@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)


@st.cache_resource
def create_graph(df, radius=500):
    G = nx.Graph()
    for idx, row in df.iterrows():
        node_id = idx + 1
        G.add_node(node_id, pos=(row['Longitude'], row['Latitude']),
                   filename=row['Filename'], folder=row['Folder'], address=row['Address'])

    positions = np.radians(np.array([G.nodes[node]['pos'][::-1] for node in G.nodes()]))
    tree = BallTree(positions)
    adjacency_list = tree.query_radius(positions, r=radius / 6371000, return_distance=True)

    for i, neighbors in enumerate(adjacency_list[0]):
        for j, distance in zip(neighbors, adjacency_list[1][i]):
            if i < j:
                G.add_edge(i + 1, j + 1, weight=distance * 6371000)
    return G


def generate_mst(G):
    return nx.minimum_spanning_tree(G)


def add_nodes_and_edges_to_map(m, G, T, image_cache, highlight_nodes=None):
    for node_id in T.nodes:
        pos = G.nodes[node_id]['pos']
        image_path = os.path.join(G.nodes[node_id]['folder'], G.nodes[node_id]['filename'])
        thumbnail_data = create_thumbnail(image_path) if os.path.isfile(image_path) else None

        if thumbnail_data:
            img_base64 = base64.b64encode(thumbnail_data).decode()
            popup_html = f"""
            <div>
                <strong>Node {node_id}</strong><br>
                {G.nodes[node_id]['address']}<br>
                <img src="data:image/jpeg;base64,{img_base64}" alt="Node Image" style="width:100%;"/>
            </div>
            """
        else:
            popup_html = f"""
            <div>
                <strong>Node {node_id}</strong><br>
                {G.nodes[node_id]['address']}<br>
                <em>No image available</em>
            </div>
            """

        color = 'red' if highlight_nodes is None or node_id in highlight_nodes else 'gray'
        fill_color = 'blue' if highlight_nodes is None or node_id in highlight_nodes else 'lightgray'

        folium.CircleMarker(
            location=(pos[1], pos[0]),
            radius=5, color=color, fill=True, fill_color=fill_color,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    for edge in T.edges:
        pos1 = G.nodes[edge[0]]['pos']
        pos2 = G.nodes[edge[1]]['pos']
        color = 'green' if highlight_nodes is None or (edge[0] in highlight_nodes and edge[1] in highlight_nodes) else 'lightgray'
        folium.PolyLine(
            locations=[(pos1[1], pos1[0]), (pos2[1], pos2[0])],
            color=color,
            weight=4
        ).add_to(m)


def display_real_map_with_graph(G, T, df, image_cache, highlight_nodes=None):
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=15)

    # Add the street network overlay
    latitude, longitude = df['Latitude'].mean(), df['Longitude'].mean()
    radius = 1000  # meters
    G_street = ox.graph_from_point((latitude, longitude), dist=radius, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(G_street)
    edges['name'] = edges['name'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

    color_map = {}
    road_midpoints = {}

    for idx, row in edges.iterrows():
        road_name = row.get('name', 'Unnamed Road')
        color = 'blue'
        folium.PolyLine(
            locations=[(point[1], point[0]) for point in row['geometry'].coords],
            color=color,
            weight=2
        ).add_to(m)
        if road_name != 'Unnamed Road':
            if road_name not in road_midpoints or row['geometry'].length > road_midpoints[road_name][1]:
                mid_point = row['geometry'].interpolate(0.5, normalized=True).coords[0]
                road_midpoints[road_name] = (mid_point, row['geometry'].length)

    for road_name, (mid_point, _) in road_midpoints.items():
        folium.map.Marker(
            [mid_point[1], mid_point[0]],
            icon=folium.DivIcon(html=f"""<div style="font-size: 10px; color: black;">{road_name}</div>""")
        ).add_to(m)

    # Add nodes and edges from the simulated graph
    add_nodes_and_edges_to_map(m, G, T, image_cache, highlight_nodes=highlight_nodes)

    # Display the map in full width
    folium_static(m, width=1400, height=800)
    return m


def display_simulated_graph(G, T, df, image_cache, highlight_nodes=None):
    m_simulation = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=15, tiles=None)

    # Overlay street network
    latitude, longitude = df['Latitude'].mean(), df['Longitude'].mean()
    radius = 1000  # meters
    G_street = ox.graph_from_point((latitude, longitude), dist=radius, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(G_street)
    edges['name'] = edges['name'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

    colors = plt.colormaps['tab20'].colors
    color_map = {}
    road_midpoints = {}

    for idx, row in edges.iterrows():
        road_name = row.get('name', 'Unnamed Road')
        if road_name not in color_map:
            color_map[road_name] = random.choice(colors)
        hex_color = to_hex(color_map[road_name])
        color = hex_color
        folium.PolyLine(
            locations=[(point[1], point[0]) for point in row['geometry'].coords],
            color=color,
            weight=2
        ).add_to(m_simulation)
        if road_name != 'Unnamed Road':
            if road_name not in road_midpoints or row['geometry'].length > road_midpoints[road_name][1]:
                mid_point = row['geometry'].interpolate(0.5, normalized=True).coords[0]
                road_midpoints[road_name] = (mid_point, row['geometry'].length)

    for road_name, (mid_point, _) in road_midpoints.items():
        folium.map.Marker(
            [mid_point[1], mid_point[0]],
            icon=folium.DivIcon(html=f"""<div style="font-size: 10px; color: black;">{road_name}</div>""")
        ).add_to(m_simulation)

    # Add nodes and edges from the simulated graph
    add_nodes_and_edges_to_map(m_simulation, G, T, image_cache, highlight_nodes=highlight_nodes)

    # Display the map in full width
    folium_static(m_simulation, width=1400, height=800)
    return m_simulation


def search_nodes(search_input, G, df):
    highlight_nodes = None
    address_info = None

    if search_input:
        # Try to interpret the input as a node ID
        try:
            node_id = int(search_input)
            if node_id in G.nodes:
                address = G.nodes[node_id]['address']

                # Find all nodes with the same address
                matching_nodes = df[df['Address'] == address].index + 1
                node_count = len(matching_nodes)
                address_info = {
                    'address': address,
                    'node_count': node_count
                }
                highlight_nodes = set(matching_nodes)
            else:
                address_info = {"error": "Node ID not found."}
        except ValueError:
            # If input is not a valid integer, treat it as an address
            search_query = search_input.lower()
            matching_nodes = df[df['Address'].str.lower().str.contains(search_query)].index + 1
            node_count = len(matching_nodes)
            if node_count > 0:
                address_info = {
                    'address': search_input,
                    'node_count': node_count
                }
            highlight_nodes = set(matching_nodes)

    return highlight_nodes, address_info


def main():
    df = load_data('paddle_output_all.xlsx')
    G = create_graph(df)
    T = generate_mst(G)

    st.title("From multi_videos to map: case of manhole")

    image_cache = {}

    map_type = st.selectbox("Choose map type", ["Real Map with Graph Overlay", "Simulated Graph"])

    search_input = st.text_input("Search by Node ID or Address")

    highlight_nodes, address_info = search_nodes(search_input, G, df)

    if address_info:
        if 'error' in address_info:
            st.write(address_info['error'])
        else:
            st.write(f"Address: {address_info['address']}")
            st.write(f"Number of nodes: {address_info['node_count']}")

    if map_type == "Real Map with Graph Overlay":
        m = display_real_map_with_graph(G, T, df, image_cache, highlight_nodes)
    elif map_type == "Simulated Graph":
        m = display_simulated_graph(G, T, df, image_cache, highlight_nodes)

if __name__ == "__main__":
    main()

