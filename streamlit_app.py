
import math
import os

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import networkx as nx
from sklearn.neighbors import KDTree
import osmnx as ox
from PIL import Image



# Set page configuration to use the full width of the browser window
st.set_page_config(layout="wide")


class_color_map = {
    'dourec': '#1f77b4',  # blue
    'rec': '#2ca02c',     # green
    'roundsqr': '#ff7f0e',# orange
    'sqr': '#d62728',     # red
}

class_name_map = {
    'dourec': 'Double Rectangle Manhole',
    'rec': 'Rectangle Manhole',
    'roundsqr': 'Round Square Manhole',
    'sqr': 'Square Manhole',
}

@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

file_path = 'file_excel.xlsx'
df = load_data(file_path)

# Get the 'Longitude' and 'Latitude' columns
lon_lat_columns = ['Longitude', 'Latitude']
node_lists = df[lon_lat_columns].values.tolist()


# Function to calculate the center of the coordinates
def calculate_center(node_lists):
    total_lon = sum([lon for lon, lat in node_lists])
    total_lat = sum([lat for lon, lat in node_lists])
    center_lon = total_lon / len(node_lists)
    center_lat = total_lat / len(node_lists)
    return center_lat, center_lon

# Build a KD-Tree to find the nearest nodes
def build_kdtree(G):
    node_ids = list(G.nodes)
    coords = np.array([(G.nodes[node]['y'], G.nodes[node]['x']) for node in node_ids])
    kdtree = KDTree(coords, leaf_size=30, metric='euclidean')
    return kdtree, node_ids, coords

@st.cache_resource
def build_kdtree_and_graph(radius=2000):
    G_street = ox.graph_from_point((center_latitude, center_longitude), dist=radius, network_type='drive')
    kdtree, node_ids, coords = build_kdtree(G_street)
    return G_street, kdtree, node_ids, coords

# Calculate the center of all the nodes
center_latitude, center_longitude = calculate_center(node_lists)

# Build KDTree and Graph
G_street, kdtree, node_ids, coords = build_kdtree_and_graph()



kdtree, node_ids, coords = build_kdtree(G_street)

# Find the closest nodes for each coordinate in node_lists
closest_nodes = []
for lon, lat in node_lists:
    dist, idx = kdtree.query([[lat, lon]], k=1)
    closest_node_id = node_ids[idx[0][0]]
    closest_nodes.append(closest_node_id)

@st.cache_resource
def compute_shortest_paths(_G):
    return dict(nx.all_pairs_dijkstra_path_length(_G, weight='length'))

all_pairs_shortest_paths = compute_shortest_paths(G_street)

# Set a distance threshold
k = 2000  # Max distance in meters

# Collect potential edges for the graph
potential_edges = []
for i in range(len(closest_nodes)):
    for j in range(i + 1, len(closest_nodes)):
        node_i = closest_nodes[i]
        node_j = closest_nodes[j]

        if node_j in all_pairs_shortest_paths[node_i]:
            shortest_path_length = all_pairs_shortest_paths[node_i][node_j]
            if shortest_path_length <= k:
                potential_edges.append((node_i, node_j, shortest_path_length))

# Create a sampled graph with only the required edges
G_sampled = nx.Graph()
G_sampled.add_weighted_edges_from(potential_edges)
for index, row in df.iterrows():
    node_id = row['node_id']  # Replace 'NodeID' with your actual unique identifier
    # Add the node to the graph with attributes
    G_sampled.add_node(node_id,
                       longitude=row['Longitude'],
                       latitude=row['Latitude'],
                       class_type=row['Class'],  # Replace 'ClassType' with the actual column name
                       file_path=row['Filename'],
                       address=row['Address'],
                       class_name=class_name_map.get(row['Class'], 'Unknown'))
# Generate the Minimum Spanning Tree (MST)
mst = nx.minimum_spanning_tree(G_sampled)

# Hàm hiển thị bản đồ với mạng lưới
def display_real_map_with_graph(G, mst, df, search_input, highlight_nodes=None):
    m = folium.Map(location=[center_latitude, center_longitude], zoom_start=15)

    # Thêm mạng lưới đường phố vào bản đồ
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    for _, row in edges.iterrows():
        road_coords = [(point[1], point[0]) for point in row['geometry'].coords]
        folium.PolyLine(locations=road_coords, color='blue', weight=1, opacity=0.6).add_to(m)

    # Vẽ các cạnh MST
    for u, v, data in mst.edges(data=True):
        path = nx.shortest_path(G, source=u, target=v, weight='length')
        path_coords = [(G.nodes[p]['y'], G.nodes[p]['x']) for p in path]
        folium.PolyLine(locations=path_coords, color='green', weight=8).add_to(m)

    # Tạo danh sách marker tạm thời
    markers_info = []
    for i, (lon, lat) in enumerate(node_lists):
        manhole_class = df.loc[i, 'Class']
        color = class_color_map.get(manhole_class, 'gray')
        node_id = df.loc[i, 'node_id']
        marker_size = 7 if highlight_nodes is not None and node_id in highlight_nodes else 5

        if search_input is None or search_input == "":
            marker_color = color
        else:
            marker_color = 'gray' if highlight_nodes is None or node_id not in highlight_nodes else color

        # Lưu thông tin marker vào danh sách tạm
        markers_info.append({
            'node_id': node_id,
            'location': (lat, lon),
            'color': marker_color,
            'size': marker_size
        })

    # Vẽ các marker trên bản đồ
    for marker in markers_info:
        folium.CircleMarker(
            tooltip=f"Click to select Node: {marker['location']}",
            location=marker['location'],
            radius=marker['size'],
            color=marker['color'],
            fill=True,
            fill_color=marker['color']
        ).add_to(m)

    return m



# Hàm mới hiển thị chỉ node và edge (không hiển thị đường)
def display_node_edge_only(G, T, df , search_input, highlight_nodes=None, ):
    m = folium.Map(location=[center_latitude, center_longitude], zoom_start=15, tiles=None)
    for i, (lon, lat) in enumerate(node_lists):
        manhole_class = df.loc[i, 'Class']
        color = class_color_map.get(manhole_class, 'gray')
        node_id = df.loc[i, 'node_id']  # Use the NodeID

        marker_size = 7 if highlight_nodes is not None and node_id in highlight_nodes else 5
        if search_input is None or search_input == "":
            marker_color = color
        else:
            marker_color = 'gray' if highlight_nodes is None or node_id not in highlight_nodes else color

        folium.CircleMarker(
            tooltip=f"Click to select Node: Lat {df['Latitude']}, Long {df['Longitude']}",

            location=(lat, lon),
            radius=marker_size, color=marker_color, fill=True, fill_color=marker_color
        ).add_to(m)

    # Draw MST edges without street network
    for u, v, data in T.edges(data=True):
        path = nx.shortest_path(G, source=u, target=v, weight='length')
        path_coords = [(G.nodes[p]['y'], G.nodes[p]['x']) for p in path]
        folium.PolyLine(locations=path_coords, color='green', weight=8).add_to(m)



    return m

def display_simulated_graph(G, mst, df, search_input, highlight_nodes=None):
    m = folium.Map(location=[center_latitude, center_longitude], zoom_start=15, tiles=None,)
    print(search_input)
    # Thêm mạng lưới đường phố vào bản đồ
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    for _, row in edges.iterrows():
        road_coords = [(point[1], point[0]) for point in row['geometry'].coords]
        folium.PolyLine(locations=road_coords, color='blue', weight=1, opacity=0.6).add_to(m)

    # Vẽ các cạnh MST
    for u, v, data in mst.edges(data=True):
        path = nx.shortest_path(G, source=u, target=v, weight='length')
        path_coords = [(G.nodes[p]['y'], G.nodes[p]['x']) for p in path]
        folium.PolyLine(locations=path_coords, color='green', weight=8).add_to(m)

    for i, (lon, lat) in enumerate(node_lists):
        manhole_class = df.loc[i, 'Class']
        color = class_color_map.get(manhole_class, 'gray')
        node_id = df.loc[i, 'node_id']  # Use the NodeID
        marker_size = 7 if highlight_nodes is not None and node_id in highlight_nodes else 5
        if search_input is None or search_input == "":
            marker_color = color
        else:
            marker_color = 'gray' if highlight_nodes is None or node_id not in highlight_nodes else color

        image_path = df.loc[df['node_id'] == node_id, 'Image_Path'].values[0]  # Get the Image Path
        folium.CircleMarker(
            tooltip=f"Click to select Node: Lat {df['Latitude']}, Long {df['Longitude']}",
            location=(lat, lon),
            radius=marker_size, color=marker_color, fill=True, fill_color=marker_color
        ).add_to(m)

    return m



def search_nodes(search_input, search_type, G, df, manhole_type=None, selected_node_id=None, search_radius=None):
    highlight_nodes = None
    address_info = None
    print(search_type)
    # Search by Node ID
    if search_type == "Node ID":
        try:
            search_id = int(search_input)
            if search_id in G:
                attribute = G.nodes[search_id]
                address = attribute.get('address')
                node_count = 1  # Searching for a single ID
                address_info = {
                    'address': address,
                    'node_count': node_count
                }
                highlight_nodes = {search_id}
            else:
                address_info = {"error": "Node ID not found."}
        except ValueError:
            address_info = {"error": "Invalid Node ID."}

    # Search by Street Name
    elif search_type == "Street Name":
        search_query = search_input.lower()
        matching_nodes = df[df['Address'].str.lower().str.contains(search_query)]
        node_count = len(matching_nodes)

        if node_count > 0:
            highlight_nodes = set(matching_nodes['node_id'].tolist())
            address_info = {
                'address': search_input,
                'node_count': node_count
            }
        else:
            address_info = {"error": "No nodes match the search query."}

    # Search by Manhole Type
    elif search_type == "Manhole Type":
        matching_nodes = df[df['Class'] == manhole_type]
        node_count = len(matching_nodes)

        if node_count > 0:
            highlight_nodes = set(matching_nodes['node_id'].tolist())
            address_info = {
                'address': manhole_type,
                'node_count': node_count
            }
        else:
            address_info = {"error": f"No nodes of type {manhole_type} found."}

    elif search_type == "Radius" and selected_node_id is not None and search_radius is not None:
        # Tìm kiếm node trong bán kính dựa trên node_id và bán kính đã chọn
        try:
            search_id = int(selected_node_id)

            lat1 = G.nodes[search_id]['latitude']
            lon1 = G.nodes[search_id]['longitude']

            nearby_nodes = []
            for node_id in G.nodes:
                if 'latitude' not in G.nodes[node_id] or 'longitude' not in G.nodes[node_id]:
                    continue
                lat2 = G.nodes[node_id]['latitude']
                lon2 = G.nodes[node_id]['longitude']
                distance = haversine(lat1, lon1, lat2, lon2)

                if distance <= search_radius:
                    nearby_nodes.append(node_id)

            node_count = len(nearby_nodes)
            if node_count > 0:
                highlight_nodes = set(nearby_nodes)
                address_info = {
                    'address': f"Nodes within {search_radius} m radius from node {selected_node_id}",
                    'node_count': node_count
                }
            else:
                address_info = {"error": f"No nodes found within {search_radius} m from node {selected_node_id}."}

        except ValueError:
            address_info = {"error": "Invalid node ID or radius input."}

    return highlight_nodes, address_info


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Bán kính trái đất (đơn vị: m)

    # Chuyển đổi độ sang radian
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Công thức Haversine
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # Khoảng cách (đơn vị: m)
    return distance

@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

def display_node_info(selected_node, image_width=500):
    st.write(f"**Node {selected_node['node_id']} Information**")
    st.write(f"Address: {selected_node['Address']}")
    st.write(f"Class: {selected_node['Class']}")

    image_path = selected_node['Image_Path']
    if os.path.isfile(image_path):
        image = load_image(image_path)  # Cached image loading
        st.image(image, caption=f"Image of Node {selected_node['node_id']}", width=image_width)
    else:
        st.write("No image available for this node.")



def calculate_distance_between_nodes(lat1, lon1, lat2, lon2):
    # Công thức Haversine để tính khoảng cách giữa hai tọa độ địa lý
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Bán kính Trái đất theo mét
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c  # Khoảng cách theo mét
        return distance

    # Tính toán khoảng cách giữa hai tọa độ
    distance = haversine(lat1, lon1, lat2, lon2)

    return distance

def find_node_by_lat_lon(df, lat, lon):
    # So sánh trực tiếp tọa độ
    node = df[(df['Latitude'] == lat) & (df['Longitude'] == lon)]
    if not node.empty:
        return node.iloc[0]
    return None

def main():
    st.title("Manhole Network Map")

    # Initialize session state variables if they don't exist
    if 'search_type' not in st.session_state:
        st.session_state.search_type = "Node ID"
    if 'manhole_type' not in st.session_state:
        st.session_state.manhole_type = None
    if 'selected_node_id' not in st.session_state:
        st.session_state.selected_node_id = None
    if 'search_radius' not in st.session_state:
        st.session_state.search_radius = None
    if "markers" not in st.session_state:
        st.session_state["markers"] = []
    if "search_active" not in st.session_state:
        st.session_state.search_active = False

    node_ids = df['node_id'].sort_values().tolist()

    with st.sidebar:
        st.header("Search and Visualization")

        # Choose map type
        map_type = st.selectbox("Choose map type",
                                ["Real Map with Graph Overlay", "Simulated Graph", "Node and Edge Only"])
        # Choose search type
        search_type = st.selectbox("Search Type", ["", "Node ID", "Street Name", "Manhole Type", "Radius"], index=0)

        # Clear search input and parameters when changing search type
        if 'search_type' in st.session_state and st.session_state.search_type != search_type:
            st.session_state.manhole_type = None
            st.session_state.selected_node_id = None
            st.session_state.search_radius = None

        st.session_state.search_type = search_type

        # Based on search type, require different inputs
        if search_type == "Manhole Type":
            selected_manhole = st.selectbox("Select Manhole Type", list(class_name_map.values()))
            st.session_state.manhole_type = [key for key, value in class_name_map.items() if value == selected_manhole][
                0]
        elif search_type == "Radius":
            st.session_state.selected_node_id = st.selectbox("Select a node as the center", node_ids)
            st.session_state.search_radius = st.slider("Select radius (meters)", min_value=100, max_value=5000,
                                                       value=1000, step=100)
        elif search_type == "Node ID" or search_type == "Street Name":
            st.text_input('Search Input')
        else :
            st.session_state.search_active = False
            search_type = ""
            st.session_state.manhole_type = None
            st.session_state.selected_node_id = None
            st.session_state.search_radius = None

        search_button = False
        if(search_type is not ""):
            search_button = st.button("Search")

    # Display map and other default functionalities if search button hasn't been pressed
    st.write("### Map Visualization")
    with st.expander("Map Display"):
        if not search_button:
            # Display the default map
            if map_type == "Real Map with Graph Overlay":
                folium_map = display_real_map_with_graph(G_street, mst, df, search_type)
            elif map_type == "Simulated Graph":
                folium_map = display_simulated_graph(G_street, mst, df, search_type)
            elif map_type == "Node and Edge Only":
                folium_map = display_node_edge_only(G_street, mst, df, search_type)
        else:
            # Perform search and display results if search is active
            if st.session_state.search_type == "Manhole Type" and not st.session_state.manhole_type:
                st.warning("Please select a manhole type.")
            elif st.session_state.search_type == "Radius" and (
                    not st.session_state.selected_node_id or not st.session_state.search_radius):
                st.warning("Please select both a node and a radius.")
            elif st.session_state.search_type in ["Node ID", "Street Name"] and not search_type:
                st.warning(f"Please enter a valid {st.session_state.search_type}.")
            else:
                highlight_nodes, address_info = search_nodes(
                    search_type if st.session_state.search_type != "Manhole Type" else None,
                    st.session_state.search_type, G_sampled, df, st.session_state.manhole_type,
                    selected_node_id=st.session_state.selected_node_id, search_radius=st.session_state.search_radius
                )

                if address_info:
                    if 'error' in address_info:
                        st.write(address_info['error'])
                    else:
                        st.write(f"Address: {address_info['address']}")
                        st.write(f"Number of nodes: {address_info['node_count']}")

                # Display the map with highlighted nodes
                st.write("### Highlighted Nodes")
                if map_type == "Real Map with Graph Overlay":
                    folium_map = display_real_map_with_graph(G_street, mst, df, search_type, highlight_nodes=highlight_nodes)
                elif map_type == "Simulated Graph":
                    folium_map = display_simulated_graph(G_street, mst, df, search_type, highlight_nodes=highlight_nodes)
                elif map_type == "Node and Edge Only":
                    folium_map = display_node_edge_only(G_street, mst, df, search_type, highlight_nodes=highlight_nodes)

        output = st_folium(folium_map, height=600, use_container_width=True, returned_objects=["last_object_clicked"])

        if output is not None and output.get("last_object_clicked") is not None:
            clicked_location = output['last_object_clicked']
            clicked_lat, clicked_lon = clicked_location['lat'], clicked_location['lng']
            nearest_node = find_node_by_lat_lon(df, clicked_lat, clicked_lon)
            display_node_info(nearest_node)
        else:
            st.write("No node selected.")

    with st.expander("Distance Calculation"):
        st.write("### Calculate Distance Between Nodes")
        node1 = st.selectbox("Select the first node", node_ids, key="node1")
        node2 = st.selectbox("Select the second node", node_ids, key="node2")
        distance = calculate_distance_between_nodes(
            df[df['node_id'] == node1].iloc[0]['Latitude'],
            df[df['node_id'] == node1].iloc[0]['Longitude'],
            df[df['node_id'] == node2].iloc[0]['Latitude'],
            df[df['node_id'] == node2].iloc[0]['Longitude']
        )
        st.write(f"Distance between Node {node1} and Node {node2}: {distance:.2f} meters")

if __name__ == "__main__":
    main()


