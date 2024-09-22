
import math
import os

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
import networkx as nx
from sklearn.neighbors import KDTree
import osmnx as ox
import random
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from PIL import Image

# Set page configuration to use the full width of the browser window
st.set_page_config(layout="wide")


# Từ điển màu sắc cho các loại manhole
class_color_map = {
    'dourec': 'blue',
    'rec': 'green',
    'roundsqr': 'orange',
    'sqr': 'red',
}

class_name_map = {
    'dourec': 'Double Rectangle Manhole',
    'rec': 'Rectangle Manhole',
    'roundsqr': 'Round Square Manhole',
    'sqr': 'Square Manhole',
}


# Read the Excel file
file_path = 'file_excel.xlsx'
df = pd.read_excel(file_path)

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


# Calculate the center of all the nodes
center_latitude, center_longitude = calculate_center(node_lists)

# Create the graph with OSMnx
radius = 2000  # Radius in meters
G_street = ox.graph_from_point((center_latitude, center_longitude), dist=radius, network_type='drive')


# Build a KD-Tree to find the nearest nodes
def build_kdtree(G):
    node_ids = list(G.nodes)
    coords = np.array([(G.nodes[node]['y'], G.nodes[node]['x']) for node in node_ids])
    kdtree = KDTree(coords, leaf_size=30, metric='euclidean')
    return kdtree, node_ids, coords


kdtree, node_ids, coords = build_kdtree(G_street)

# Find the closest nodes for each coordinate in node_lists
closest_nodes = []
for lon, lat in node_lists:
    dist, idx = kdtree.query([[lat, lon]], k=1)
    closest_node_id = node_ids[idx[0][0]]
    closest_nodes.append(closest_node_id)

# Compute the shortest paths between the closest nodes
all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G_street, weight='length'))

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
def display_real_map_with_graph(G, mst, df,search_input ,highlight_nodes = None ):
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
                location=(lat, lon),
                radius=marker_size, color=marker_color, fill=True, fill_color=marker_color
            ).add_to(m)

    # Hiển thị bản đồ trong Streamlit
    folium_static(m, width=600, height=400)


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
            print(marker_color)

        folium.CircleMarker(
            location=(lat, lon),
            radius=marker_size, color=marker_color, fill=True, fill_color=marker_color
        ).add_to(m)

    # Draw MST edges without street network
    for u, v, data in T.edges(data=True):
        path = nx.shortest_path(G, source=u, target=v, weight='length')
        path_coords = [(G.nodes[p]['y'], G.nodes[p]['x']) for p in path]
        folium.PolyLine(locations=path_coords, color='green', weight=8).add_to(m)

    folium_static(m, width=600, height=400)


def display_simulated_graph(G, mst, df, search_input, highlight_nodes=None,):
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=15, tiles=None)
    # Overlay street network with random colors
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    colors = plt.colormaps['tab20'].colors  # Lấy danh sách màu sắc từ colormap
    color_map = {}
    road_midpoints = {}

    for idx, row in edges.iterrows():
        road_name = row.get('name', 'Unnamed Road')
        if isinstance(road_name, list):
            road_name = ', '.join(road_name)  # Kết hợp các tên thành chuỗi
        if road_name not in color_map:
            color_map[road_name] = random.choice(colors)  # Chọn màu ngẫu nhiên cho tên đường mới
        hex_color = to_hex(color_map[road_name])  # Chuyển đổi màu sắc sang mã hex
        folium.PolyLine(
            locations=[(point[1], point[0]) for point in row['geometry'].coords],
            color=hex_color,  # Sử dụng mã màu hex đã chuyển đổi
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

    # Add MST edges to the map
    for u, v, data in mst.edges(data=True):
        path = nx.shortest_path(G, source=u, target=v, weight='length')
        path_coords = [(G.nodes[p]['y'], G.nodes[p]['x']) for p in path]
        folium.PolyLine(locations=path_coords, color='green', weight=4).add_to(m)
        # Thêm các node với màu dựa trên loại manhole
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
            location=(lat, lon),
            radius=marker_size, color=marker_color, fill=True, fill_color=marker_color
        ).add_to(m)


    # Display the map
    folium_static(m, width=600, height=400)


def search_nodes(search_input, search_type, G, df):
    highlight_nodes = None
    address_info = None

    if search_input:
        if search_type == "Node ID":
            # Tìm kiếm theo node ID
            try:
                search_id = int(search_input)
                if search_id in G:
                    attribute = G.nodes[search_id]
                    address = attribute.get('address')
                    node_count = 1  # Chỉ kiểm tra một ID
                    address_info = {
                        'address': address,
                        'node_count': node_count
                    }
                    highlight_nodes = {search_id}
                else:
                    address_info = {"error": "Node ID not found."}
            except ValueError:
                address_info = {"error": "Invalid Node ID."}

        elif search_type == "Street Name":
            # Tìm kiếm theo tên đường
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


        elif search_type == "Manhole Type":

            # Tìm kiếm theo loại cống

            matching_classes = [key for key, value in class_name_map.items() if value.lower() == search_input.lower()]

            if matching_classes:

                # Tìm các node tương ứng với các loại cống

                matching_nodes = df[df['Class'].isin(matching_classes)]

                node_count = len(matching_nodes)

                if node_count > 0:

                    highlight_nodes = set(matching_nodes['node_id'].tolist())

                    address_info = {

                        'address': search_input,

                        'node_count': node_count

                    }

                else:

                    address_info = {"error": "No nodes match the search query."}

            else:

                address_info = {"error": "Manhole type not found."}
        elif search_type == "Radius":
            try:
                radius = float(search_input)  # Bán kính người dùng nhập vào
                nearby_nodes = []

                for index, row in df.iterrows():
                    lat = row['Latitude']
                    lon = row['Longitude']
                    distance = haversine(center_latitude, center_longitude, lat, lon)

                    if distance <= radius:
                        nearby_nodes.append(row['node_id'])

                node_count = len(nearby_nodes)
                if node_count > 0:
                    highlight_nodes = set(nearby_nodes)
                    address_info = {
                        'address': f"Nodes within {radius} m radius",
                        'node_count': node_count
                    }
                else:
                    address_info = {"error": "No nodes found within the specified radius."}

            except ValueError:
                address_info = {"error": "Invalid radius input."}

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

def display_node_info(df, selected_node, image_width=300):
    # Get the row corresponding to the selected NodeID
    node_data = df[df['node_id'] == selected_node].iloc[0]
    st.write(f"**Node {node_data['node_id']} Information**")
    st.write(f"Address: {node_data['Address']}")
    st.write(f"Class: {node_data['Class']}")

    image_path = os.path.join(node_data['Folder'], node_data['Filename'])
    if os.path.isfile(image_path):
        image = Image.open(image_path)
        st.image(image, caption=f"Image of Node {node_data['node_id']}", width=image_width)
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



def main():
    st.title("From multi_videos to map: case of manhole")
    node_ids = df['node_id']

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        map_type = st.selectbox("Choose map type", ["Real Map with Graph Overlay", "Simulated Graph", "Node and Edge Only"])

        search_type = st.selectbox("Search Type", ["Node ID", "Street Name", "Manhole Type", "Radius"])
        search_input = st.text_input("Search Input")

        # Tìm kiếm node dựa trên input
        highlight_nodes, address_info = search_nodes(search_input, search_type, G_sampled, df)

        if address_info:
            if 'error' in address_info:
                st.write(address_info['error'])
            else:
                st.write(f"Address: {address_info['address']}")
                st.write(f"Number of nodes: {address_info['node_count']}")

        # Hiển thị bản đồ
        if map_type == "Real Map with Graph Overlay":
            display_real_map_with_graph(G_street, mst, df, search_input, highlight_nodes)
        elif map_type == "Simulated Graph":
            display_simulated_graph(G_street, mst, df, search_input, highlight_nodes)
        elif map_type == "Node and Edge Only":
            display_node_edge_only(G_street, mst, df, search_input, highlight_nodes)

    # Các phần còn lại của hàm main không thay đổi


    with col2:
        # Hiển thị danh sách các node để người dùng chọn
        selected_node = st.selectbox("Select a Node to View Info", node_ids)
        # Hiển thị thông tin và hình ảnh của node đã chọn
        display_node_info(df, selected_node)

        # Chọn hai node để tính khoảng cách
        st.write("### Calculate Distance Between Two Nodes")
        node1 = st.selectbox("Select the first node", node_ids, key="node1")
        node_data1 = df[df['node_id'] == node1].iloc[0]

        st.write(f"Address {node_data1['Address']}")
        st.write(f'Latitude {node_data1['Latitude']}')
        st.write(f'Longitude {node_data1['Longitude']}')
        node2 = st.selectbox("Select the second node", node_ids, key="node2")

        node_data2 = df[df['node_id'] == node2].iloc[0]
        st.write(f"Address {node_data2['Address']}")
        st.write(f'Latitude {node_data2['Latitude']}')
        st.write(f'Longitude {node_data2['Longitude']}')
        if node1 and node2:

            lat1 = node_data1['Latitude']
            lon1= node_data1['Longitude']
            lat2 = node_data2['Latitude']
            lon2 = node_data2['Longitude']
            # Gọi hàm tính khoảng cách giữa hai node
            distance = calculate_distance_between_nodes(lat1, lon1, lat2, lon2)

            # Hiển thị kết quả khoảng cách
            st.write(f"Distance between Node {node1} and Node {node2}: {distance:.2f} meters")

    with col3:
        # Hiển thị chú thích
        st.write("### Node Legend")
        for class_name, color in class_color_map.items():
            st.markdown(
                f'<div style="display: inline-block; width: 20px; height: 20px; background-color: {color};"></div> {class_name_map[class_name]}',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()

