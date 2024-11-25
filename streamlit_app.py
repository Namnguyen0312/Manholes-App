import json
import math
import os
import re
from geopy.geocoders import OpenCage
import numpy as np
import pandas as pd
import streamlit as st
import folium
from paddleocr import PaddleOCR
from streamlit_folium import st_folium
import networkx as nx
from sklearn.neighbors import KDTree, BallTree
import osmnx as ox
from PIL import Image
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from ultralytics import YOLOv10
from keras.src.saving import load_model
# Set page configuration to use the full width of the browser window
st.set_page_config(layout="wide")

class_color_map = {
    'dourec': '#1f77b4',  # blue
    'rec': '#2ca02c',  # green
    'roundsqr': '#ff7f0e',  # orange
    'sqr': '#d62728',  # red
}

class_name_map = {
    'dourec': 'Double Rectangle Manhole',
    'rec': 'Rectangle Manhole',
    'roundsqr': 'Round Square Manhole',
    'sqr': 'Square Manhole',
}

HOME = "utils"
YOLO_MODEL_PATH = f"{HOME}/best.pt"
CLASSIFICATION_MODEL_PATH = f"{HOME}/saved_model_inceptionv3.h5"

# Tải mô hình YOLO và mô hình phân loại
yolo_model = YOLOv10(YOLO_MODEL_PATH)
classification_model = load_model(CLASSIFICATION_MODEL_PATH)

class_mapping = {
    0: "Double Rectangle Manhole",
    1: "Rectangle Manhole",
    2: "Round Square Manhole",
    3: "Square Manhole",
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

coordinates = df[['Latitude', 'Longitude']].values  # Giả sử dữ liệu tọa độ là 'Latitude' và 'Longitude'
positions = np.radians(coordinates)
tree = BallTree(positions)

# Tìm 2 node gần nhất trong Ball Tree (k=3 vì node gần nhất là node này và 2 node gần nhất khác)
adjacency_list = tree.query_radius(positions, r=500 / 6371000,
                                   return_distance=True)  # 'k=3' vì sẽ trả về node hiện tại và 2 node gần nhất khác

node_ids = list(G_sampled.nodes)
for i, neighbors in enumerate(adjacency_list[0]):
    node_i = node_ids[i + 44]
    for j, distance in zip(neighbors, adjacency_list[1][i]):
        node_j = node_ids[j + 44]
        if node_i < node_j:
            G_sampled.add_edge(node_i, node_j, weight=distance * 6371000)

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
        if u in G.nodes and v in G.nodes:
            # Tìm đường đi ngắn nhất giữa u và v
            path = nx.shortest_path(G, source=u, target=v, weight='length')
            # Lấy tọa độ các node trong đường đi
            path_coords = [(G.nodes[p]['y'], G.nodes[p]['x']) for p in path]
            # Vẽ đường đi lên bản đồ
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

    for edge in mst.edges:
        # Kiểm tra xem các node có tồn tại trong G hay không
        if edge[0] in G_sampled.nodes and edge[1] in G_sampled.nodes:
            pos1_lat = G_sampled.nodes[edge[0]].get('latitude')
            pos1_lon = G_sampled.nodes[edge[0]].get('longitude')
            pos2_lat = G_sampled.nodes[edge[1]].get('latitude')
            pos2_lon = G_sampled.nodes[edge[1]].get('longitude')

            # Nếu các node có tọa độ, vẽ đường nối
            if pos1_lat is not None and pos1_lon is not None and pos2_lat is not None and pos2_lon is not None:
                color = 'red' if highlight_nodes is None or (
                        edge[0] in highlight_nodes and edge[1] in highlight_nodes) else 'lightgray'
                folium.PolyLine(
                    locations=[(pos1_lat, pos1_lon), (pos2_lat, pos2_lon)],
                    color=color,
                    weight=4
                ).add_to(m)
            else:
                continue
        else:
            continue

    return m


# Hàm mới hiển thị chỉ node và edge (không hiển thị đường)
def display_node_edge_only(G, T, df, search_input, highlight_nodes=None, ):
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
        if u in G.nodes and v in G.nodes:
            # Tìm đường đi ngắn nhất giữa u và v
            path = nx.shortest_path(G, source=u, target=v, weight='length')
            # Lấy tọa độ các node trong đường đi
            path_coords = [(G.nodes[p]['y'], G.nodes[p]['x']) for p in path]
            # Vẽ đường đi lên bản đồ
            folium.PolyLine(locations=path_coords, color='green', weight=8).add_to(m)

    for edge in mst.edges:
        # Kiểm tra xem các node có tồn tại trong G hay không
        if edge[0] in G_sampled.nodes and edge[1] in G_sampled.nodes:
            pos1_lat = G_sampled.nodes[edge[0]].get('latitude')
            pos1_lon = G_sampled.nodes[edge[0]].get('longitude')
            pos2_lat = G_sampled.nodes[edge[1]].get('latitude')
            pos2_lon = G_sampled.nodes[edge[1]].get('longitude')

            # Nếu các node có tọa độ, vẽ đường nối
            if pos1_lat is not None and pos1_lon is not None and pos2_lat is not None and pos2_lon is not None:
                color = 'red' if highlight_nodes is None or (
                        edge[0] in highlight_nodes and edge[1] in highlight_nodes) else 'lightgray'
                folium.PolyLine(
                    locations=[(pos1_lat, pos1_lon), (pos2_lat, pos2_lon)],
                    color=color,
                    weight=4
                ).add_to(m)
            else:
                continue
        else:
            continue

    return m


def display_simulated_graph(G, mst, df, search_input, highlight_nodes=None):
    m = folium.Map(location=[center_latitude, center_longitude], zoom_start=15, tiles=None, )
    # Thêm mạng lưới đường phố vào bản đồ
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    for _, row in edges.iterrows():
        road_coords = [(point[1], point[0]) for point in row['geometry'].coords]
        folium.PolyLine(locations=road_coords, color='blue', weight=1, opacity=0.6).add_to(m)

    # Vẽ các cạnh MST
    for u, v, data in mst.edges(data=True):
        if u in G.nodes and v in G.nodes:
            # Tìm đường đi ngắn nhất giữa u và v
            path = nx.shortest_path(G, source=u, target=v, weight='length')
            # Lấy tọa độ các node trong đường đi
            path_coords = [(G.nodes[p]['y'], G.nodes[p]['x']) for p in path]
            # Vẽ đường đi lên bản đồ
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
            tooltip=f"Click to select Node: Lat {df['Latitude']}, Long {df['Longitude']}",
            location=(lat, lon),
            radius=marker_size, color=marker_color, fill=True, fill_color=marker_color
        ).add_to(m)

    for edge in mst.edges:
        # Kiểm tra xem các node có tồn tại trong G hay không
        if edge[0] in G_sampled.nodes and edge[1] in G_sampled.nodes:
            pos1_lat = G_sampled.nodes[edge[0]].get('latitude')
            pos1_lon = G_sampled.nodes[edge[0]].get('longitude')
            pos2_lat = G_sampled.nodes[edge[1]].get('latitude')
            pos2_lon = G_sampled.nodes[edge[1]].get('longitude')

            # Nếu các node có tọa độ, vẽ đường nối
            if pos1_lat is not None and pos1_lon is not None and pos2_lat is not None and pos2_lon is not None:
                color = 'red' if highlight_nodes is None or (
                        edge[0] in highlight_nodes and edge[1] in highlight_nodes) else 'lightgray'
                folium.PolyLine(
                    locations=[(pos1_lat, pos1_lon), (pos2_lat, pos2_lon)],
                    color=color,
                    weight=4
                ).add_to(m)
            else:
                continue
        else:
            continue

    return m


def search_nodes(search_input, search_type, G, df, manhole_type=None, selected_node_id=None, search_radius=None):
    highlight_nodes = None
    address_info = None
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


def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data into a variable
    return data

def initialize_ocr():
    """Khởi tạo PaddleOCR."""
    return PaddleOCR(use_angle_cls=True, lang='vi')

# Hàm lấy địa chỉ từ tọa độ
def get_address_from_coordinates(latitude, longitude, api_key):
    """Trả về địa chỉ từ tọa độ (latitude, longitude)."""
    geolocator = OpenCage(api_key)
    location = geolocator.reverse((latitude, longitude), language='vi')
    return location.address if location else None

# Hàm xử lý OCR trên một hình ảnh
def process_image(image_file, ocr, api_key):
    """Thực hiện OCR trên hình ảnh tải lên và trả về thông tin tọa độ."""
    results = []

    # Đọc hình ảnh từ buffer và chuyển thành numpy array
    image = Image.open(image_file)  # Mở từ Streamlit file uploader
    image_np = np.array(image)     # Chuyển thành mảng numpy
    image_pil = Image.fromarray(image_np)

    # Thực hiện OCR trực tiếp trên numpy array
    result = ocr.ocr(image_np, cls=True)

    latitude, longitude = None, None
    address = None

    for line in result[0]:
        text = line[1][0]

        # Tìm kiếm giá trị latitude
        lat_match = re.search(r"Lat\.?\s*([0-9.]+)[°\s]*", text, re.IGNORECASE)
        if lat_match:
            latitude = float(lat_match.group(1))

        # Tìm kiếm giá trị longitude
        long_match = re.search(r"Long\.?\s*([0-9.]+)[°\s]*", text, re.IGNORECASE)
        if long_match:
            longitude = float(long_match.group(1))

        # Nếu tìm được cả latitude và longitude
        if latitude is not None and longitude is not None:
            address = get_address_from_coordinates(latitude, longitude, api_key)

    # Thực hiện phát hiện đối tượng bằng YOLO
    detections = yolo_model(source=image_pil, conf=0.25)[0]

    # Truy cập các bounding box từ đối tượng detections.boxes
    for i, box in enumerate(detections.boxes):
        class_id = box.cls[0].item()  # Lấy class ID từ tọa độ
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        x1, y1, x2, y2 = map(int, cords)

        # Cắt ảnh từ bounding box
        cropped_img = image_pil.crop((x1, y1, x2, y2))
        cropped_img_array = np.array(cropped_img)

        # Phân loại hình ảnh đã cắt
        class_name, confidence = classify_image(cropped_img_array, classification_model)

        # Lưu tất cả thông tin vào một đối tượng
        results.append({
            "Filename": os.path.basename(image_file),
            "Latitude": latitude,
            "Longitude": longitude,
            "Address": address,
            "Class_Name": class_name,
        })

    return results



def classify_image(image_array, model):
    """Dự đoán lớp của một hình ảnh."""
    # Resize hình ảnh thành đầu vào của mô hình (224x224)
    resized_image = np.array(Image.fromarray(image_array).resize((224, 224))) / 255.0

    # Đảm bảo hình ảnh có 3 kênh (RGB)
    if resized_image.shape[-1] != 3:
        resized_image = np.stack((resized_image,) * 3, axis=-1)

    # Thêm chiều batch
    resized_image = resized_image[np.newaxis, ...]
    print("Shape before resize:", image_array.shape)
    print("Shape after resize:", resized_image.shape)
    # Dự đoán lớp
    prediction = model.predict(resized_image)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_mapping.get(class_index, "Unknown"), confidence

# Hàm xử lý toàn bộ các hình ảnh
def process_uploaded_files(uploaded_files, ocr, api_key):
    """Xử lý toàn bộ các file được upload."""
    all_results = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=f"Processing: {uploaded_file.name}", use_column_width=True)

        # Lưu file tạm thời từ Streamlit
        with open(uploaded_file.name, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        # Gọi hàm xử lý và phân loại
        results = process_image(temp_file.name, ocr, api_key)
        all_results.extend(results)

    return all_results

# Hàm tạo và tải về file Excel
def generate_download_button(results, output_file="ocr_results.xlsx"):
    """Tạo DataFrame và nút tải về file Excel."""
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
        df.to_excel(output_file, index=False, engine='openpyxl')

        # Nút tải file Excel
        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Excel file",
                data=f,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("No geographic information found in uploaded images.")


def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if 'search_type' not in st.session_state:
        st.session_state.search_type = "Node ID"
    if 'manhole_type' not in st.session_state:
        st.session_state.manhole_type = None
    if 'selected_node_id' not in st.session_state:
        st.session_state.selected_node_id = None
    if 'search_radius' not in st.session_state:
        st.session_state.search_radius = None
    if 'search_input' not in st.session_state:
        st.session_state.search_input = None
    if "markers" not in st.session_state:
        st.session_state["markers"] = []
    if "search_active" not in st.session_state:
        st.session_state.search_active = False
    if "selected_node_info" not in st.session_state:
        st.session_state.selected_node_info = None

    node_ids = df['node_id'].sort_values().tolist()

    st.title("AquaNetCT")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Graph Interaction", "Chatbot", "Detect Object"])



    # Tab 1: Graph Interaction
    with tab1:
        with st.sidebar:
            # Sidebar for search and node information
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
                st.session_state.manhole_type = \
                    [key for key, value in class_name_map.items() if value == selected_manhole][
                        0]
            elif search_type == "Radius":
                st.session_state.selected_node_id = st.selectbox("Select a node as the center", node_ids)
                st.session_state.search_radius = st.slider("Select radius (meters)", min_value=100, max_value=5000,
                                                           value=1000, step=100)
            elif search_type == "Node ID" or search_type == "Street Name":
                st.session_state.search_input = st.text_input('Search Input')
            else:
                st.session_state.search_active = False
                search_type = ""
                st.session_state.manhole_type = None
                st.session_state.selected_node_id = None
                st.session_state.search_radius = None

            search_button = False
            if search_type:
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
                        st.session_state.search_input if st.session_state.search_type != "Manhole Type" else None,
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
                        folium_map = display_real_map_with_graph(G_street, mst, df, search_type,
                                                                 highlight_nodes=highlight_nodes)
                    elif map_type == "Simulated Graph":
                        folium_map = display_simulated_graph(G_street, mst, df, search_type,
                                                             highlight_nodes=highlight_nodes)
                    elif map_type == "Node and Edge Only":
                        folium_map = display_node_edge_only(G_street, mst, df, search_type,
                                                            highlight_nodes=highlight_nodes)

            output = st_folium(folium_map, height=600, use_container_width=True,
                               returned_objects=["last_object_clicked"])

            # Check if a node was clicked
            if output and output.get("last_object_clicked"):
                if st.session_state.search_active:
                    st.warning("Please reset the search before clicking on nodes.")
                else:
                    clicked_location = output['last_object_clicked']
                    clicked_lat, clicked_lon = clicked_location['lat'], clicked_location['lng']

                    # Find nearest node and display its info
                    nearest_node = find_node_by_lat_lon(df, clicked_lat, clicked_lon)
                    if nearest_node is not None and not nearest_node.empty:
                        # Format node details for display and save in session state
                        node_info = {
                            "node_id": nearest_node['node_id'],  # Ensure proper indexing
                            "lat": nearest_node['Latitude'],
                            "lon": nearest_node['Longitude'],
                            "address": nearest_node['Address'],
                            "manhole_type": class_name_map[nearest_node['Class']],
                            "image_path": nearest_node['Image_Path']
                        }
                        st.session_state.selected_node_info = node_info

                        # Immediately update the display of the selected node info
                        st.subheader("Selected Node Information")
                        col1, col2 = st.columns(2)

                        # Column 1: Node Information
                        with col1:
                            st.write(f"**Node ID:** {node_info['node_id']}")
                            st.write(f"**Latitude:** {node_info['lat']:.6f}")
                            st.write(f"**Longitude:** {node_info['lon']:.6f}")
                            st.write(f"**Address:** {node_info.get('address', 'N/A')}")
                            st.write(f"**Manhole Type:** {node_info.get('manhole_type', 'N/A')}")

                        # Column 2: Image
                        with col2:
                            if os.path.isfile(node_info.get('image_path', 'N/A')):
                                image = load_image(node_info.get('image_path', 'N/A'))  # Cached image loading
                                st.image(image, caption=f"Image of Node {node_info['node_id']}", width=300)
                            else:
                                st.write("No image available for this node.")

            else:
                st.write("No node selected.")

            # Display a message if search is active
            if st.session_state.search_active:
                st.warning("To click on nodes, please reset your search first.")

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

    # Tab 2: Chatbot
    with tab2:

        raw_data = load_json_file('extracted_data.json')

        # Create text chunks from PDF data
        text_chunks = []
        for pdf_name, pdf_text in raw_data.items():
            chunks = get_chunk_text(pdf_text)
            text_chunks.extend(chunks)

        # Handle the conversation input and output
        messages = st.container(height=300)

        # Check if user input is available
        if prompt := st.chat_input("Say something"):

            # Get the assistant's response
            response = st.session_state.conversation({'question': prompt})

            # Update the chat history
            st.session_state.chat_history = response['chat_history']

            # Print the entire chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    messages.chat_message("user").write(message.content)

                else:
                    messages.chat_message("assistant").write(message.content)
    # Tab 3: Create file
    with tab3:
        API_KEY = os.getenv('OPENCAGE_API_KEY')
        st.title("OCR Location Extractor")
        st.write("Upload images to extract geographic information and save results to an Excel file.")

        uploaded_files = st.file_uploader(
            "Upload Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"]
        )

        if uploaded_files:
            ocr = initialize_ocr()
            results = process_uploaded_files(uploaded_files, ocr, API_KEY)
            st.success("Processing complete.")
            generate_download_button(results)

if __name__ == "__main__":
    main()
