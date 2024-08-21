import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
import networkx as nx
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from sklearn.neighbors import BallTree
import numpy as np

# Set page configuration to use the full width of the browser window
st.set_page_config(layout="wide")

# Đọc dữ liệu từ file Excel
df = pd.read_excel('paddle_output_all.xlsx')

# Tạo ứng dụng Streamlit
st.title("From multi_videos to map: case of manhole")

# Chia ứng dụng thành hai cột
col1, col2 = st.columns([2, 1])  # Adjust proportions as needed

# Cột bên trái: Chọn loại bản đồ và hiển thị
with col1:
    # Chọn loại bản đồ: Bản đồ thật hoặc Graph
    map_type = st.selectbox("Chọn loại bản đồ", ["Bản đồ thật", "Graph mô phỏng"])

    # Tạo một graph rỗng
    G = nx.Graph()

    # Thêm các node từ dữ liệu Latitude và Longitude
    for idx, row in df.iterrows():
        node_id = idx + 1  # ID của node dựa trên chỉ số hàng
        G.add_node(node_id, pos=(row['Longitude'], row['Latitude']),
                   filename=row['Filename'], folder=row['Folder'], address=row['Address'])

    # Slider để điều chỉnh ngưỡng khoảng cách giữa các node
    threshold = st.slider("Chọn ngưỡng khoảng cách (m)", min_value=10, max_value=100, value=50, step=10)

    # Lấy danh sách các vị trí từ node và chuyển đổi sang radians (cần thiết cho BallTree)
    positions = np.radians(np.array([G.nodes[node]['pos'][::-1] for node in G.nodes()]))

    # Tạo BallTree và tìm các cặp node nằm trong khoảng cách ngưỡng (sử dụng Haversine distance)
    tree = BallTree(positions, metric='haversine')
    adjacency_list = tree.query_radius(positions, r=threshold / 6371000, return_distance=True)

    # Thêm các cạnh vào graph dựa trên kết quả từ BallTree
    for i, neighbors in enumerate(adjacency_list[0]):
        for j, distance in zip(neighbors, adjacency_list[1][i]):
            if i < j:  # Đảm bảo không thêm cạnh hai lần
                G.add_edge(i + 1, j + 1, weight=distance * 6371000)  # Nhân lại với bán kính Trái đất để có khoảng cách thực tế

    # Hiển thị bản đồ dựa trên lựa chọn của người dùng
    if map_type == "Bản đồ thật":
        # Tạo bản đồ với Folium
        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=15)

        # Vẽ các node lên bản đồ và gán ID
        for node_id in G.nodes:
            pos = G.nodes[node_id]['pos']
            address = G.nodes[node_id]['address']
            folium.Marker(
                [pos[1], pos[0]],  # Latitude, Longitude
                popup=f"Node {node_id}: {address}",
                tooltip=address
            ).add_to(m)

        # Hiển thị bản đồ
        folium_static(m)

    elif map_type == "Graph mô phỏng":
        # Vẽ graph mô phỏng với Plotly
        pos = nx.get_node_attributes(G, 'pos')

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_x.append(None)
            edge_y.append(None)

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=2, color='#888')))

        # Add nodes without displaying text
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='lightblue'),
                                 hoverinfo='text', text=[f'Node {node}' for node in G.nodes()]))

        fig.update_layout(showlegend=False, title='Graph mô phỏng', xaxis_title='Longitude', yaxis_title='Latitude')

        # Capture click events
        selected_points = plotly_events(fig)

        st.plotly_chart(fig)

# Cột bên phải: Hiển thị thông tin và hình ảnh của node khi nhấn
with col2:
    st.header("Thông tin Node")
    if map_type == "Graph mô phỏng" and selected_points:
        selected_node = int(selected_points[0].get('pointIndex', -1)) + 1  # +1 to match the node ID

        if selected_node in G.nodes:
            node_data = G.nodes[selected_node]
            st.write(f"**Địa chỉ**: {node_data['address']}")
            image_path = f"{node_data['folder']}/{node_data['filename']}"
            st.image(image_path)
        else:
            st.write("Không có node nào được chọn!")

# Chức năng tìm kiếm số lượng node thuộc một con đường nào đó
st.header("Tìm kiếm theo tên đường")
search_road = st.text_input("Nhập tên đường")

if search_road:
    matching_nodes = [node_id for node_id in G.nodes if search_road.lower() in G.nodes[node_id]['address'].lower()]
    node_count = len(matching_nodes)
    st.write(f"Số lượng node trên đường '{search_road}': {node_count}")

    if node_count > 0:
        if map_type == "Bản đồ thật":
            m_search = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=15)
            for node_id in matching_nodes:
                pos = G.nodes[node_id]['pos']
                address = G.nodes[node_id]['address']
                folium.Marker(
                    [pos[1], pos[0]],  # Latitude, Longitude
                    popup=f"Node {node_id}: {address}",
                    tooltip=address,
                    icon=folium.Icon(color='red')
                ).add_to(m_search)
            folium_static(m_search)

        elif map_type == "Graph mô phỏng":
            # Highlight the matching nodes in red
            for node_id in G.nodes:
                if node_id in matching_nodes:
                    fig.add_trace(go.Scatter(
                        x=[pos[node_id][0]], y=[pos[node_id][1]],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        text=[f'Node {node_id}'],
                        textposition='top center'
                    ))

            st.plotly_chart(fig)
