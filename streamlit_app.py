"""Website 3D tương tác để khám phá DBSCAN.

Chạy bằng: streamlit run streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from dbscan_demo import DemoData, embed_in_3d, make_data


st.set_page_config(page_title="DBSCAN 3D Visualizer", page_icon="●", layout="wide")


def neighbor_distances(points: np.ndarray, min_samples: int) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=min_samples).fit(points)
    distances, _ = model.kneighbors(points)
    return np.sort(distances[:, -1])


def get_uploaded_data(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str | None,
    seed: int,
    dimensions: int,
) -> DemoData:
    columns = [x_column, y_column] + ([z_column] if dimensions == 3 and z_column else [])
    selected = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(selected) < 3:
        raise ValueError("Cần ít nhất 3 hàng số hợp lệ ở các cột đã chọn.")
    points = selected.to_numpy()
    if dimensions == 3 and z_column is None:
        points = embed_in_3d(points, seed)
    return DemoData(
        StandardScaler().fit_transform(points),
        f"Dữ liệu tải lên ({len(points)} điểm, {dimensions} đặc trưng)",
    )


def cluster_figure(points: np.ndarray, labels: np.ndarray, core_mask: np.ndarray) -> go.Figure:
    figure = go.Figure()
    palette = ["#00B8D9", "#FF5630", "#6554C0", "#36B37E", "#FFAB00", "#172B4D"]

    if points.shape[1] == 2:
        for position, label in enumerate(sorted(set(labels))):
            mask = labels == label
            if label == -1:
                figure.add_trace(
                    go.Scatter(
                        x=points[mask, 0], y=points[mask, 1], mode="markers",
                        name="Noise (-1)", marker={"size": 8, "color": "#1f1f1f", "symbol": "x"},
                    )
                )
                continue
            color = palette[position % len(palette)]
            border = mask & ~core_mask
            if border.any():
                figure.add_trace(
                    go.Scatter(
                        x=points[border, 0], y=points[border, 1], mode="markers",
                        name=f"Cụm {label} — border",
                        marker={"size": 7, "color": color, "opacity": 0.45},
                    )
                )
            core = mask & core_mask
            figure.add_trace(
                go.Scatter(
                    x=points[core, 0], y=points[core, 1], mode="markers",
                    name=f"Cụm {label} — core",
                    marker={"size": 10, "color": color, "line": {"color": "white", "width": 1}},
                )
            )
        figure.update_layout(
            title="Kết quả DBSCAN trong không gian 2D",
            height=650,
            margin={"l": 0, "r": 0, "b": 0, "t": 45},
            xaxis_title="Đặc trưng X (chuẩn hoá)",
            yaxis_title="Đặc trưng Y (chuẩn hoá)",
            legend={"orientation": "h", "y": -0.16},
        )
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return figure

    for position, label in enumerate(sorted(set(labels))):
        mask = labels == label
        if label == -1:
            figure.add_trace(
                go.Scatter3d(
                    x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
                    mode="markers", name="Noise (-1)",
                    marker={"size": 4, "color": "#1f1f1f", "symbol": "x"},
                )
            )
            continue

        color = palette[position % len(palette)]
        border = mask & ~core_mask
        if border.any():
            figure.add_trace(
                go.Scatter3d(
                    x=points[border, 0], y=points[border, 1], z=points[border, 2],
                    mode="markers", name=f"Cụm {label} — border",
                    marker={"size": 4, "color": color, "opacity": 0.45},
                )
            )
        core = mask & core_mask
        figure.add_trace(
            go.Scatter3d(
                x=points[core, 0], y=points[core, 1], z=points[core, 2],
                mode="markers", name=f"Cụm {label} — core",
                marker={"size": 6, "color": color, "line": {"color": "white", "width": 1}},
            )
        )

    figure.update_layout(
        title="Kết quả DBSCAN trong không gian 3D",
        height=650,
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
        scene={
            "xaxis_title": "Đặc trưng X (chuẩn hoá)",
            "yaxis_title": "Đặc trưng Y (chuẩn hoá)",
            "zaxis_title": "Đặc trưng Z (chuẩn hoá)",
            "aspectmode": "cube",
        },
        legend={"orientation": "h", "y": -0.12},
    )
    return figure


def distance_figure(distances: np.ndarray, eps: float) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(y=distances, mode="lines", name="k-distance", line={"color": "#236192"})
    )
    figure.add_hline(
        y=eps, line_dash="dash", line_color="#C23B22", annotation_text=f"eps = {eps:.2f}"
    )
    figure.update_layout(
        title="k-distance plot",
        xaxis_title="Điểm (đã sắp xếp)",
        yaxis_title="Khoảng cách tới láng giềng thứ k",
        height=350,
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )
    return figure


st.title("DBSCAN 2D / 3D Visualizer")
st.caption("Ở 3D: kéo để xoay, lăn chuột để phóng to. Rê chuột lên điểm để xem toạ độ.")

with st.sidebar:
    st.header("Thiết lập")
    source = st.radio("Nguồn dữ liệu", ("Dữ liệu mẫu", "Tải CSV"))
    display_mode = st.radio("Chế độ phân cụm và hiển thị", ("2D", "3D"), horizontal=True)
    eps = st.slider("eps — bán kính lân cận", 0.05, 2.00, 0.35, 0.01)
    min_samples = st.slider("min_samples", 2, 30, 5)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

dimensions = int(display_mode[0])

if source == "Dữ liệu mẫu":
    dataset = st.selectbox(
        "Chọn bộ dữ liệu",
        {
            "Make Moons — make_moons.csv": "moons",
            "Make Circles — make-circles.ipynb": "circles",
            "Blobs — cụm cầu có điểm nhiễu": "blobs",
        },
    )
    try:
        data = make_data(dataset, int(seed), dimensions=dimensions)
    except (FileNotFoundError, KeyError, ValueError) as error:
        st.error(f"Không thể nạp bộ dữ liệu: {error}")
        st.stop()
else:
    uploaded_file = st.file_uploader("Tải tệp CSV", type="csv")
    if uploaded_file is None:
        st.info("Tải CSV có ít nhất hai cột số để bắt đầu.")
        st.stop()
    preview = pd.read_csv(uploaded_file)
    numeric_columns = preview.select_dtypes(include="number").columns.tolist()
    if len(numeric_columns) < 2:
        st.error("CSV cần có ít nhất hai cột mang kiểu số.")
        st.stop()
    first_col, second_col, third_col = st.columns(3)
    with first_col:
        x_column = st.selectbox("Cột X", numeric_columns, index=0)
    with second_col:
        y_column = st.selectbox("Cột Y", numeric_columns, index=1)
    if dimensions == 3:
        with third_col:
            z_column = st.selectbox("Cột Z", ["Tạo Z từ X, Y"] + numeric_columns, index=0)
    else:
        z_column = "Tạo Z từ X, Y"
    if x_column == y_column or (dimensions == 3 and z_column in (x_column, y_column)):
        st.warning("Hãy chọn các cột X, Y, Z khác nhau.")
        st.stop()
    actual_z = None if z_column == "Tạo Z từ X, Y" else z_column
    try:
        data = get_uploaded_data(
            preview, x_column, y_column, actual_z, int(seed), dimensions
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()
    st.dataframe(preview.head(10), use_container_width=True)

model = DBSCAN(eps=eps, min_samples=min_samples)
labels = model.fit_predict(data.points)
core_mask = np.zeros(labels.size, dtype=bool)
core_mask[model.core_sample_indices_] = True
n_clusters = len(set(labels) - {-1})
n_noise = int(np.sum(labels == -1))
valid = labels != -1

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Số cụm", n_clusters)
metric_b.metric("Điểm nhiễu", f"{n_noise} / {len(labels)}")
if n_clusters >= 2 and len(set(labels[valid])) >= 2:
    metric_c.metric("Silhouette (bỏ noise)", f"{silhouette_score(data.points[valid], labels[valid]):.3f}")
else:
    metric_c.metric("Silhouette (bỏ noise)", "Không xác định")

st.plotly_chart(cluster_figure(data.points, labels, core_mask), use_container_width=True)
st.plotly_chart(
    distance_figure(neighbor_distances(data.points, min_samples), eps), use_container_width=True
)

with st.expander("Cách đọc kết quả"):
    st.markdown(
        """
        - **Core point**: điểm lớn, có đủ `min_samples` điểm trong bán kính `eps`.
        - **Border point**: thuộc lân cận của core point nhưng chưa đủ mật độ để tự là core point.
        - **Noise**: dấu `x` đen, nhãn `-1`; không thuộc cụm nào.
        - Make Moons/Make Circles/Blobs vốn là 2D nên website tạo `Z` bằng phép biến đổi phi tuyến của `X`, `Y`; không dùng nhãn thật và DBSCAN chạy trên cả ba chiều.
        """
    )

st.markdown("### Ba bộ dữ liệu trong demo")
st.markdown(
    """
    1. **Make Moons**: đọc từ `make_moons.csv`; hai cụm cong lồng nhau.
    2. **Make Circles**: dùng cấu hình từ `make-circles.ipynb` — 1.000 điểm, `noise=0.03`.
    3. **Blobs**: ba cụm hình cầu cùng các điểm rải rác để quan sát nhãn noise (`-1`).
    """
)
