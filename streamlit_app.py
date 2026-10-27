"""Website tương tác để khám phá DBSCAN, OPTICS và HDBSCAN.

Chạy bằng: streamlit run streamlit_app.py
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, HDBSCAN

from dbscan_demo import make_data
from cluster_display import align_for_display, axis_limits
from clustering_metrics import calculate_metrics
from density_examples import DENSITY_EXAMPLES, label_recovery
from optics_demo import EXTRACTION_XI, OpticsStructure, compute_structure, extract_labels
from reachability_plot import reachability_bars


st.set_page_config(page_title="DBSCAN / OPTICS / HDBSCAN", page_icon="●", layout="wide")

PALETTE = ["#00B8D9", "#FF5630", "#6554C0", "#36B37E", "#FFAB00", "#172B4D"]


def cluster_color(label: int) -> str:
    return PALETTE[label] if label < len(PALETTE) else f"hsl({(label * 137.508) % 360:.3f},65%,42%)"


def synchronize_axes(figure: go.Figure, points: np.ndarray) -> go.Figure:
    limits = axis_limits(points)
    if points.shape[1] == 2:
        figure.update_xaxes(range=limits[0], autorange=False, constrain="domain")
        figure.update_yaxes(range=limits[1], autorange=False, scaleanchor="x", scaleratio=1, constrain="domain")
    else:
        figure.update_layout(scene={
            f"{axis}axis": {"range": limits[i].tolist(), "autorange": False}
            for i, axis in enumerate("xyz")
        })
    return figure


def cluster_figure(points: np.ndarray, labels: np.ndarray,
                   core_mask: np.ndarray | None = None, algorithm: str = "DBSCAN") -> go.Figure:
    figure = go.Figure()
    if core_mask is None:
        for label in sorted(set(labels)):
            mask = labels == label
            marker = {"size": 7 if points.shape[1] == 2 else 4,
                      "color": "#1f1f1f" if label == -1 else cluster_color(int(label))}
            if label == -1:
                marker["symbol"] = "x"
            options = dict(x=points[mask, 0], y=points[mask, 1], mode="markers",
                           name="Noise (-1)" if label == -1 else f"Cụm {label}", marker=marker)
            if points.shape[1] == 3:
                figure.add_trace(go.Scatter3d(z=points[mask, 2], **options))
            else:
                figure.add_trace(go.Scatter(**options))
        figure.update_layout(
            title=f"{algorithm} trong không gian {points.shape[1]}D", height=650,
            margin={"l": 0, "r": 0, "b": 0, "t": 45},
            xaxis_title="Đặc trưng X (chuẩn hoá)", yaxis_title="Đặc trưng Y (chuẩn hoá)",
            scene={"xaxis_title": "X (chuẩn hoá)", "yaxis_title": "Y (chuẩn hoá)",
                   "zaxis_title": "Z (chuẩn hoá)", "aspectmode": "cube"},
            legend={"orientation": "h", "y": -0.16},
        )
        if points.shape[1] == 2:
            figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return synchronize_axes(figure, points)

    if points.shape[1] == 2:
        for label in sorted(set(labels)):
            mask = labels == label
            if label == -1:
                figure.add_trace(
                    go.Scatter(
                        x=points[mask, 0], y=points[mask, 1], mode="markers",
                        name="Noise (-1)", marker={"size": 8, "color": "#1f1f1f", "symbol": "x"},
                    )
                )
                continue
            color = cluster_color(int(label))
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
        return synchronize_axes(figure, points)

    for label in sorted(set(labels)):
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

        color = cluster_color(int(label))
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
    return synchronize_axes(figure, points)


def ordering_figure(points: np.ndarray, model: OpticsStructure) -> go.Figure:
    rank = np.empty(len(points), dtype=int)
    rank[model.ordering] = np.arange(len(points))
    options = dict(
        x=points[:, 0], y=points[:, 1], mode="markers", name="Thứ tự duyệt",
        customdata=np.column_stack((np.arange(len(points)), rank)),
        hovertemplate="Điểm gốc: %{customdata[0]}<br>Thứ tự OPTICS: %{customdata[1]}<extra></extra>",
        marker={"size": 6 if points.shape[1] == 2 else 4, "color": rank,
                "colorscale": "Viridis", "colorbar": {"title": "Thứ tự OPTICS"}},
    )
    trace = go.Scatter3d(z=points[:, 2], **options) if points.shape[1] == 3 else go.Scatter(**options)
    figure = go.Figure(trace)
    figure.update_layout(
        title="OPTICS — thứ tự duyệt (không phải nhãn cụm)", height=650,
        xaxis_title="X (chuẩn hoá)", yaxis_title="Y (chuẩn hoá)",
        scene={"xaxis_title": "X (chuẩn hoá)", "yaxis_title": "Y (chuẩn hoá)",
               "zaxis_title": "Z (chuẩn hoá)", "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "b": 0, "t": 60},
    )
    if points.shape[1] == 2:
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


@st.cache_data(show_spinner=False)
def fit_dbscan(points: np.ndarray, eps: float, min_samples: int):
    started = perf_counter()
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return model, (perf_counter() - started) * 1000


@st.cache_data(show_spinner=False)
def fit_optics_graph(points: np.ndarray, min_samples: int):
    started = perf_counter()
    model = compute_structure(points, min_samples)
    return model, (perf_counter() - started) * 1000


@st.cache_data(show_spinner=False)
def fit_optics(points: np.ndarray, min_samples: int, method: str,
               xi: float, cut_eps: float, min_cluster_size: float):
    model, graph_ms = fit_optics_graph(points, min_samples)
    started = perf_counter()
    labels = extract_labels(model, min_samples, method=method, xi=xi,
                            eps=cut_eps, min_cluster_size=min_cluster_size)
    return model, labels, graph_ms + (perf_counter() - started) * 1000


@st.cache_data(show_spinner=False)
def fit_hdbscan(points: np.ndarray, min_samples: int, min_cluster_size: int):
    started = perf_counter()
    model = HDBSCAN(
        min_samples=min_samples, min_cluster_size=min_cluster_size,
        metric="euclidean", cluster_selection_method="eom",
        cluster_selection_epsilon=0.0, copy=True,
    ).fit(points)
    return model, (perf_counter() - started) * 1000


@st.cache_data(show_spinner=False)
def clustering_metrics(points: np.ndarray, labels: np.ndarray) -> dict:
    return calculate_metrics(points, labels)


def show_metrics(metrics: dict, total: int) -> None:
    a, b, c = st.columns(3)
    a.metric("Số cụm", metrics["Số cụm"])
    b.metric("Điểm nhiễu", f'{metrics["Điểm nhiễu"]} / {total}')
    score = metrics["Silhouette (bỏ noise)"]
    c.metric("Silhouette (bỏ noise)", "Không xác định" if score is None else f"{score:.3f}")


def hdbscan_figure(points: np.ndarray, labels: np.ndarray,
                   min_samples: int, min_cluster_size: int) -> go.Figure:
    figure = cluster_figure(points, labels, algorithm="HDBSCAN")
    figure.update_layout(
        title=f"HDBSCAN<br><sup>MinPts={min_samples} · min_cluster_size={min_cluster_size}</sup>",
        margin={"t": 70},
    )
    return figure


def apply_density_preset(name: str) -> None:
    preset = DENSITY_EXAMPLES[name]
    st.session_state.update({
        "display_mode": "2D", "seed": 42, "min_samples": preset["min_samples"],
        "dbscan_eps": preset["eps"],
        "hdbscan_min_cluster_size": 20,
        "optics_method": "xi", "optics_xi": 0.10, "optics_min_cluster_size": 0.10,
    })


@st.cache_data(show_spinner=False)
def sweep_dbscan(points: np.ndarray, true_labels: np.ndarray, min_samples: int) -> pd.DataFrame:
    rows = []
    for eps_value in np.round(np.arange(0.02, 2.001, 0.02), 2):
        labels = DBSCAN(eps=float(eps_value), min_samples=min_samples).fit_predict(points)
        rows.append({"eps": float(eps_value), "Số cụm": len(set(labels) - {-1}),
                     "Tỷ lệ nhiễu (%)": round(100 * float((labels == -1).mean()), 2),
                     "Khớp nhãn gốc (%)": 100 * label_recovery(true_labels, labels)})
    return pd.DataFrame(rows)


st.title("DBSCAN / OPTICS / HDBSCAN — 2D / 3D")
st.caption("Ở 3D: kéo để xoay, lăn chuột để phóng to. Rê chuột lên điểm để xem toạ độ.")

with st.sidebar:
    st.header("Thiết lập")
    display_mode = st.radio("Chế độ phân cụm và hiển thị", ("2D", "3D"), horizontal=True, key="display_mode")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="seed")

dimensions = int(display_mode[0])

datasets = {
    "moons": "Make Moons — make_moons.csv",
    "blobs": "Blobs — cụm cầu có điểm nhiễu",
    "six_density": DENSITY_EXAMPLES["six_density"]["name"],
}
dataset = st.selectbox("Chọn bộ dữ liệu", list(datasets), format_func=datasets.get)
dataset_purpose = {
    "moons": "Mục tiêu: quan sát cụm cong, khả năng nối mật độ và độ nhạy với eps/MinPts. Không mặc định thuật toán nào tốt nhất.",
    "blobs": "Mục tiêu: quan sát các cụm dạng cầu và điểm nhiễu; đối chiếu cách DBSCAN, OPTICS và HDBSCAN giữ hoặc loại các điểm ở rìa cụm.",
    "six_density": "Mục tiêu: khảo sát nhiều mức mật độ, các thung lũng reachability và kết quả chọn cụm ổn định. Các nhóm sinh có thể chồng lấn; không bắt buộc mọi thuật toán tìm đúng 6 cụm.",
}
st.caption(dataset_purpose[dataset])
if dataset in DENSITY_EXAMPLES:
    st.info(DENSITY_EXAMPLES[dataset]["description"])
    st.button("Áp dụng cấu hình thuyết trình (2D, seed 42)",
              on_click=apply_density_preset, args=(dataset,))
    st.caption("Cấu hình đã kiểm tra sau StandardScaler ở 2D, seed 42. Đổi seed, số chiều hoặc tham số có thể thay đổi kết quả.")
    if dataset == "six_density":
        st.markdown("Nguồn cấu trúc dữ liệu: [ví dụ OPTICS của scikit-learn](https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html). Demo dùng seed và chuẩn hoá riêng.")
try:
    data = make_data(dataset, int(seed), dimensions=dimensions)
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"Không thể nạp bộ dữ liệu: {error}")
    st.stop()

with st.sidebar:
    min_samples = st.slider("MinPts (min_samples) — dùng chung", 2, min(100, len(data.points)), 5, key="min_samples")
    st.caption("Ba thuật toán dùng chung dữ liệu chuẩn hoá, khoảng cách Euclidean và min_samples (tính cả chính điểm đang xét).")

dbscan_tab, optics_tab, hdbscan_tab, compare_tab = st.tabs([
    "DBSCAN", "OPTICS", "HDBSCAN", "So sánh 3 thuật toán",
])
with dbscan_tab:
    eps = st.slider("DBSCAN eps — bán kính lân cận", 0.01, 3.00, 0.35, 0.01, key="dbscan_eps")
with optics_tab:
    st.info(f"Bước 1: dựng ordering OPTICS với MinPts={min_samples}, max_eps=∞ và Euclidean. Bước 2 bên dưới chỉ trích cụm, không dựng lại ordering.")
    optics_method = st.radio("Bước 2 — cách trích cụm OPTICS", ["xi", "dbscan"],
                            format_func=lambda value: "Xi — theo độ dốc" if value == "xi" else "Cắt ngang ε′",
                            horizontal=True, key="optics_method")
    optics_xi, optics_cut, optics_size = EXTRACTION_XI, 1.5, .10
    if optics_method == "xi":
        optics_xi = st.slider("ξ — ngưỡng độ dốc khi trích cụm", .01, .20, .10, .01, key="optics_xi")
        optics_size = st.slider("Kích thước cụm tối thiểu khi trích Xi — tỷ lệ số điểm",
                                .01, .50, .10, .01, key="optics_min_cluster_size")
        st.caption("ξ mặc định 0.1 như trang 47. Kích thước cụm là tiêu chí trích nhãn, không phải tham số dựng ordering.")
    else:
        optics_cut = st.slider("ε′ — mức cắt OPTICS (không phải eps DBSCAN)", .01, 3., .9, .01, key="optics_cut")
        st.caption("Như trang 45–46: điểm vượt ε′ mở cụm mới nếu core-distance ≤ ε′, nếu không thì là noise. Gần DBSCAN, không đảm bảo giống hệt ở border.")
optics_name = "OPTICS + Xi" if optics_method == "xi" else "OPTICS + cắt ε′"
with hdbscan_tab:
    hdbscan_min_cluster_size = st.slider(
        "HDBSCAN min_cluster_size — số điểm tối thiểu của cụm",
        2, min(500, len(data.points)), 20, key="hdbscan_min_cluster_size",
    )
    st.caption("Tham số là ngưỡng kích thước, không phải số cụm mong muốn. Nhiều giá trị liên tiếp có thể cho cùng một phân hoạch.")
    st.info(f"HDBSCAN dùng MinPts = {min_samples} ở thanh bên và min_cluster_size = {hdbscan_min_cluster_size} điểm. Không cần chọn eps như DBSCAN.")
    st.caption("Xây dựng phân cấp mật độ rồi chọn các cụm ổn định bằng EOM. Cụm nhỏ có thể bị loại thành noise. min_cluster_size ở đây là số điểm, không phải tỷ lệ; cluster_selection_epsilon cố định bằng 0.")

with st.spinner("Đang chạy DBSCAN, OPTICS và HDBSCAN…"):
    dbscan, dbscan_ms = fit_dbscan(data.points, eps, min_samples)
    optics, optics_labels, optics_ms = fit_optics(
        data.points, min_samples, optics_method, optics_xi, optics_cut, optics_size,
    )
    hdbscan, hdbscan_ms = fit_hdbscan(data.points, min_samples, hdbscan_min_cluster_size)
    db_metrics = clustering_metrics(data.points, dbscan.labels_)
    op_metrics = clustering_metrics(data.points, optics_labels)
    hdb_metrics = clustering_metrics(data.points, hdbscan.labels_)
if data.true_labels is not None:
    db_metrics = {**db_metrics, "Khớp nhãn gốc (%)": round(100 * label_recovery(data.true_labels, dbscan.labels_), 2)}
    op_metrics = {**op_metrics, "Khớp nhãn gốc (%)": round(100 * label_recovery(data.true_labels, optics_labels), 2)}
    hdb_metrics = {**hdb_metrics, "Khớp nhãn gốc (%)": round(100 * label_recovery(data.true_labels, hdbscan.labels_), 2)}
core_mask = np.zeros(len(data.points), dtype=bool)
core_mask[dbscan.core_sample_indices_] = True
db_display, op_display = align_for_display(hdbscan.labels_, dbscan.labels_, optics_labels)

# Compare only runs on the same data / MinPts, ignoring arbitrary label numbers.
hdb_context = (dataset, dimensions, int(seed), min_samples)
previous_hdb = st.session_state.get("hdbscan_previous_result")
hdb_feedback = ""
if previous_hdb is not None and previous_hdb["context"] == hdb_context:
    if previous_hdb["size"] != hdbscan_min_cluster_size:
        matched_labels, = align_for_display(previous_hdb["labels"], hdbscan.labels_)
        changed = int(np.count_nonzero(matched_labels != previous_hdb["labels"]))
        change_text = f'{previous_hdb["size"]} → {hdbscan_min_cluster_size}'
        if changed == 0:
            hdb_feedback = (f"Đã áp dụng min_cluster_size: {change_text}. Phân hoạch không đổi: "
                            f'{hdb_metrics["Số cụm"]} cụm, {hdb_metrics["Điểm nhiễu"]} điểm noise. '
                            "EOM vẫn chọn cùng các cụm ổn định; đây không phải lỗi vẽ lại.")
        else:
            hdb_feedback = (f"Đã áp dụng min_cluster_size: {change_text}. "
                            f"Có {changed} điểm thay đổi nhóm hoặc trạng thái noise sau khi ghép nhãn tương ứng.")
    else:
        hdb_feedback = previous_hdb["feedback"]
st.session_state["hdbscan_previous_result"] = {
    "context": hdb_context, "size": hdbscan_min_cluster_size,
    "labels": hdbscan.labels_.copy(), "feedback": hdb_feedback,
}

with dbscan_tab:
    show_metrics(db_metrics, len(data.points))
    st.plotly_chart(cluster_figure(data.points, db_display, core_mask),
                    use_container_width=True, key="dbscan_demo")

with hdbscan_tab:
    show_metrics(hdb_metrics, len(data.points))
    if hdb_feedback:
        st.info(hdb_feedback)
    st.plotly_chart(hdbscan_figure(data.points, hdbscan.labels_, min_samples, hdbscan_min_cluster_size),
                    use_container_width=True, key="hdbscan_demo")
    st.caption("Màu biểu diễn nhãn cụm; dấu x đen là noise (-1). Không chia core/border theo một eps cố định như DBSCAN.")

with optics_tab:
    show_metrics(op_metrics, len(data.points))
    st.plotly_chart(cluster_figure(data.points, op_display, algorithm=optics_name),
                    use_container_width=True, key="optics_demo")
    st.markdown("**Đọc biểu đồ như trong slide:** mỗi cột là một điểm theo thứ tự OPTICS. "
                "Một dải cột thấp tạo thành **thung lũng**, gợi ý vùng dày; cột tăng cao thể hiện "
                "chuyển sang vùng thưa hơn hoặc ranh giới giữa các vùng.")
    st.plotly_chart(reachability_bars(
        optics, op_display, {int(label): cluster_color(int(label)) for label in set(op_display) if label >= 0},
        cut_eps=optics_cut if optics_method == "dbscan" else None,
    ), use_container_width=True, key="reachability", config={"displaylogo": False, "scrollZoom": True})
    st.caption("Màu cột trùng với cụm trên biểu đồ điểm; cột đen là noise sau bước trích nhãn. "
               "Dữ liệu không bị gom hay làm mượt. "
               "Ký hiệu ∞ ở mép trên không phải độ cao thật và không mặc nhiên là noise. "
               "Một thung lũng không tự động tương ứng một cụm; nhãn còn phụ thuộc cách trích đã chọn.")
    with st.expander("Xem bảng kết quả OPTICS theo thứ tự duyệt"):
        st.caption("Predecessor là chỉ số điểm đã dẫn tới điểm hiện tại; -1 đánh dấu điểm khởi đầu, không phải nhãn noise.")
        order = optics.ordering
        st.dataframe(pd.DataFrame({
            "Thứ tự OPTICS": np.arange(len(order)), "Chỉ số điểm gốc": order,
            "Core-distance": optics.core_distances[order],
            "Reachability": optics.reachability[order], "Predecessor": optics.predecessor[order],
            "Nhãn cụm (hiển thị)": op_display[order],
        }), hide_index=True, use_container_width=True)

with compare_tab:
    st.caption(f"Cùng {len(data.points)} điểm, {dimensions} chiều, StandardScaler, min_samples={min_samples}.")
    extraction_text = f"xi={optics_xi:.2f}, kích thước cụm≥{optics_size:.0%}" if optics_method == "xi" else f"ε′={optics_cut:.2f}"
    st.caption(f"DBSCAN: eps={eps:.2f} · {optics_name}: {extraction_text} · HDBSCAN: min_cluster_size={hdbscan_min_cluster_size}, EOM.")
    left, middle, right = st.columns(3)
    with left:
        st.plotly_chart(cluster_figure(data.points, db_display, algorithm="DBSCAN"),
                        use_container_width=True, key="compare_dbscan")
    with middle:
        st.plotly_chart(cluster_figure(data.points, op_display, algorithm=optics_name),
                        use_container_width=True, key="compare_optics")
    with right:
        st.plotly_chart(hdbscan_figure(data.points, hdbscan.labels_, min_samples, hdbscan_min_cluster_size),
                        use_container_width=True, key="compare_hdbscan")
    st.caption("Ba biểu đồ dùng cùng phạm vi trục, kích thước điểm và ký hiệu noise x đen. Màu được ghép theo độ chồng lấp với cụm HDBSCAN để dễ đối chiếu; chỉ đổi tên nhãn hiển thị, không đổi thành viên cụm hay chỉ số. Cùng màu không có nghĩa hai cụm hoàn toàn giống nhau.")
    comparison = pd.DataFrame([
        {"Thuật toán": "DBSCAN", "Đầu ra": "Nhãn cụm và noise", **db_metrics,
         "Thời gian tham khảo (ms)": round(dbscan_ms, 2)},
        {"Thuật toán": optics_name, "Đầu ra": "Nhãn cụm và noise (" + extraction_text + ")", **op_metrics,
         "Thời gian tham khảo (ms)": round(optics_ms, 2)},
        {"Thuật toán": "HDBSCAN", "Đầu ra": "Nhãn cụm và noise (EOM)", **hdb_metrics,
         "Thời gian tham khảo (ms)": round(hdbscan_ms, 2)},
    ])
    st.dataframe(comparison.style.format({
        "Silhouette (bỏ noise)": "{:.4f}", "Tỷ lệ nhiễu (%)": "{:.2f}",
        "Thời gian tham khảo (ms)": "{:.2f}", "Khớp nhãn gốc (%)": "{:.2f}",
    }, na_rep="Không xác định"), hide_index=True, use_container_width=True)
    st.caption("Silhouette chỉ xác định khi 2 ≤ số cụm < số điểm không nhiễu; một cụm hoặc toàn noise hiển thị ‘Không xác định’, không phải 0. Chỉ số tính trên dữ liệu chuẩn hoá, trước khi ghép màu.")
    st.caption("Thời gian tham khảo: DBSCAN/HDBSCAN đo fit bằng perf_counter; OPTICS cộng thời gian dựng cấu trúc và trích cụm, hai giai đoạn có thể được đo ở các lần khác nhau do cache. Không gồm vẽ biểu đồ, ghép màu, tính chỉ số; không phải thời gian chờ hiện tại hoặc benchmark trung bình nhiều lần.")
    st.info("Cả ba kết quả có nhãn cụm/noise. OPTICS cần bước trích nhãn từ ordering; đổi tiêu chí trích không thay đổi cấu trúc gốc. Đọc Silhouette cùng tỷ lệ noise vì tập điểm được giữ lại có thể khác nhau.")
    if data.true_labels is not None:
        st.markdown("#### Đối chiếu ba kết quả với các nhóm gốc")
        st.caption("Khớp nhãn gốc: ghép cụm với nhóm sinh dữ liệu theo ánh xạ một-một tối ưu, chia số điểm khớp cho toàn bộ số điểm. Noise tính là không khớp. Nhãn gốc chỉ dùng đánh giá, không đưa vào thuật toán. Các phân bố chồng lấn không nhất thiết tạo các cụm tách biệt hoàn toàn.")
        with st.expander("Xem các nhóm gốc để đối chiếu"):
            st.plotly_chart(cluster_figure(data.points, data.true_labels, algorithm="Nhãn sinh dữ liệu"),
                            use_container_width=True, key="ground_truth")
        if st.checkbox("Quét 100 mức eps của DBSCAN để so sánh", key="show_sweep"):
            with st.spinner("Đang khảo sát eps=0.02…2.00, bước 0.02…"):
                sweep = sweep_dbscan(data.points, data.true_labels, min_samples)
            figure = go.Figure(go.Scatter(x=sweep["eps"], y=sweep["Khớp nhãn gốc (%)"],
                                         mode="lines+markers", name="DBSCAN"))
            figure.add_hline(y=hdb_metrics["Khớp nhãn gốc (%)"], line_color="#36B37E", line_dash="dash",
                             annotation_text="HDBSCAN — cấu hình hiện tại")
            figure.add_hline(y=op_metrics["Khớp nhãn gốc (%)"], line_color="#6554C0", line_dash="dot",
                             annotation_text=optics_name + " — cấu hình hiện tại", annotation_position="bottom right")
            figure.update_layout(title="Khớp nhãn gốc theo eps (cùng min_samples)",
                                 xaxis_title="DBSCAN eps sau chuẩn hoá", yaxis_title="Khớp nhãn gốc (%)",
                                 yaxis_range=[0, 105], height=380)
            st.plotly_chart(figure, use_container_width=True, key="eps_sweep")
            best = sweep.loc[sweep["Khớp nhãn gốc (%)"].idxmax()]
            st.write(f'DBSCAN tốt nhất trong 100 mức đã thử: {best["Khớp nhãn gốc (%)"]:.2f}% '
                     f'tại eps={best["eps"]:.2f}. HDBSCAN hiện tại: {hdb_metrics["Khớp nhãn gốc (%)"]:.2f}%. '
                     f'{optics_name} hiện tại: {op_metrics["Khớp nhãn gốc (%)"]:.2f}%.')
            st.caption("Khảo sát hữu hạn trong [0.02, 2.00], giữ nguyên min_samples; không phải tối ưu toàn bộ tham số. Kết quả có thể thay đổi theo cấu hình và dữ liệu.")
            st.dataframe(sweep, hide_index=True, use_container_width=True)

with st.expander("Cách đọc kết quả"):
    st.markdown(
        """
        - **Core point**: điểm lớn, có đủ `min_samples` điểm trong bán kính `eps`.
        - **Border point**: thuộc lân cận của core point nhưng chưa đủ mật độ để tự là core point.
        - **Noise**: dấu `x` đen, nhãn `-1`; không thuộc cụm nào.
        - HDBSCAN và OPTICS + Xi hiển thị cụm/noise, không chia core/border theo eps cố định. Tab so sánh dùng cùng kiểu điểm cho cả ba.
        - Dữ liệu mẫu vốn là 2D. Ở 3D, website tạo `Z` từ `X`, `Y`; cả ba thuật toán chạy trên cả ba chiều, không dùng nhãn thật để phân cụm.
        """
    )

st.markdown("### Ba bộ dữ liệu để trình bày")
st.markdown(
    """
    1. **Make Moons**: hai cụm cong lồng nhau — minh hoạ hình dạng phi cầu và ảnh hưởng của eps/MinPts.
    2. **Blobs**: ba cụm dạng cầu và các điểm rải rác — so sánh phân cụm và nhận diện noise.
    3. **6 vùng mật độ**: phỏng theo ví dụ scikit-learn — khảo sát reachability của OPTICS và chọn cụm ổn định của HDBSCAN trên dữ liệu phức tạp hơn.
    """
)
