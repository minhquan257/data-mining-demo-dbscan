"""Giao diện mô phỏng các ví dụ nhỏ trong slide, độc lập dữ liệu so sánh."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_samples

from cluster_display import align_for_display
from optics_demo import compute_structure, extract_labels
from slide_simulation import DBSCAN_POINTS, OPTICS_POINTS, dbscan_steps, optics_steps, density_hierarchy

COLORS = ["#236192", "#159783", "#7654ad", "#de7526", "#c5465e"]


def letters(indices):
    return ", ".join(chr(97 + int(i)) for i in indices) or "—"


def points_plot(points, labels=None, *, visited=None, core=None, queue=(),
                current=None, radius=None, title="Dữ liệu theo slide"):
    figure = go.Figure()
    names = [chr(97 + i) for i in range(len(points))]
    for i, (x, y) in enumerate(points):
        if visited is not None:
            if not visited[i]:
                color, symbol = ("#a9c4e6" if i in queue else "#c5cad3"), "circle"
            elif labels is not None and labels[i] == -1:
                color, symbol = "#222222", "x"
            elif core is not None:
                color, symbol = ("#236192" if core[i] else "#de7526"), "circle"
            else:
                color, symbol = "#236192", "circle"
        elif labels is not None:
            color = "#222222" if labels[i] == -1 else COLORS[int(labels[i]) % len(COLORS)]
            symbol = "x" if labels[i] == -1 else "circle"
        else:
            color, symbol = "#236192", "circle"
        figure.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[names[i]], textposition="top center",
            marker={"color": color, "size": 14 if i == current else 10, "symbol": symbol},
            name=names[i], showlegend=False,
            hovertemplate=f"Điểm {names[i]}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
        ))
    if current is not None and radius is not None and np.isfinite(radius):
        x, y = points[current]
        figure.add_shape(type="circle", x0=x-radius, x1=x+radius, y0=y-radius, y1=y+radius,
                         line={"color": "#ff5630", "dash": "dash"})
    figure.update_layout(title=title, height=410, margin={"l": 15, "r": 15, "t": 45, "b": 20},
                         xaxis_title="X (đơn vị gốc)", yaxis_title="Y (đơn vị gốc)")
    figure.update_xaxes(range=[-.9, 6.3])
    figure.update_yaxes(range=[-2.5, 1.2], scaleanchor="x", scaleratio=1)
    return figure


def _move(key, delta, maximum):
    st.session_state[key] = max(0, min(maximum, st.session_state.get(key, 0) + delta))


def _set(key, value):
    st.session_state[key] = value


def step_controls(key, maximum):
    step = st.slider("Bước xử lý", 0, maximum, 0, key=key)
    left, right = st.columns(2)
    left.button("← Bước trước", key=key + "_prev", on_click=_move, args=(key, -1, maximum), disabled=step == 0)
    right.button("Bước tiếp →", key=key + "_next", on_click=_move, args=(key, 1, maximum), disabled=step == maximum)
    return step


def reach_plot(graph, *, frame=None, labels=None, cut=None, xi=None):
    order = graph.ordering if frame is None else np.array(frame["order"], dtype=int)
    values = graph.reachability[order] if frame is None else frame["reach"][order]
    finite = np.isfinite(values)
    top = max(float(graph.reachability[np.isfinite(graph.reachability)].max()) * 1.2, 1.)
    colors = [COLORS[int(labels[i]) % len(COLORS)] if labels is not None and labels[i] >= 0
              else ("#222222" if labels is not None else "#638cbe") for i in order]
    figure = go.Figure(go.Bar(
        x=np.arange(len(order)), y=np.where(finite, values, top), marker_color=colors,
        text=[f"{v:.2f}" if np.isfinite(v) else "∞" for v in values], textposition="outside",
        customdata=np.array([chr(97 + int(i)) for i in order]),
        hovertemplate="Điểm %{customdata}<br>Reachability: %{text}<extra></extra>",
    ))
    figure.update_xaxes(range=[-.7, len(graph.ordering)-.3], tickmode="array",
                        tickvals=np.arange(len(order)), ticktext=[chr(97+int(i)) for i in order])
    figure.update_yaxes(range=[0, top*1.2], title="Reachability (∞ chỉ là ký hiệu)")
    if cut is not None:
        figure.add_hline(y=cut, line_color="#de7526", line_dash="dash", annotation_text=f"ε′={cut:.2f}")
    if xi is not None:
        for i in range(len(values)-1):
            if not (finite[i] and finite[i+1]):
                continue
            if values[i+1] <= (1-xi)*values[i]:
                color = "#159783"
            elif values[i] <= (1-xi)*values[i+1]:
                color = "#de7526"
            else:
                continue
            figure.add_vrect(x0=i-.45, x1=i+1.45, fillcolor=color, opacity=.1, line_width=0, layer="below")
    figure.update_layout(title="Reachability theo ordering", height=350, showlegend=False,
                         margin={"l": 20, "r": 20, "t": 45, "b": 25})
    return figure


def hierarchy_plot(hierarchy, eps=None, selected=False):
    figure = go.Figure()
    top = max(2.5, float(hierarchy.linkage[-1, 2]) * 1.15)
    for cid, node in hierarchy.nodes.items():
        birth = top if node["birth"] == 0 else 1/node["birth"]
        death = 1/node["death"]
        x0, x1 = min(node["members"])-.35, max(node["members"])+.35
        active = selected and cid in hierarchy.selected
        color = COLORS[sorted(hierarchy.selected).index(cid) % len(COLORS)] if active else "#638cbe"
        figure.add_shape(type="rect", x0=x0, x1=x1, y0=death, y1=birth,
                         fillcolor=color, opacity=.65 if active else .25, line={"width": 1, "color": color})
        figure.add_trace(go.Scatter(x=[(x0+x1)/2], y=[np.sqrt(birth*death)], mode="text",
                                   text=[("Toàn bộ" if cid == 0 else f"C{cid}") + (" ✓" if active else "")],
                                   customdata=[node["stability"]], showlegend=False,
                                   hovertemplate="%{text}<br>Stability=%{customdata:.4f}<extra></extra>"))
    if eps is not None:
        figure.add_hline(y=eps, line_color="#de7526", line_dash="dash", annotation_text=f"ε={eps}")
    figure.update_layout(title="Cây cụm rút gọn — trục ε (log)", height=400,
                         xaxis={"range": [-1, 16], "visible": False},
                         yaxis={"type": "log", "title": "ε = 1/λ", "range": [np.log10(.2), np.log10(top*1.1)]},
                         margin={"l": 40, "r": 20, "t": 45, "b": 20})
    return figure


@st.cache_data(show_spinner=False)
def _db_trace(eps, min_samples):
    return dbscan_steps(DBSCAN_POINTS, eps, min_samples)


@st.cache_data(show_spinner=False)
def _op_trace(min_samples):
    return compute_structure(OPTICS_POINTS, min_samples), optics_steps(OPTICS_POINTS, min_samples)


@st.cache_data(show_spinner=False)
def _hierarchy(min_samples, size):
    return density_hierarchy(OPTICS_POINTS, min_samples, size)


def render_slide_demo():
    st.markdown("### Theo slide: quan sát quá trình, không chỉ kết quả cuối")
    st.caption("Dữ liệu nhỏ khôi phục từ hình vector trong Density-based clustering.pptx.pdf. Giữ đơn vị gốc, không StandardScaler, luôn 2D và độc lập với bộ dữ liệu/thiết lập ở phía trên. Các khoảng cách được tính lại, không chép số kết quả từ slide.")
    db_tab, op_tab, hdb_tab, score_tab = st.tabs(["DBSCAN từng điểm", "OPTICS và trích cụm", "Cây HDBSCAN / EOM", "Silhouette từng điểm"])
    with db_tab:
        st.caption("Trang 6–22: 15 điểm a–o. Mặc định ε=0.8, MinPts=4: 6 lõi, 8 biên, 1 noise.")
        left, right = st.columns(2)
        eps = left.slider("ε — dữ liệu 15 điểm", .1, 2., .8, .05, key="lesson_db_eps")
        minpts = right.slider("MinPts — DBSCAN từng bước", 2, 8, 4, key="lesson_db_minpts")
        frames = _db_trace(eps, minpts)
        frame = frames[step_controls("lesson_db_step", len(frames)-1)]
        current = frame["current"]
        st.info((f"Xét {chr(97+current)}: |Nε|={len(frame['neighbors'])}. " if current is not None else "") + frame["message"])
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(points_plot(DBSCAN_POINTS, frame["labels"], visited=frame["visited"],
                core=frame["core"], queue=frame["queue"], current=current, radius=eps,
                title="Xám: chưa xét · xanh nhạt: trong N · xanh: lõi · cam: biên · x: noise"),
                use_container_width=True, key="lesson_db_points")
        with right:
            st.write("Danh sách N của cụm đang mở rộng")
            st.dataframe(pd.DataFrame([{"Điểm": chr(97+q), "Đã xét": bool(frame["visited"][q]),
                                        "Vừa thêm": q in frame["added"]} for q in frame["queue"]]),
                         hide_index=True, use_container_width=True)
            st.write("Láng giềng điểm đang xét:", letters(frame["neighbors"]))
        if current is None and frame["visited"].all():
            labels = frame["labels"]
            st.success(f'{len(set(labels)-{-1})} cụm · {int(frame["core"].sum())} lõi · '
                       f'{int(((labels>=0)&~frame["core"]).sum())} biên · {int((labels==-1).sum())} noise')
            for group in sorted(set(labels)-{-1}):
                st.write(f"Cụm {group+1}: {letters(np.flatnonzero(labels==group))}")
            st.write("Noise:", letters(np.flatnonzero(labels==-1)))
    with op_tab:
        st.caption("Trang 28–47: 16 điểm a–p; mặc định MinPts=4, max_eps=∞. Hàng đợi hiển thị các ứng viên theo (reachability, chỉ số gốc).")
        minpts = st.slider("MinPts — OPTICS / cây HDBSCAN theo slide", 2, 8, 4, key="lesson_op_minpts")
        graph, frames = _op_trace(minpts)
        frame = frames[step_controls("lesson_op_step", len(frames)-1)]
        current = frame["current"]
        if current is not None:
            reach = frame["reach"][current]
            st.info(f'Lấy {chr(97+current)}: reach={reach:.2f}, core-dist={graph.core_distances[current]:.2f}. '
                    f'Cập nhật {len(frame["updates"])} ứng viên; có {len(frame["queue"])} điểm trong hàng đợi.')
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(points_plot(OPTICS_POINTS, visited=frame["visited"], current=current,
                radius=None if current is None else graph.core_distances[current],
                title="Đỏ nét đứt: core-distance · xanh: đã xét · xám: chưa xét"),
                use_container_width=True, key="lesson_op_points")
        with right:
            st.dataframe(pd.DataFrame([{"Điểm": chr(97+q), "Reach mới": frame["reach"][q],
                "Reach trước": frame["updates"].get(q, (None, None))[0],
                "Vừa cập nhật": q in frame["updates"]} for q in frame["queue"]]),
                hide_index=True, use_container_width=True)
        st.plotly_chart(reach_plot(graph, frame=frame), use_container_width=True, key="lesson_op_progress")
        st.markdown("#### Trích cụm từ ordering hoàn chỉnh — không chạy lại OPTICS")
        st.caption("Phần dưới luôn dùng đủ 16 điểm để so sánh hai cách trích cụm, không phụ thuộc vị trí thanh bước ở trên.")
        method = st.radio("Cách trích cụm", ["Cắt ngang ε′", "Xi — theo độ dốc"], horizontal=True, key="lesson_op_method")
        cut, xi = None, None
        if method == "Cắt ngang ε′":
            cut = st.slider("ε′ — ngưỡng cắt reachability", .1, 3., 1.5, .05, key="lesson_op_cut")
            a, b = st.columns(2)
            a.button("ε′=1.5 — gộp hai nhóm đầu", on_click=_set, args=("lesson_op_cut", 1.5), key="lesson_cut15")
            b.button("ε′=0.9 — nhóm thưa thành noise", on_click=_set, args=("lesson_op_cut", .9), key="lesson_cut09")
            labels = extract_labels(graph, minpts, method="dbscan", eps=cut)
        else:
            xi = st.slider("ξ — thay đổi tương đối", .01, .20, .10, .01, key="lesson_op_xi")
            labels = extract_labels(graph, minpts, xi=xi)
            st.latex(r"r_{i+1}\leq(1-\xi)r_i\quad\mathrm{(xuống)},\qquad r_i\leq(1-\xi)r_{i+1}\quad\mathrm{(lên)}")
            st.caption("Nền xanh/cam đánh dấu các cặp đủ dốc; nhãn được trích bằng toàn bộ quy tắc Xi của scikit-learn, không chỉ ngưỡng từng cặp.")
        st.plotly_chart(reach_plot(graph, labels=labels, cut=cut, xi=xi), use_container_width=True, key="lesson_op_extraction")
        st.plotly_chart(points_plot(OPTICS_POINTS, labels, title="Cụm/noise sau bước trích"), use_container_width=True, key="lesson_op_labels")
        st.write(f"{len(set(labels)-{-1})} cụm · {int((labels==-1).sum())} noise")
    with hdb_tab:
        minpts = st.session_state.get("lesson_op_minpts", 4)
        st.caption(f"Trang 49–53: dùng cùng 16 điểm và MinPts={minpts} từ phần OPTICS theo slide. Mutual-reachability = max(core(a), core(b), d(a,b)).")
        size = st.slider("min_cluster_size — cây 16 điểm", 2, 8, 4, key="lesson_hdb_size")
        hierarchy = _hierarchy(minpts, size)
        stage = st.select_slider("Giai đoạn phân cấp", options=["1 · ε=2.5", "2 · ε=2.0", "3 · ε=0.8", "4 · Chọn cụm bền (EOM)"], key="lesson_hdb_stage")
        selected = stage.startswith("4")
        eps = None if selected else {"1": 2.5, "2": 2., "3": .8}[stage[0]]
        model = HDBSCAN(min_samples=minpts, min_cluster_size=size, metric="euclidean", copy=True).fit(OPTICS_POINTS)
        labels = align_for_display(hierarchy.labels, model.labels_)[0] if selected else hierarchy.cut(eps, size)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(points_plot(OPTICS_POINTS, labels, title="Cụm được chọn (EOM)" if selected else f"Thành phần ở ε={eps}"), use_container_width=True, key="lesson_hdb_points")
        with right:
            st.plotly_chart(hierarchy_plot(hierarchy, eps, selected), use_container_width=True, key="lesson_hdb_tree")
        st.caption("Các mức ε chỉ để khảo sát cây mutual-reachability/DBSCAN*, không phải tham số eps chạy lại HDBSCAN. Hình cuối dùng nhãn HDBSCAN của scikit-learn. Cây rút gọn và độ bền được tính riêng bằng công thức, không đọc thuộc tính riêng của thư viện.")
        if adjusted_rand_score(model.labels_, hierarchy.labels) != 1.:
            st.warning("Có khác biệt xử lý các mức đồng hạng giữa cây minh hoạ và thư viện; hình cuối vẫn dùng nhãn scikit-learn.")
        table = []
        for cid, node in hierarchy.nodes.items():
            table.append({"Cụm": "Toàn bộ" if cid == 0 else f"C{cid}", "Các điểm khi sinh": letters(node["members"]),
                          "λ sinh": node["birth"], "λ kết thúc": node["death"], "Độ bền": node["stability"],
                          "Tổng độ bền nhánh con tốt nhất": node["descendant_score"], "EOM chọn": cid in hierarchy.selected})
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        st.latex(r"\mathrm{stability}(C)=\sum_{p\in C}(\lambda_p-\lambda_{sinh}),\quad\lambda=1/\varepsilon")
        st.caption("EOM so sánh độ bền cha với tổng phương án tốt nhất của các nhánh con; root không được chọn khi allow_single_cluster=False. Với MinPts=4 và min_cluster_size=4: C1=5.43, C3+C4=7.52 → chọn C3,C4; đồng thời giữ C2.")
    with score_tab:
        points = np.array([(0, 0), (2, 0), (0, -2), (3, 4), (4, 3), (6, 8), (8, 6)], dtype=float)
        labels = np.array([0, 0, 0, 1, 1, 2, 2])
        distances = pairwise_distances(points)
        a = distances[0, 1:3].mean()
        other_means = [distances[0, labels == group].mean() for group in [1, 2]]
        b = min(other_means)
        score = silhouette_samples(points, labels)[0]
        st.caption("Trang 57: điểm a trên hình là điểm i đang xét; cụm A gồm a,b,c, cụm B gồm d,e, cụm C gồm f,g. Không dùng noise như một cụm để tính Silhouette.")
        st.plotly_chart(points_plot(points, labels, current=0, title="Ví dụ tính Silhouette cho điểm i (a)" ).update_xaxes(range=[-1, 10]).update_yaxes(range=[-3, 10]), use_container_width=True, key="lesson_silhouette")
        st.write(f"a(i) = (2+2)/2 = {a:.1f}; khoảng cách trung bình tới B và C: {other_means[0]:.1f}, {other_means[1]:.1f}; b(i) = {b:.1f}.")
        st.latex(r"s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}=\frac{5-2}{5}=0.6")
        st.metric("Silhouette của i — tính bằng scikit-learn", f"{score:.2f}")
        st.info("Trang 58: luôn báo cáo cả Silhouette trên điểm không nhiễu và tỷ lệ noise. Bảng so sánh ba thuật toán hiện đã hiển thị cả hai.")
