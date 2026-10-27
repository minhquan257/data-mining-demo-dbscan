"""Reachability dạng cột: giữ nguyên từng điểm, không gom hoặc làm mượt."""

import numpy as np
import plotly.graph_objects as go

from optics_demo import OpticsStructure


def reachability_bars(model: OpticsStructure, labels: np.ndarray,
                      colors: dict[int, str], *, cut_eps: float | None = None) -> go.Figure:
    order = model.ordering
    values = model.reachability[order]
    ordered_labels = np.asarray(labels)[order]
    finite = np.isfinite(values)
    positions = np.arange(len(order))
    figure = go.Figure()
    for label in sorted(set(ordered_labels)):
        mask = (ordered_labels == label) & finite
        if not mask.any():
            continue
        name = "Noise (-1)" if label < 0 else f"Cụm {label}"
        figure.add_trace(go.Bar(
            x=positions[mask], y=values[mask], width=1., name=name,
            marker={"color": "#222222" if label < 0 else colors[int(label)], "line": {"width": 0}},
            customdata=order[mask],
            hovertemplate=("Thứ tự duyệt: %{x}<br>Chỉ số điểm gốc: %{customdata}"
                           "<br>Reachability: %{y:.4f}<extra>" + name + "</extra>"),
        ))
    # Infinity is not a finite-height column and does not mean noise.
    for i in positions[~finite]:
        figure.add_annotation(x=int(i), y=1.025, xref="x", yref="paper", text="∞",
                              showarrow=False, font={"size": 16, "color": "#333333"},
                              hovertext=f"Điểm gốc {int(order[i])}: reachability vô hạn, chưa có điểm dẫn tới nó.")
    if cut_eps is not None:
        figure.add_hline(y=cut_eps, line_color="#de7526", line_dash="dash",
                         annotation_text=f"Mức cắt ε′ = {cut_eps:.2f}", annotation_position="top right")
    figure.update_layout(
        title={"text": "OPTICS — Reachability", "x": 0, "xref": "paper",
               "y": .98, "yanchor": "top", "font": {"size": 18}},
        height=580, barmode="overlay", bargap=0, template="plotly_white",
        margin={"l": 65, "r": 25, "t": 140, "b": 85},
        xaxis={"title": "Thứ tự duyệt OPTICS — mỗi cột là một điểm",
               "range": [-.5, len(order)-.5], "showgrid": False,
               "rangeslider": {"visible": False}},
        yaxis={"title": "Reachability — thấp hơn nghĩa là dày hơn", "rangemode": "tozero", "fixedrange": False},
        legend={"orientation": "h", "y": 1.10, "yanchor": "bottom", "x": 0,
                "xanchor": "left", "font": {"size": 12}, "itemsizing": "constant"},
        hovermode="closest", dragmode="zoom",
    )
    return figure
