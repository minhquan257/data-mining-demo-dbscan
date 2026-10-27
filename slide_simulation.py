"""Dữ liệu và vết chạy nhỏ để đối chiếu Density-based clustering.pptx.pdf.

Toạ độ được khôi phục từ tâm các điểm vector trong PDF (72 đơn vị vẽ / 1
đơn vị khoảng cách), làm tròn 2 chữ số; chỉ tịnh tiến và đảo chiều trục Y.
Không StandardScaler: giữ đúng các khoảng cách và eps trong slide.
"""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances

from optics_demo import compute_structure


DBSCAN_POINTS = np.array([
    (0, 0), (-.45, .3), (-.4, -.4), (.45, .4), (.55, -.3),
    (1.1, .15), (1.65, -.2), (1.8, .45), (2.4, .1),
    (4.1, -.4), (3.65, 0), (4.55, 0), (3.75, -.9), (4.5, -.85), (2.8, -.95),
])
OPTICS_POINTS = np.array([
    (0, 0), (.25, -.25), (.51, -.5), (.01, -.55), (.57, .03),
    (1.57, .04), (1.84, -.26), (2.09, -.49), (2.08, -.01), (1.52, -.52),
    (4.26, -.78), (4.36, -2.06), (5.64, -.77), (5.52, -2.04), (5.54, .55), (4.37, .54),
])


def dbscan_steps(points: np.ndarray, eps: float, min_samples: int) -> list[dict]:
    distances = pairwise_distances(points)
    neighborhoods = [np.flatnonzero(row <= eps + 1e-12).tolist() for row in distances]
    core = np.array([len(neighbors) >= min_samples for neighbors in neighborhoods])
    visited = np.zeros(len(points), dtype=bool)
    labels = np.full(len(points), -2, dtype=int)  # -2: chưa xét, -1: noise
    frames = []

    def record(current, queue, added, message):
        frames.append(dict(current=current, queue=list(queue), added=list(added),
                           labels=labels.copy(), visited=visited.copy(), core=core.copy(),
                           neighbors=[] if current is None else neighborhoods[current], message=message))

    record(None, [], [], "Dữ liệu ban đầu: chưa xét điểm nào.")
    cluster = -1
    for point in range(len(points)):
        if visited[point]:
            continue
        visited[point] = True
        if not core[point]:
            labels[point] = -1
            record(point, [], [], "Chưa đủ MinPts: tạm gán noise; có thể được kéo vào cụm sau.")
            continue
        cluster += 1
        labels[point] = cluster
        queue = list(neighborhoods[point])
        record(point, queue, [q for q in queue if q != point], f"Điểm lõi: mở Cụm {cluster + 1}.")
        position = 0
        while position < len(queue):
            q = queue[position]
            position += 1
            newly_visited = not visited[q]
            added = []
            if newly_visited:
                visited[q] = True
                if core[q]:
                    added = [i for i in neighborhoods[q] if i not in queue]
                    queue.extend(added)
            if labels[q] < 0:
                labels[q] = cluster
            if newly_visited:
                message = "Lõi: mở rộng danh sách N." if core[q] else "Biên: thuộc cụm nhưng không mở rộng N."
                record(q, queue, added, message)
    record(None, [], [], "Hoàn tất phân cụm; nhãn noise là -1.")
    return frames


def optics_steps(points: np.ndarray, min_samples: int) -> list[dict]:
    """Vết cập nhật reachability; đối chiếu ordering với scikit-learn.

    Hàng đợi được biểu diễn bằng mảng và sắp (reachability, chỉ số gốc),
    tương ứng cách chọn nhỏ nhất của scikit-learn; không giả định có heap.
    """
    graph = compute_structure(points, min_samples)
    distances = pairwise_distances(points)
    reach = np.full(len(points), np.inf)
    predecessor = np.full(len(points), -1, dtype=int)
    visited = np.zeros(len(points), dtype=bool)
    order = []
    frames = [dict(current=None, order=[], reach=reach.copy(), predecessor=predecessor.copy(),
                   queue=[], updates={}, visited=visited.copy())]
    for _ in points:
        remaining = np.flatnonzero(~visited)
        point = int(remaining[np.argmin(reach[remaining])])
        visited[point] = True
        order.append(point)
        updates = {}
        for q in np.flatnonzero(~visited):
            candidate = float(np.around(max(graph.core_distances[point], distances[point, q]), 15))
            if candidate < reach[q]:
                updates[int(q)] = (float(reach[q]), candidate)
                reach[q] = candidate
                predecessor[q] = point
        queue = sorted(np.flatnonzero(~visited & np.isfinite(reach)).tolist(), key=lambda q: (reach[q], q))
        frames.append(dict(current=point, order=list(order), reach=reach.copy(),
                           predecessor=predecessor.copy(), queue=queue, updates=updates,
                           visited=visited.copy()))
    return frames


@dataclass
class DensityHierarchy:
    linkage: np.ndarray
    nodes: dict
    selected: set
    labels: np.ndarray

    def cut(self, eps: float, min_cluster_size: int) -> np.ndarray:
        raw = fcluster(self.linkage, t=eps, criterion="distance")
        labels = np.full(len(raw), -1, dtype=int)
        next_label = 0
        for group in np.unique(raw):
            mask = raw == group
            if mask.sum() >= min_cluster_size:
                labels[mask] = next_label
                next_label += 1
        return labels


def density_hierarchy(points: np.ndarray, min_samples: int, min_cluster_size: int) -> DensityHierarchy:
    """Cây single linkage mutual-reachability, rút gọn và chọn cụm EOM.

    Bản minh hoạ cho dữ liệu nhỏ, dùng API công khai NumPy, SciPy,
    scikit-learn, không truy cập _single_linkage_tree_ riêng của thư viện.
    """
    n = len(points)
    core = compute_structure(points, min_samples).core_distances
    distances = pairwise_distances(points)
    mutual = np.maximum(np.maximum(distances, core[:, None]), core[None, :])
    np.fill_diagonal(mutual, 0)
    tree = linkage(squareform(mutual, checks=False), method="single")
    if np.any(tree[:, 2] <= 0):
        raise ValueError("Phần cây minh hoạ yêu cầu các khoảng cách ghép dương.")
    members = {i: [i] for i in range(n)}
    for i, row in enumerate(tree):
        members[n + i] = members[int(row[0])] + members[int(row[1])]
    nodes = {}

    def new_cluster(node, birth, parent):
        cid = len(nodes)
        nodes[cid] = dict(members=sorted(members[node]), birth=birth, death=birth,
                          stability=0., children=[], parent=parent)
        return cid

    root = new_cluster(2 * n - 2, 0., None)

    def descend(node, cid):
        if node < n:
            return
        left, right, height, _ = tree[node - n]
        left, right = int(left), int(right)
        lam = 1. / height
        big_left = len(members[left]) >= min_cluster_size
        big_right = len(members[right]) >= min_cluster_size
        current = nodes[cid]
        current["death"] = max(current["death"], lam)
        if big_left and big_right:
            current["stability"] += (lam - current["birth"]) * len(members[node])
            # Stable ordering by first original point helps map C1/C2 to the slide.
            child_pairs = []
            for child in sorted([left, right], key=lambda x: min(members[x])):
                child_cid = new_cluster(child, lam, cid)
                current["children"].append(child_cid)
                child_pairs.append((child, child_cid))
            for child, child_cid in child_pairs:
                descend(child, child_cid)
        elif big_left or big_right:
            big, small = (left, right) if big_left else (right, left)
            current["stability"] += (lam - current["birth"]) * len(members[small])
            descend(big, cid)
        else:
            current["stability"] += (lam - current["birth"]) * len(members[node])

    descend(2 * n - 2, root)

    def choose(cid):
        descendants = [choose(child) for child in nodes[cid]["children"]]
        total = sum(score for score, _ in descendants)
        chosen = set().union(*(selection for _, selection in descendants))
        nodes[cid]["descendant_score"] = total
        if cid != root and nodes[cid]["stability"] >= total:
            return nodes[cid]["stability"], {cid}
        return total, chosen

    _, selected = choose(root)
    labels = np.full(n, -1, dtype=int)
    # Giữ thành viên tại lúc cụm sinh; điểm rơi khỏi nhánh cha trước lúc
    # một cụm con được sinh không được đưa vào cụm con đó.
    for label, cid in enumerate(sorted(selected)):
        labels[nodes[cid]["members"]] = label
    return DensityHierarchy(tree, nodes, selected, labels)
