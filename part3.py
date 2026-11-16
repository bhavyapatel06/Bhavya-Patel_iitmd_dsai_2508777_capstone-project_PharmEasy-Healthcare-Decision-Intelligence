"""Part 3: deterministic K-means, hierarchical clustering and PCA."""
import numpy as np
from common import load_data, write_results, write_csv, ensure_outputs, OUTPUT_DIR


def kmeans(z, k):
    n = len(z)
    step = n // k
    centroids = z[np.array([j * step for j in range(k)], dtype=int)].copy()
    labels = np.full(n, -1, dtype=int)
    passes = 0
    while True:
        distances = np.sum((z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)  # argmin resolves ties to lower cluster index
        passes += 1
        changed = np.any(new_labels != labels)
        labels = new_labels
        for j in range(k):
            members = z[labels == j]
            if len(members):
                centroids[j] = np.mean(members, axis=0)
        if not changed:
            break
    final_dist = np.sum((z - centroids[labels]) ** 2, axis=1)
    return labels, centroids, float(np.sum(final_dist)), passes


def hierarchical(matrix, linkage):
    clusters = [[i] for i in range(6)]
    merges = []
    for merge_no in range(1, 6):
        candidates = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                vals = [matrix[a, b] for a in clusters[i] for b in clusters[j]]
                if linkage == "single":
                    distance = min(vals)
                elif linkage == "complete":
                    distance = max(vals)
                else:
                    distance = sum(vals) / len(vals)
                candidates.append((distance, i, j))
        distance, i, j = min(candidates, key=lambda t: (t[0], min(clusters[t[1]]), min(clusters[t[2]])))
        a, b = clusters[i], clusters[j]
        if min(a) > min(b):
            a, b = b, a
        name = lambda c: "+".join(f"E{v+1}" for v in sorted(c))
        merges.append((linkage, merge_no, name(a), name(b), float(distance)))
        merged = sorted(a + b)
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)] + [merged]
        clusters.sort(key=lambda c: min(c))
    heights = [r[4] for r in merges]
    gaps = np.diff(heights)
    largest_gap_index = int(np.argmax(gaps))
    merges_below = largest_gap_index + 1
    count_at_cut = 6 - merges_below
    cut_clusters = ["+".join(f"E{v+1}" for v in c) for c in clusters] if count_at_cut == 1 else None
    # Reconstruct the partition after exactly the merges below the largest gap.
    active = [[i] for i in range(6)]
    for row in merges[:merges_below]:
        members = set(int(s[1:]) - 1 for s in (row[2] + "+" + row[3]).split("+"))
        active = [c for c in active if not set(c).issubset(members)] + [sorted(members)]
        active.sort(key=lambda c: min(c))
    partition = ["+".join(f"E{v+1}" for v in c) for c in active]
    return merges, partition, count_at_cut


def main():
    ensure_outputs()
    _, x, y, value = load_data()
    x5 = x[:, :5]
    mean = np.mean(x5, axis=0)
    sd = np.std(x5, axis=0, ddof=1)
    z = (x5 - mean) / sd

    out = {}
    labels_by_k = {}
    wcss = {}
    for k in range(2, 7):
        labels, centroids, score, passes = kmeans(z, k)
        labels_by_k[k] = labels
        wcss[k] = score
        out[f"wcss_k{k}"] = score
        out[f"kmeans_passes_k{k}"] = passes
    elbow_k = 5
    for k in (3, 4, 5):
        if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k]):
            elbow_k = k
            break
    out["elbow_k"] = elbow_k

    labels = labels_by_k[3]
    cluster_ids = list(range(3))
    cluster_ids.sort(key=lambda j: float(np.mean(x5[labels == j, 2])))
    segment_rows = []
    segment_label = np.empty(len(y), dtype="U1")
    for name, cid in zip(("A", "B", "C"), cluster_ids):
        mask = labels == cid
        segment_label[mask] = name
        out[f"size_segment_{name}"] = int(np.sum(mask))
        segment_rows.append((name, int(np.sum(mask)), *[float(np.mean(x5[mask, j])) for j in range(5)],
                             float(np.mean(y[mask])), float(np.mean(value[mask]))))
    write_csv(OUTPUT_DIR / "part3_segments.csv",
              ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
              segment_rows)

    matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ])
    hierarchical_rows = []
    for linkage in ("single", "complete", "average"):
        merges, partition, count = hierarchical(matrix, linkage)
        hierarchical_rows.extend(merges)
        out[f"{linkage}_clusters_at_cut"] = ";".join(partition)
        out[f"{linkage}_cluster_count_at_cut"] = count
    write_csv(OUTPUT_DIR / "part3_hierarchical.csv",
              ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hierarchical_rows)

    # PCA on the centred, standardised data. eigh returns ascending eigenvalues.
    z_centered = z - np.mean(z, axis=0)
    covariance = (z_centered.T @ z_centered) / (len(z_centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for j in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, j])))
        if eigenvectors[pivot, j] < 0:
            eigenvectors[:, j] *= -1
    total = float(np.sum(eigenvalues))
    ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)
    projection = z_centered @ eigenvectors[:, :2]
    out["eigenvalue_sum"] = total
    for i in range(5):
        out[f"eigenvalue_{i+1}"] = float(eigenvalues[i])
        out[f"explained_ratio_{i+1}"] = float(ratios[i])
        out[f"cumulative_ratio_{i+1}"] = float(cumulative[i])
    out["components_for_80pct"] = int(np.argmax(cumulative >= 0.8) + 1)
    for j in range(5):
        out[f"pc1_loading_X{j+1}"] = float(eigenvectors[j, 0])
        out[f"pc2_loading_X{j+1}"] = float(eigenvectors[j, 1])
    for i in range(3):
        out[f"row{i+1}_pc1"] = float(projection[i, 0])
        out[f"row{i+1}_pc2"] = float(projection[i, 1])
    for name in ("A", "B", "C"):
        out[f"mean_pc1_segment_{name}"] = float(np.mean(projection[segment_label == name, 0]))

    write_results(OUTPUT_DIR / "part3_results.txt", out)
    print("Part 3 complete:", OUTPUT_DIR / "part3_results.txt")

if __name__ == "__main__":
    main()
