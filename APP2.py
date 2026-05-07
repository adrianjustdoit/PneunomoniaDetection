import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── scikit-image ──────────────────────────────────────────────────────────────
try:
    from skimage.feature import local_binary_pattern
    from skimage import exposure, morphology, measure
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.morphology import binary_fill_holes
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

# ── sklearn ───────────────────────────────────────────────────────────────────
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── scipy ─────────────────────────────────────────────────────────────────────
try:
    from scipy.spatial.distance import cdist
    from scipy.ndimage import binary_fill_holes as scipy_fill_holes
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return x * torch.sigmoid(self.mlp(avg) + self.mlp(mx))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class LIFNeuron(nn.Module):
    def __init__(self, beta=0.9, threshold=1.0, surrogate_slope=10.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope
    def forward(self, input_current, membrane):
        membrane = self.beta * membrane + input_current
        hard_spike = (membrane >= self.threshold).float()
        surrogate = torch.sigmoid(self.surrogate_slope * (membrane - self.threshold))
        spike = hard_spike + surrogate - surrogate.detach()
        membrane = membrane * (1.0 - hard_spike.detach())
        return spike, membrane

class TimeAwareSpikingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps=5, attention_ratio=8, dropout=0.2):
        super().__init__()
        self.time_steps = int(time_steps)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.ca = ChannelAttention(out_channels, ratio=attention_ratio)
        self.sa = SpatialAttention()
        self.lif = LIFNeuron()
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        mem = torch.zeros((x.shape[0], self.conv[0].out_channels, x.shape[2], x.shape[3]), device=x.device)
        spike_sum, mem_sum = 0, 0
        for _ in range(self.time_steps):
            h = self.dropout(self.sa(self.ca(self.conv(x))))
            spike, mem = self.lif(h, mem)
            spike_sum, mem_sum = spike_sum + spike, mem_sum + mem
        return self.pool(F.relu((spike_sum + mem_sum) / self.time_steps))

class MDASNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, time_steps=5, attention_ratio=8, dropout_rate=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.block1 = TimeAwareSpikingBlock(32, 64, time_steps, attention_ratio, dropout_rate)
        self.block2 = TimeAwareSpikingBlock(64, 128, time_steps, attention_ratio, dropout_rate)
        self.block3 = TimeAwareSpikingBlock(128, 192, time_steps, attention_ratio, dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout_rate),
            nn.Linear(192, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(self.global_pool(x))

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p=0.2), nn.Linear(192, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ═══════════════════════════════════════════════════════════════════════════════
# BALANCED MORPHOLOGICAL FILTER (BMF)
# ═══════════════════════════════════════════════════════════════════════════════

class BalancedMorphologicalFilter:
    """
    BMF: removes noise, preserves anatomical edges, enhances contrast.
    - Opening removes bright small noise
    - Closing fills dark small gaps
    - Top-hat emphasizes bright local structures
    - Bottom-hat emphasizes dark local structures
    - enhanced = image + top_hat - bottom_hat
    """
    def __init__(self, kernel_size=5, iterations=1, denoise="median"):
        ks = int(kernel_size)
        self.kernel_size = ks if ks % 2 == 1 else ks + 1
        self.iterations = int(iterations)
        self.denoise = denoise
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))

    def _to_gray(self, image):
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        else:
            img = image.copy()
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img is None:
            raise ValueError("Image could not be loaded")
        return img.astype(np.uint8)

    def apply(self, image, return_steps=False):
        gray = self._to_gray(image)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        opening  = cv2.morphologyEx(norm, cv2.MORPH_OPEN,    self.kernel, iterations=self.iterations)
        closing  = cv2.morphologyEx(norm, cv2.MORPH_CLOSE,   self.kernel, iterations=self.iterations)
        top_hat  = cv2.morphologyEx(norm, cv2.MORPH_TOPHAT,  self.kernel, iterations=self.iterations)
        bot_hat  = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT,self.kernel, iterations=self.iterations)

        enhanced = norm.astype(np.float32) + top_hat.astype(np.float32) - bot_hat.astype(np.float32)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        if self.denoise == "median":
            enhanced = cv2.medianBlur(enhanced, 3)
        elif self.denoise == "bilateral":
            enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

        if SKIMAGE_OK:
            p2, p98 = np.percentile(enhanced, (2, 98))
            if p98 > p2:
                enhanced = exposure.rescale_intensity(enhanced, in_range=(p2, p98), out_range=(0, 255)).astype(np.uint8)
            else:
                enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if return_steps:
            return {"gray": gray, "normalized": norm, "opening": opening,
                    "closing": closing, "top_hat": top_hat, "bottom_hat": bot_hat, "enhanced": enhanced}
        return enhanced


# ─── BMF Quality Metrics ──────────────────────────────────────────────────────

def image_entropy(img):
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())

def contrast_measure(img):
    return float(np.std(img))

def sharpness_laplacian(img):
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def bmf_quality_metrics(raw_gray, enhanced):
    raw_r = cv2.resize(raw_gray, (enhanced.shape[1], enhanced.shape[0])) if raw_gray.shape != enhanced.shape else raw_gray
    metrics = {"Entropy": image_entropy(enhanced), "Contrast STD": contrast_measure(enhanced),
               "Sharpness (Laplacian)": sharpness_laplacian(enhanced)}
    if SKIMAGE_OK:
        try:
            metrics["PSNR"] = float(peak_signal_noise_ratio(raw_r, enhanced, data_range=255))
        except Exception:
            metrics["PSNR"] = float("nan")
        try:
            metrics["SSIM"] = float(structural_similarity(raw_r, enhanced, data_range=255))
        except Exception:
            metrics["SSIM"] = float("nan")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SIMVC — SEMANTIC INVARIANT MULTI-VIEW CLUSTERING SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticInvariantMultiViewClustering:
    """
    SIMVC-inspired segmentation:
    - 8-view pixel feature extraction (intensity, LBP, local variance,
      Sobel edges, BMF morphological response, x/y coords, center prior)
    - KMeans / Spectral / Fuzzy C-Means clustering
    - Anatomically-guided ROI cluster selection
    - Post-processing: small-object removal, binary closing, hole filling,
      bilateral lung component selection
    """
    def __init__(self, n_clusters=3, method="kmeans", working_size=128, random_state=SEED):
        self.n_clusters   = int(n_clusters)
        self.method       = method
        self.working_size = int(working_size)
        self.random_state = random_state
        self.scaler       = StandardScaler() if SKLEARN_OK else None

    def _resize_working(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (self.working_size, self.working_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8), (h, w)

    def extract_multiview_features(self, image):
        img, original_shape = self._resize_working(image)
        img_f = img.astype(np.float32) / 255.0
        H, W  = img.shape

        intensity = img_f

        if SKIMAGE_OK:
            lbp = local_binary_pattern(img, P=8, R=1, method="uniform").astype(np.float32)
            lbp = cv2.normalize(lbp, None, 0, 1, cv2.NORM_MINMAX)
        else:
            lbp = np.zeros_like(img_f)

        mean    = cv2.blur(img_f, (5, 5))
        mean_sq = cv2.blur(img_f ** 2, (5, 5))
        local_var = np.maximum(mean_sq - mean ** 2, 0)
        local_var = cv2.normalize(local_var, None, 0, 1, cv2.NORM_MINMAX)

        sx   = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        sy   = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sx**2 + sy**2)
        edge = cv2.normalize(edge, None, 0, 1, cv2.NORM_MINMAX)

        bmf_img = BalancedMorphologicalFilter(kernel_size=5).apply(img)
        morph   = bmf_img.astype(np.float32) / 255.0

        yy, xx  = np.mgrid[0:H, 0:W]
        x_norm  = xx.astype(np.float32) / max(W - 1, 1)
        y_norm  = yy.astype(np.float32) / max(H - 1, 1)
        center_prior = 1.0 - np.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2)
        center_prior = cv2.normalize(center_prior, None, 0, 1, cv2.NORM_MINMAX)

        stack = np.stack([intensity, lbp, local_var, edge, morph, x_norm, y_norm, center_prior], axis=-1)
        X     = stack.reshape(-1, stack.shape[-1])
        return X, stack, img, original_shape

    def _fuzzy_cmeans(self, X, m=2.0, max_iter=50, error=1e-4):
        rng = np.random.default_rng(self.random_state)
        n, c = X.shape[0], self.n_clusters
        U = rng.random((n, c))
        U = U / U.sum(axis=1, keepdims=True)
        for _ in range(max_iter):
            U_old = U.copy()
            um      = U ** m
            centers = (um.T @ X) / (um.sum(axis=0)[:, None] + 1e-12)
            dist    = cdist(X, centers) + 1e-8 if SCIPY_OK else (np.linalg.norm(X[:, None] - centers[None], axis=-1) + 1e-8)
            inv     = dist ** (-2 / (m - 1))
            U       = inv / inv.sum(axis=1, keepdims=True)
            if np.linalg.norm(U - U_old) < error:
                break
        return np.argmax(U, axis=1), centers

    def _cluster(self, X):
        Xs = self.scaler.fit_transform(X) if self.scaler else X
        if self.method == "spectral" and SKLEARN_OK:
            model  = SpectralClustering(n_clusters=self.n_clusters, affinity="nearest_neighbors",
                                        assign_labels="kmeans", random_state=self.random_state)
            labels = model.fit_predict(Xs)
            centers = np.vstack([Xs[labels == k].mean(axis=0) for k in range(self.n_clusters)])
        elif self.method == "fuzzy":
            labels, centers = self._fuzzy_cmeans(Xs)
        else:
            if SKLEARN_OK:
                model   = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
                labels  = model.fit_predict(Xs)
                centers = model.cluster_centers_
            else:
                # fallback: random init
                rng = np.random.default_rng(self.random_state)
                centers = Xs[rng.choice(len(Xs), self.n_clusters, replace=False)]
                for _ in range(50):
                    dists  = np.linalg.norm(Xs[:, None] - centers[None], axis=-1)
                    labels = np.argmin(dists, axis=1)
                    new_c  = np.vstack([Xs[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
                                        for k in range(self.n_clusters)])
                    if np.allclose(centers, new_c):
                        break
                    centers = new_c
        return labels, centers, Xs

    def _select_roi_cluster(self, labels, feature_stack):
        H, W, _ = feature_stack.shape
        scores   = []
        for k in range(self.n_clusters):
            mask = labels.reshape(H, W) == k
            if not mask.any():
                scores.append(-np.inf); continue
            intensity    = feature_stack[..., 0][mask].mean()
            texture      = feature_stack[..., 2][mask].mean()
            edge         = feature_stack[..., 3][mask].mean()
            morph        = feature_stack[..., 4][mask].mean()
            center       = feature_stack[..., 7][mask].mean()
            area_penalty = abs(mask.mean() - 0.35)
            s = 0.25*center + 0.20*edge + 0.20*texture + 0.20*morph + 0.15*intensity - 0.25*area_penalty
            scores.append(float(s))
        return int(np.argmax(scores)), scores

    def _postprocess_mask(self, mask_small, original_shape):
        if SKIMAGE_OK:
            mask = morphology.remove_small_objects(mask_small.astype(bool), min_size=max(16, mask_small.size // 200))
            mask = morphology.binary_closing(mask, morphology.disk(3))
            mask = binary_fill_holes(mask)
            labeled = measure.label(mask)
            props   = measure.regionprops(labeled)
            if props:
                top2 = sorted(props, key=lambda p: p.area, reverse=True)[:2]
                keep = np.zeros_like(mask, dtype=bool)
                for p in top2:
                    keep[labeled == p.label] = True
                mask = keep
        else:
            mask = mask_small.astype(bool)
        h, w = original_shape
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    def fit_predict(self, image):
        X, feature_stack, img_small, original_shape = self.extract_multiview_features(image)
        labels, centers, Xs = self._cluster(X)
        H, W = img_small.shape
        cluster_map = labels.reshape(H, W)
        roi_cluster, cluster_scores = self._select_roi_cluster(labels, feature_stack)
        mask_small  = cluster_map == roi_cluster
        mask        = self._postprocess_mask(mask_small, original_shape)

        if image.ndim == 3:
            original_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = image.copy()
        segmented = (original_gray * mask.astype(np.uint8)).astype(np.uint8)

        metrics = {}
        if SKLEARN_OK:
            n = len(labels)
            sample_size = 3000
            if n > sample_size:
                rng = np.random.default_rng(SEED)
                idx = rng.choice(n, size=sample_size, replace=False)
                Xe, le = Xs[idx], labels[idx]
            else:
                Xe, le = Xs, labels
            try: metrics["Silhouette"]        = float(silhouette_score(Xe, le))
            except: metrics["Silhouette"]     = float("nan")
            try: metrics["Davies-Bouldin"]    = float(davies_bouldin_score(Xe, le))
            except: metrics["Davies-Bouldin"] = float("nan")
            try: metrics["Calinski-Harabasz"] = float(calinski_harabasz_score(Xe, le))
            except: metrics["Calinski-Harabasz"] = float("nan")
            comp = sum(
                np.mean(np.linalg.norm(Xs[labels == k] - centers[k], axis=1))
                for k in np.unique(labels) if (labels == k).any()
            )
            metrics["Compactness"] = float(comp / max(len(np.unique(labels)), 1))

        # Build contour overlay
        overlay = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)
        if SKIMAGE_OK:
            contours = measure.find_contours(mask.astype(float), 0.5)
            for c in contours:
                pts = c[:, ::-1].astype(np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 220, 170), thickness=2)

        return {"cluster_map": cluster_map, "mask": mask, "segmented": segmented,
                "metrics": metrics, "overlay": overlay, "working_image": img_small,
                "cluster_scores": cluster_scores, "roi_cluster": roi_cluster}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name   = model_path.lower()
    if "mdasnn" in name:
        ratio = 8 if "default" in name else 13
        model = MDASNN(num_classes=2, attention_ratio=ratio).to(device)
        arch  = f"MDASNN (attention ratio={ratio})"
    else:
        model = SimpleCNN(num_classes=2).to(device)
        arch  = "SimpleCNN"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device, True, None, arch
    except Exception as e:
        return None, device, False, str(e), arch


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CXR · Neural Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Root variables ──────────────────────────────── */
:root {
  --bg:       #060b14;
  --surface:  #0d1526;
  --surface2: #111d33;
  --border:   #1e3050;
  --accent:   #00e5c0;
  --accent2:  #4d9eff;
  --danger:   #ff5a6e;
  --warn:     #ffb347;
  --text:     #ccd9f0;
  --muted:    #607090;
  --font-head: 'Syne', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'DM Mono', monospace;
}

/* ── Reset & base ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}

[data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 1px solid var(--border); }

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }

/* ── Hero banner ─────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, #050d1f 0%, #0a1a35 40%, #082030 70%, #05121e 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 3rem 3.5rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at 20% 50%, rgba(0,229,192,.07) 0%, transparent 60%),
              radial-gradient(ellipse at 80% 30%, rgba(77,158,255,.06) 0%, transparent 60%);
  pointer-events: none;
}
.hero-title {
  font-family: var(--font-head);
  font-size: 2.8rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(90deg, #ffffff 0%, #a8d4ff 50%, var(--accent) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 .5rem;
}
.hero-sub {
  font-family: var(--font-mono);
  font-size: .88rem;
  color: var(--muted);
  letter-spacing: .05em;
}
.hero-badges { margin-top: 1.4rem; display: flex; gap: .6rem; flex-wrap: wrap; }
.badge {
  background: rgba(255,255,255,.04);
  border: 1px solid var(--border);
  border-radius: 30px;
  padding: .25rem .85rem;
  font-family: var(--font-mono);
  font-size: .7rem;
  color: var(--muted);
  letter-spacing: .06em;
}
.badge.accent { border-color: var(--accent); color: var(--accent); }

/* ── Stage cards ─────────────────────────────────── */
.stage-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.6rem 2rem;
  margin-bottom: 1.4rem;
  position: relative;
}
.stage-card::before {
  content: '';
  position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  border-radius: 3px 0 0 3px;
}
.stage-card.bmf::before  { background: var(--accent2); }
.stage-card.simvc::before { background: var(--accent); }
.stage-card.pred::before  { background: var(--warn); }

.stage-header {
  font-family: var(--font-head);
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: .6rem;
}
.stage-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 24px; height: 24px;
  border-radius: 50%;
  font-size: .7rem; font-weight: 700;
  background: var(--border);
  font-family: var(--font-mono);
}

/* ── Image caption ───────────────────────────────── */
.img-label {
  font-family: var(--font-mono);
  font-size: .68rem;
  color: var(--muted);
  letter-spacing: .07em;
  text-align: center;
  margin-top: .3rem;
  text-transform: uppercase;
}

/* ── Metrics row ─────────────────────────────────── */
.metric-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: .9rem 1.1rem;
  text-align: center;
}
.metric-val {
  font-family: var(--font-mono);
  font-size: 1.4rem;
  font-weight: 500;
  color: var(--accent);
}
.metric-key {
  font-size: .68rem;
  color: var(--muted);
  margin-top: .2rem;
  text-transform: uppercase;
  letter-spacing: .06em;
}

/* ── Diagnosis result ────────────────────────────── */
.dx-normal {
  background: linear-gradient(135deg, #072018 0%, #0b2d22 100%);
  border: 1px solid #1a5c40;
  border-radius: 14px;
  padding: 1.8rem 2rem;
  text-align: center;
}
.dx-pneumonia {
  background: linear-gradient(135deg, #200d10 0%, #2d0f14 100%);
  border: 1px solid #5c1a24;
  border-radius: 14px;
  padding: 1.8rem 2rem;
  text-align: center;
}
.dx-label {
  font-family: var(--font-head);
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -.02em;
  margin-bottom: .4rem;
}
.dx-conf {
  font-family: var(--font-mono);
  font-size: .8rem;
  color: var(--muted);
}
.dx-conf span {
  color: var(--text);
  font-weight: 500;
}

/* ── Probability bars ────────────────────────────── */
.prob-row {
  display: flex;
  align-items: center;
  gap: .8rem;
  margin-bottom: .7rem;
}
.prob-label {
  font-family: var(--font-mono);
  font-size: .75rem;
  color: var(--muted);
  width: 80px;
  flex-shrink: 0;
}
.prob-bar-bg {
  flex: 1;
  height: 8px;
  background: var(--border);
  border-radius: 4px;
  overflow: hidden;
}
.prob-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width .6s ease;
}
.prob-pct {
  font-family: var(--font-mono);
  font-size: .75rem;
  width: 50px;
  text-align: right;
  flex-shrink: 0;
}

/* ── Sidebar styling ─────────────────────────────── */
.sidebar-section {
  font-family: var(--font-head);
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.2rem 0 .5rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: .3rem;
}
.model-rec {
  background: rgba(0,229,192,.06);
  border: 1px solid rgba(0,229,192,.2);
  border-radius: 8px;
  padding: .6rem .8rem;
  font-family: var(--font-mono);
  font-size: .7rem;
  color: var(--accent);
  margin-bottom: .6rem;
}

/* ── Info box ────────────────────────────────────── */
.info-box {
  background: rgba(77,158,255,.06);
  border: 1px solid rgba(77,158,255,.2);
  border-radius: 8px;
  padding: .7rem 1rem;
  font-size: .8rem;
  color: var(--text);
  margin: .8rem 0;
}

/* ── Divider ─────────────────────────────────────── */
.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.5rem 0;
}

/* ── Streamlit overrides ─────────────────────────── */
.stSelectbox > div > div { background: var(--surface2) !important; border-color: var(--border) !important; }
.stSlider .rc-slider-rail { background: var(--border) !important; }
.stSlider .rc-slider-track { background: var(--accent2) !important; }
.stFileUploader { background: var(--surface2) !important; border-color: var(--border) !important; }
.stExpander { background: var(--surface) !important; border-color: var(--border) !important; }

/* ── Pipeline flow ───────────────────────────────── */
.pipeline-flow {
  display: flex; align-items: center; gap: .5rem;
  font-family: var(--font-mono); font-size: .72rem;
  color: var(--muted);
  margin: .8rem 0 1.4rem;
  flex-wrap: wrap;
}
.pf-step {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: .25rem .7rem;
  color: var(--text);
}
.pf-arrow { color: var(--border); }
.pf-active { border-color: var(--accent); color: var(--accent); }

/* ── Footer ─────────────────────────────────────── */
.footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: .68rem;
  color: var(--muted);
  text-align: center;
  letter-spacing: .04em;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='font-family:var(--font-head,Syne);font-size:1.15rem;font-weight:800;
         letter-spacing:-.01em;color:#ccd9f0;padding:.5rem 0 .2rem;'>
      🫁 CXR · Neural Diagnostics
    </div>
    <div style='font-family:var(--font-mono,monospace);font-size:.68rem;color:#607090;
         margin-bottom:1rem;'>BMF + SIMVC + Deep Learning Pipeline</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Model Selection</div>', unsafe_allow_html=True)

    MODEL_INFO = {
        "cnn_plus_bmf_plus_simvc.pth": ("★ Recommended", "SimpleCNN · BMF+SIMVC pipeline match"),
        "cnn_plus_bmf.pth":            ("", "SimpleCNN · BMF only"),
        "cnn_raw.pth":                 ("", "SimpleCNN · raw input"),
        "best_mdasnn_default.pth":     ("", "MDASNN · attention ratio 8"),
        "best_mdasnn_model.pth":       ("", "MDASNN · attention ratio 13"),
    }

    st.markdown("""
    <div class="model-rec">
      ★ <strong>cnn_plus_bmf_plus_simvc.pth</strong> is the recommended model — 
      trained with the same BMF+SIMVC preprocessing used at inference.
    </div>""", unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Select model weights (.pth):",
        list(MODEL_INFO.keys()),
        format_func=lambda k: f"{MODEL_INFO[k][0]} {k}" if MODEL_INFO[k][0] else k,
    )

    model, device, is_loaded, error_msg, arch_label = load_model(selected_model)

    if is_loaded:
        st.success(f"✓ Loaded — {arch_label}")
    else:
        st.error("✗ Failed to load model")
        with st.expander("Error details"):
            st.code(error_msg)

    st.markdown('<div class="sidebar-section">BMF Parameters</div>', unsafe_allow_html=True)
    bmf_kernel    = st.slider("Kernel size", 3, 11, 5, step=2, help="Morphological structuring element size (odd)")
    bmf_iters     = st.slider("Iterations",  1,  5, 1)
    bmf_denoise   = st.selectbox("Denoising", ["median", "bilateral", "none"])

    st.markdown('<div class="sidebar-section">SIMVC Parameters</div>', unsafe_allow_html=True)
    simvc_clusters = st.slider("Clusters (k)", 2, 6, 3)
    simvc_method   = st.selectbox("Clustering method", ["kmeans", "fuzzy", "spectral"])
    simvc_wsize    = st.select_slider("Working size (px)", [64, 96, 128, 160], value=128)

    st.markdown('<div class="sidebar-section">Input Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Chest X-Ray (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if not SKIMAGE_OK:
        st.warning("scikit-image not found. BMF quality metrics and SIMVC post-processing limited.")
    if not SKLEARN_OK:
        st.warning("scikit-learn not found. Clustering limited to basic K-Means fallback.")


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-title">Chest X-Ray · Neural Diagnostics</div>
  <div class="hero-sub">FULL BMF PREPROCESSING  ·  SIMVC SEGMENTATION  ·  DEEP NEURAL CLASSIFICATION</div>
  <div class="hero-badges">
    <span class="badge accent">BMF Enhanced</span>
    <span class="badge">SIMVC Multi-View Clustering</span>
    <span class="badge">MDASNN / SimpleCNN</span>
    <span class="badge">Pneumonia Detection</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — only when image is uploaded
# ═══════════════════════════════════════════════════════════════════════════════

if uploaded_file is None:
    st.markdown("""
    <div style='text-align:center;padding:5rem 2rem;'>
      <div style='font-size:4rem;margin-bottom:1.5rem;'>🫁</div>
      <div style='font-family:var(--font-head,Syne);font-size:1.3rem;font-weight:700;
           color:#ccd9f0;margin-bottom:.6rem;'>Upload a Chest X-Ray to begin</div>
      <div style='font-family:var(--font-mono,monospace);font-size:.8rem;color:#607090;'>
        Use the sidebar to select a model and upload your image.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Decode uploaded image ──────────────────────────────────────────────────────
raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
raw_img   = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
if raw_img is None:
    st.error("Could not decode image. Please upload a valid JPG or PNG file.")
    st.stop()
raw_img = cv2.resize(raw_img, (224, 224))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — BMF PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="stage-card bmf">
  <div class="stage-header">
    <span class="stage-num" style="background:#1a3060;color:#4d9eff;">1</span>
    Balanced Morphological Filter — Preprocessing
  </div>
""", unsafe_allow_html=True)

with st.spinner("Applying BMF pipeline …"):
    bmf = BalancedMorphologicalFilter(kernel_size=bmf_kernel, iterations=bmf_iters,
                                       denoise=bmf_denoise if bmf_denoise != "none" else "none")
    steps = bmf.apply(raw_img, return_steps=True)

step_labels = ["Raw Gray", "Normalized", "Opening", "Closing", "Top-Hat", "Bottom-Hat", "BMF Enhanced"]
step_keys   = ["gray", "normalized", "opening", "closing", "top_hat", "bottom_hat", "enhanced"]
cols = st.columns(7)
for col, key, label in zip(cols, step_keys, step_labels):
    col.image(steps[key], use_column_width=True, clamp=True)
    col.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)

# Quality metrics
metrics = bmf_quality_metrics(steps["gray"], steps["enhanced"])
m_cols  = st.columns(len(metrics))
for mc, (k, v) in zip(m_cols, metrics.items()):
    mc.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{v:.2f}</div>
      <div class="metric-key">{k}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

bmf_img = steps["enhanced"]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — SIMVC SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="stage-card simvc">
  <div class="stage-header">
    <span class="stage-num" style="background:#0d2820;color:#00e5c0;">2</span>
    SIMVC — Semantic Invariant Multi-View Clustering Segmentation
  </div>
""", unsafe_allow_html=True)

with st.spinner("Running SIMVC segmentation …"):
    simvc  = SemanticInvariantMultiViewClustering(
        n_clusters=simvc_clusters, method=simvc_method, working_size=simvc_wsize
    )
    result = simvc.fit_predict(bmf_img)

seg_panels = [
    ("Original", raw_img, False),
    ("BMF Enhanced", bmf_img, False),
    ("Cluster Map", result["cluster_map"], True),
    ("ROI Mask", result["mask"].astype(np.uint8) * 255, False),
    ("Contour Overlay", result["overlay"], False),
]
s_cols = st.columns(5)
for sc, (label, img, is_cmap) in zip(s_cols, seg_panels):
    if is_cmap:
        # Colorize cluster map
        norm_cmap = (img / max(img.max(), 1) * 255).astype(np.uint8)
        colored   = cv2.applyColorMap(norm_cmap, cv2.COLORMAP_TURBO)
        sc.image(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), use_column_width=True)
    elif isinstance(img, np.ndarray) and img.ndim == 3:
        sc.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img,
                 use_column_width=True)
    else:
        sc.image(img, use_column_width=True, clamp=True)
    sc.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)

# Clustering metrics
if result["metrics"]:
    m_cols2 = st.columns(len(result["metrics"]))
    for mc, (k, v) in zip(m_cols2, result["metrics"].items()):
        val_str = f"{v:.3f}" if not (v != v) else "n/a"  # nan check
        mc.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{val_str}</div>
          <div class="metric-key">{k}</div>
        </div>""", unsafe_allow_html=True)

roi_area = int(result["mask"].sum())
st.markdown(f"""
<div class="info-box">
  📐  ROI area: <strong>{roi_area:,}</strong> pixels &nbsp;·&nbsp;
  Selected cluster: <strong>{result['roi_cluster']}</strong> &nbsp;·&nbsp;
  Method: <strong>{simvc_method.upper()}</strong> &nbsp;·&nbsp;
  k = <strong>{simvc_clusters}</strong>
</div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

seg_img = result["segmented"]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="stage-card pred">
  <div class="stage-header">
    <span class="stage-num" style="background:#2a1c08;color:#ffb347;">3</span>
    Deep Neural Classification
  </div>
""", unsafe_allow_html=True)

# Pipeline input selection
if "raw" in selected_model:
    final_img    = raw_img
    active_steps = ["RAW INPUT", "→", "MODEL"]
elif "simvc" in selected_model:
    final_img    = seg_img
    active_steps = ["RAW INPUT", "→", "BMF", "→", "SIMVC", "→", "MODEL"]
else:
    final_img    = bmf_img
    active_steps = ["RAW INPUT", "→", "BMF", "→", "MODEL"]

flow_html = ""
for step in active_steps:
    if step == "→":
        flow_html += '<span class="pf-arrow">→</span>'
    elif step == "MODEL":
        flow_html += f'<span class="pf-step pf-active">{step}</span>'
    else:
        flow_html += f'<span class="pf-step">{step}</span>'

st.markdown(f'<div class="pipeline-flow">{flow_html}</div>', unsafe_allow_html=True)

if not is_loaded or model is None:
    st.error("⚠ Model failed to load — classification unavailable. Check sidebar for details.")
else:
    try:
        with torch.no_grad():
            tensor = torch.from_numpy(final_img).float().unsqueeze(0).unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        class_names = ["Normal", "Pneumonia"]
        pred_idx    = int(np.argmax(probs))
        confidence  = probs[pred_idx] * 100

        left, right = st.columns([1, 1])

        with left:
            is_pneumonia = pred_idx == 1
            dx_class     = "dx-pneumonia" if is_pneumonia else "dx-normal"
            icon         = "🦠" if is_pneumonia else "✓"
            label_color  = "#ff5a6e" if is_pneumonia else "#00e5c0"
            st.markdown(f"""
            <div class="{dx_class}">
              <div style='font-size:2.5rem;margin-bottom:.5rem;'>{icon}</div>
              <div class="dx-label" style="color:{label_color};">{class_names[pred_idx]}</div>
              <div class="dx-conf">Confidence: <span>{confidence:.2f}%</span></div>
              <div class="dx-conf" style="margin-top:.3rem;">Model: <span>{arch_label}</span></div>
            </div>""", unsafe_allow_html=True)

        with right:
            st.markdown("<div style='padding:.5rem 0;'>", unsafe_allow_html=True)
            for i, (cname, prob) in enumerate(zip(class_names, probs)):
                bar_color  = "#ff5a6e" if i == 1 else "#00e5c0"
                active_bar = "font-weight:600;color:#ccd9f0;" if i == pred_idx else ""
                st.markdown(f"""
                <div class="prob-row">
                  <div class="prob-label" style="{active_bar}">{cname}</div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill"
                         style="width:{prob*100:.1f}%;background:{bar_color};"></div>
                  </div>
                  <div class="prob-pct" style="{active_bar}">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box" style="margin-top:1rem;">
              Input tensor: <strong>1 × 1 × 224 × 224</strong> &nbsp;·&nbsp;
              Device: <strong>{'CUDA' if device.type == 'cuda' else 'CPU'}</strong>
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Feed-forward error: {e}")
        st.info("Verify that the selected model architecture matches the weights file.")

st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
  CXR · Neural Diagnostics &nbsp;·&nbsp;
  BMF Preprocessing &nbsp;·&nbsp; SIMVC Multi-View Segmentation &nbsp;·&nbsp;
  MDASNN / SimpleCNN Classification &nbsp;·&nbsp;
  For research purposes only — not a medical device.
</div>
""", unsafe_allow_html=True)
