import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# Scikit-Image & Sklearn untuk BMF & SIMVC
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy, label, regionprops
from skimage.morphology import disk, opening, closing, white_tophat, black_tophat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.ndimage import binary_fill_holes

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Pneumonia Detection Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Clinical Theme with Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background & Text Colors */
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }
    
    /* Glassmorphism Expanders & Cards */
    .streamlit-expanderHeader {
        background: rgba(15, 22, 41, 0.8) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(0, 212, 170, 0.2) !important;
        color: #00d4aa !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(15, 22, 41, 0.4) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-top: none !important;
        border-bottom-left-radius: 8px !important;
        border-bottom-right-radius: 8px !important;
        backdrop-filter: blur(10px);
    }
    
    /* Header Gradient */
    .hero-header {
        background: linear-gradient(90deg, #0f1629 0%, #1a2a42 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #00d4aa;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURES
# ==========================================
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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # index 0
            nn.BatchNorm2d(32),                           # index 1
            nn.ReLU(inplace=True),                        # index 2
            nn.MaxPool2d(kernel_size=2, stride=2),        # index 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # index 4
            nn.BatchNorm2d(64),                           # index 5
            nn.ReLU(inplace=True),                        # index 6
            nn.MaxPool2d(kernel_size=2, stride=2),        # index 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # index 8
            nn.BatchNorm2d(128),                          # index 9
            nn.ReLU(inplace=True),                        # index 10
            nn.MaxPool2d(kernel_size=2, stride=2),        # index 11
            nn.Conv2d(128, 192, kernel_size=3, padding=1),# index 12
            nn.BatchNorm2d(192),                          # index 13
            nn.ReLU(inplace=True),                        # index 14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                      # index 0
            nn.Flatten(),                                 # index 1
            nn.Dropout(p=0.2),                            # index 2
            nn.Linear(192, num_classes)                   # index 3
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model(model_name, device):
    if "mdasnn" in model_name.lower():
        if "default" in model_name.lower():
            model = MDASNN(num_classes=2, attention_ratio=8)
        else:
            model = MDASNN(num_classes=2, attention_ratio=13)
    else:
        model = SimpleCNN(num_classes=2)
    
    try:
        model.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading weights: {e}")
    model.to(device)
    model.eval()
    return model

# ==========================================
# 3. BALANCED MORPHOLOGICAL FILTER (BMF)
# ==========================================
class BalancedMorphologicalFilter:
    def __init__(self, kernel_size=5, denoise_method='median'):
        self.kernel = disk(kernel_size)
        self.denoise_method = denoise_method

    def contrast_measure(self, img):
        return img.std()

    def sharpness_laplacian(self, img):
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def apply(self, image, return_steps=False):
        steps = {}
        
        # 1. Grayscale & Normalize
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        steps['gray'] = gray
        norm = rescale_intensity(gray, out_range=(0, 255)).astype(np.uint8)
        steps['normalized'] = norm

        # 2. Denoising
        if self.denoise_method == 'median':
            denoised = cv2.medianBlur(norm, 5)
        else:
            denoised = cv2.bilateralFilter(norm, 9, 75, 75)
        steps['denoised'] = denoised

        # 3. Morphological Operations
        opened = opening(denoised, self.kernel)
        steps['opening'] = opened
        
        closed = closing(opened, self.kernel)
        steps['closing'] = closed
        
        top = white_tophat(closed, self.kernel)
        steps['top_hat'] = top
        
        bottom = black_tophat(closed, self.kernel)
        steps['bottom_hat'] = bottom

        # 4. Balancing Formula: Original + TopHat - BottomHat
        enhanced = cv2.add(closed, top)
        enhanced = cv2.subtract(enhanced, bottom)
        steps['enhanced'] = enhanced
        
        if return_steps:
            return enhanced, steps
        return enhanced

    def get_quality_metrics(self, original, enhanced):
        return {
            'PSNR (dB)': round(psnr(original, enhanced), 2),
            'SSIM': round(ssim(original, enhanced, data_range=255), 4),
            'Entropy': round(shannon_entropy(enhanced), 2),
            'Contrast': round(self.contrast_measure(enhanced), 2),
            'Sharpness': round(self.sharpness_laplacian(enhanced), 2)
        }

# ==========================================
# 4. SEMANTIC INVARIANT MULTI-VIEW CLUSTERING
# ==========================================
class SemanticInvariantMultiViewClustering:
    def __init__(self, n_clusters=3, method='kmeans', working_size=128):
        self.n_clusters = n_clusters
        self.method = method
        self.working_size = working_size

    def extract_multi_view_features(self, img):
        # View 1: Intensity
        v_intensity = img.flatten()
        
        # View 2: LBP (Texture)
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform').flatten()
        
        # View 3: Edge (Sobel)
        edge = sobel(img).flatten()
        
        # View 4 & 5: Spatial Coordinates (Center prior)
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
        y_norm = (y / h).flatten()
        x_norm = (x / w).flatten()
        
        features = np.column_stack((v_intensity, lbp, edge, y_norm, x_norm))
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def apply(self, bmf_image):
        # Resize for performance
        h_orig, w_orig = bmf_image.shape
        img_small = cv2.resize(bmf_image, (self.working_size, self.working_size))
        
        # Extract features
        X = self.extract_multi_view_features(img_small)
        
        # Clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        cluster_map = labels.reshape((self.working_size, self.working_size))
        
        # ROI Identification (Heuristic: Lung usually dark but not background edge)
        # Select cluster closest to middle intensity
        centers = kmeans.cluster_centers_[:, 0] # intensity center
        lung_cluster_idx = np.argsort(centers)[len(centers)//2]
        
        roi_mask = (cluster_map == lung_cluster_idx).astype(np.uint8) * 255
        
        # Post-processing
        roi_mask = cv2.resize(roi_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        roi_mask = binary_fill_holes(roi_mask).astype(np.uint8) * 255
        
        # Apply mask to original
        segmented = cv2.bitwise_and(bmf_image, bmf_image, mask=roi_mask)
        
        metrics = {}
        if len(np.unique(labels)) > 1:
            # Subsample for speed
            idx = np.random.choice(X.shape[0], 2000, replace=False)
            metrics['Silhouette'] = round(silhouette_score(X[idx], labels[idx]), 3)
            metrics['Davies-Bouldin'] = round(davies_bouldin_score(X[idx], labels[idx]), 3)
        
        return cluster_map, roi_mask, segmented, metrics

# ==========================================
# 5. MAIN UI APPLICATION
# ==========================================
def main():
    # Hero Section
    st.markdown("""
        <div class="hero-header">
            <h1 style='margin-bottom:0;'>MDASNN-PD-CXRI Dashboard</h1>
            <p style='color: #00d4aa; font-size: 1.1rem; margin-top:0;'>
                Optimized Multi-Dimensional Attention Spiking Neural Network for Pneumonia Detection
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Model Selection")
        available_models = [
            "cnn_plus_bmf_plus_simvc.pth",
            "cnn_plus_bmf.pth",
            "cnn_raw.pth",
            "best_mdasnn_default.pth",
            "best_mdasnn_model.pth"
        ]
        model_choice = st.radio(
            "Select Inference Model:",
            available_models,
            index=0,
            help="Select the trained weights for classification."
        )
        if "cnn" in model_choice:
            st.success("★ Recommended: Matches preprocessing pipeline.")
        else:
            st.warning("⚠ Warning: Potential train/inference mismatch with SIMVC.")
            
        st.divider()
        
        st.subheader("BMF Parameters")
        bmf_kernel = st.slider("Kernel Size", 3, 15, 5, step=2)
        bmf_denoise = st.selectbox("Denoise Method", ["median", "bilateral"])
        
        st.subheader("SIMVC Parameters")
        simvc_clusters = st.slider("Number of Clusters", 2, 6, 3)
        
        st.divider()
        uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])

    # Execution Flow
    if uploaded_file is not None:
        # Load Image
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(image_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Init Tools
        bmf = BalancedMorphologicalFilter(kernel_size=bmf_kernel, denoise_method=bmf_denoise)
        simvc = SemanticInvariantMultiViewClustering(n_clusters=simvc_clusters)
        
        # -----------------------------------------
        # STAGE 1: BMF Preprocessing
        # -----------------------------------------
        st.markdown("### 1. Balanced Morphological Filters (BMF)")
        with st.expander("🔍 View Preprocessing Pipeline & Metrics", expanded=True):
            with st.spinner("Applying BMF..."):
                img_bmf, bmf_steps = bmf.apply(img_gray, return_steps=True)
                bmf_metrics = bmf.get_quality_metrics(img_gray, img_bmf)
                
                # Visualizations
                cols = st.columns(4)
                cols[0].image(bmf_steps['normalized'], caption="Normalized", use_container_width=True)
                cols[1].image(bmf_steps['denoised'], caption="Denoised", use_container_width=True)
                cols[2].image(bmf_steps['top_hat'], caption="White Top-Hat", use_container_width=True)
                cols[3].image(bmf_steps['enhanced'], caption="Final BMF Enhanced", use_container_width=True)
                
                # Metrics
                st.markdown("<br>", unsafe_allow_html=True)
                m_cols = st.columns(5)
                for i, (k, v) in enumerate(bmf_metrics.items()):
                    m_cols[i].metric(label=k, value=v)

        # -----------------------------------------
        # STAGE 2: SIMVC Segmentation
        # -----------------------------------------
        st.markdown("### 2. Semantic Invariant Multi-view Clustering (SIMVC)")
        with st.expander("🧠 View ROI Segmentation & Cluster Maps", expanded=True):
            with st.spinner("Performing Clustering..."):
                cluster_map, roi_mask, img_segmented, simvc_metrics = simvc.apply(img_bmf)
                
                # Contour overlay
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_overlay = cv2.cvtColor(img_bmf.copy(), cv2.COLOR_GRAY2RGB)
                cv2.drawContours(img_overlay, contours, -1, (0, 212, 170), 2)
                
                # Visualizations
                cols = st.columns(3)
                # Map cluster output ke colormap yang cantik
                cluster_viz = (cluster_map * (255 // simvc_clusters)).astype(np.uint8)
                cluster_viz = cv2.applyColorMap(cluster_viz, cv2.COLORMAP_VIRIDIS)
                
                cols[0].image(cluster_viz, caption="Multi-View Cluster Map", use_container_width=True)
                cols[1].image(roi_mask, caption="Lung ROI Mask", use_container_width=True)
                cols[2].image(img_overlay, caption="ROI Boundary Overlay", use_container_width=True)
                
                # Metrics
                if simvc_metrics:
                    st.markdown("<br>", unsafe_allow_html=True)
                    m_cols = st.columns(2)
                    for i, (k, v) in enumerate(simvc_metrics.items()):
                        m_cols[i].metric(label=f"Unsupervised {k}", value=v)

        # -----------------------------------------
        # STAGE 3: Classification Result
        # -----------------------------------------
        st.markdown("### 3. Diagnostic Inference")
        with st.container():
            with st.spinner(f"Running inference via {model_choice}..."):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = load_model(model_choice, device)
                
                # Determine what input to give based on model
                if "raw" in model_choice:
                    input_img = img_gray
                elif "simvc" in model_choice:
                    input_img = img_segmented
                else:
                    input_img = img_bmf
                    
                input_img = cv2.resize(input_img, (224, 224))
                
                try:
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(input_img).float().unsqueeze(0).unsqueeze(0) / 255.0
                        input_tensor = input_tensor.to(device)
                        
                        outputs = model(input_tensor)
                        probs = F.softmax(outputs, dim=1).cpu().numpy()[0] * 100
                        
                    classes = ['Normal', 'Pneumonia']
                    pred_idx = np.argmax(probs)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    classes = ['Normal', 'Pneumonia']
                    probs = [50.0, 50.0]
                    pred_idx = 0
                
                # Plotly Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probs[pred_idx],
                    title = {'text': f"<b>{classes[pred_idx]}</b>", 'font': {'size': 24, 'color': '#00d4aa'}},
                    number = {'suffix': "%", 'font': {'color': 'white'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#00d4aa"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(255, 107, 107, 0.3)"},
                            {'range': [50, 80], 'color': "rgba(255, 167, 38, 0.3)"},
                            {'range': [80, 100], 'color': "rgba(0, 212, 170, 0.2)"}],
                    }
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300, margin=dict(l=20, r=20, t=50, b=20))
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.markdown(f"""
                    <div style="background: rgba(15, 22, 41, 0.8); padding: 20px; border-radius: 10px; border-left: 5px solid #00d4aa; height: 100%;">
                        <h3 style="color: #e2e8f0; margin-top:0;">Pipeline Summary</h3>
                        <p><b>Model:</b> <code>{model_choice}</code></p>
                        <p><b>Preprocessing:</b> Balanced Morphological Filter (BMF) w/ {bmf_denoise} denoise.</p>
                        <p><b>Segmentation:</b> Semantic Invariant Multi-View Clustering ({simvc_clusters} clusters).</p>
                        <hr style="border-color: rgba(255,255,255,0.1);">
                        <p style="color: #a0aec0; font-size: 0.9em;"><i>Disclaimer: This tool is for research and educational purposes only and does not substitute professional medical diagnosis.</i></p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()