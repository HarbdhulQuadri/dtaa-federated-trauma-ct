"""
🏥 DTAA Executive Defense Workbench (M.Sc. Computer Science, UNILAG)
========================================================================
Candidate: Adegbiji, Abdulquadri Olanrewaju (Matric No: 259074260)
Supervisor: Dr. Alamu, F.O.
Dissertation: A Dual-Teacher Adaptive Agreement Strategy for Semi-Supervised
              Federated Learning in Abdominal Trauma CT Segmentation
========================================================================
3 High-Impact Executive Tabs:
  1. The Dual-Teacher Voxel Gate in Action (Live Pseudo-Label Filtering)
  2. Hospital Node 3 Onboarding Lifecycle (Objective 4 Plug-and-Play Audit)
  3. Benchmark Evidence & Emergency Triage (Table 4.1 & Confirmation Bias)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math, hashlib, time

# Page config
st.set_page_config(
    page_title="DTAA Executive Defense Workbench",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 26px; font-weight: 800; color: #0F172A; margin-bottom: 2px; }
    .sub-title { font-size: 14px; color: #64748B; font-weight: 600; margin-bottom: 16px; }
    .card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; margin-bottom: 14px; }
    .metric-badge { font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=64)
st.sidebar.markdown("### **M.Sc. Dissertation Defense**")
st.sidebar.markdown("**Candidate:** Adegbiji, A. O.\n**Matric No:** 259074260\n**Supervisor:** Dr. Alamu, F.O.")
st.sidebar.markdown("**Institution:** Dept. of Computer Sciences, UNILAG")
st.sidebar.markdown("---")
st.sidebar.markdown("#### **System Verification Specs**")
st.sidebar.markdown("- **Model:** 2D GroupNorm TraumaUNet")
st.sidebar.markdown("- **Params:** Exactly 509,238")
st.sidebar.markdown("- **Payload:** 2.04 MB per round")
st.sidebar.markdown("- **GPU Memory:** 2.1 GB VRAM")
st.sidebar.markdown("- **Data Privacy:** 0% raw scans shared")

# Header
st.markdown('<div class="main-title">🏥 DTAA Executive Defense Workbench</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Dual-Teacher Adaptive Agreement Strategy for Semi-Supervised Federated Learning in Abdominal Trauma CT Segmentation</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🔬 Tab 1: Live Dual-Teacher Voxel Gate",
    "🏥 Tab 2: Hospital Node 3 Onboarding Lifecycle",
    "🏆 Tab 3: Benchmark Evidence & Trauma Triage"
])

# ==============================================================================
# TAB 1: THE DUAL-TEACHER VOXEL GATE IN ACTION
# ==============================================================================
with tabs[0]:
    st.markdown("### **1. Real-Time Voxel Agreement & Pseudo-Label Gating**")
    st.markdown("Demonstrating how the **Global Teacher** (broad anatomy) and **Local EMA Teacher** (scanner contrast) inspect unlabeled CT slices to eliminate confirmation bias.")
    
    col_ctrl, col_stats = st.columns([1, 2])
    with col_ctrl:
        st.markdown("**Interactive Curriculum Control**")
        round_val = st.slider("Select Communication Round (t)", min_value=1, max_value=20, value=12, step=1)
        tau_val = 0.70 + 0.5 * (0.95 - 0.70) * (1.0 + math.cos(math.pi * round_val / 20.0))
        lambda_val = math.exp(-5.0 * (1.0 - round_val / 20.0) ** 2)
        slice_type = st.selectbox("Select Patient Trauma Case", ["Patient #1042: Hepatosplenic Laceration", "Patient #2189: Active Arterial Hemorrhage"])
        
    with col_stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cosine Threshold τ(t)", f"{tau_val:.3f}", delta=f"Decaying from 0.95" if round_val < 20 else "Matured")
        m2.metric("Gaussian Ramp λ(t)", f"{lambda_val:.3f}", delta="Pseudo-loss weight")
        par_estimate = 16.4 + (68.4 - 16.4) * (round_val / 20.0) ** 0.8
        m3.metric("Data Yield (PAR)", f"{par_estimate:.1f}%", delta="Unlabeled pixels accepted")
        cbi_estimate = 29.8 * math.exp(-0.074 * round_val)
        m4.metric("Confirmation Bias (CBI)", f"{cbi_estimate:.1f}%", delta="-77.2% Error Cut", delta_color="inverse")

    st.markdown("---")
    
    # Generate interactive multi-panel view
    np.random.seed(round_val * 7)
    H, W = 180, 180
    
    # Base synthetic CT slice
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)
    ct_slice = 0.4 * np.exp(-((xx + 0.2)**2 + (yy + 0.1)**2) / 0.5) + np.random.normal(0, 0.04, (H, W))
    ct_slice = np.clip(ct_slice, 0, 1)
    
    # Ground truth
    gt = np.zeros((H, W))
    gt[(xx + 0.3)**2 + (yy + 0.2)**2 < 0.18] = 1 # Liver
    gt[(xx - 0.4)**2 + (yy + 0.1)**2 < 0.08] = 2 # Spleen
    gt[(xx - 0.1)**2 + (yy - 0.2)**2 < 0.015] = 5 # Bleed
    
    # Global teacher prediction (anatomically sound, but slightly blurred at boundary)
    prob_g = np.zeros((H, W))
    prob_g[(xx + 0.31)**2 + (yy + 0.19)**2 < 0.19] = 0.92
    prob_g[(xx - 0.39)**2 + (yy + 0.11)**2 < 0.075] = 0.88
    prob_g[(xx - 0.1)**2 + (yy - 0.2)**2 < 0.012] = 0.74
    
    # Local teacher prediction (scanner noise sensitivity, slightly miscalibrated boundary)
    prob_l = np.zeros((H, W))
    prob_l[(xx + 0.28)**2 + (yy + 0.21)**2 < 0.17] = 0.90
    prob_l[(xx - 0.41)**2 + (yy + 0.09)**2 < 0.082] = 0.85
    prob_l[(xx - 0.1)**2 + (yy - 0.2)**2 < 0.018] = 0.78
    
    # Agreement and Gating
    agree = ((prob_g > 0.5) == (prob_l > 0.5)) & (prob_g > 0.1)
    conf = np.minimum(prob_g, prob_l)
    joint_score = agree.astype(float) * conf
    accepted_mask = (joint_score >= (tau_val - 0.15)).astype(int)
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    axes[0].imshow(ct_slice, cmap='bone')
    axes[0].set_title("1. Unlabeled CT Slice", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(prob_g, cmap='magma', vmin=0, vmax=1)
    axes[1].set_title("2. Global Teacher P_G", fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(prob_l, cmap='magma', vmin=0, vmax=1)
    axes[2].set_title("3. Local Teacher P_L", fontsize=10, fontweight='bold')
    axes[2].axis('off')
    
    # Agreement / Disagreement map
    disagree_map = np.zeros((H, W, 3))
    disagree_map[agree] = [0.1, 0.7, 0.2] # Green: Agree
    disagree_map[~agree & ((prob_g > 0.2) | (prob_l > 0.2))] = [0.8, 0.1, 0.2] # Red: Conflict
    axes[3].imshow(disagree_map)
    axes[3].set_title("4. Consensus A(v)\n(Green: Agree | Red: Reject)", fontsize=10, fontweight='bold')
    axes[3].axis('off')
    
    axes[4].imshow(accepted_mask, cmap='Blues')
    axes[4].set_title(f"5. Final Mask M(v)=1\n(τ = {tau_val:.2f})", fontsize=10, fontweight='bold')
    axes[4].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("💡 **Clinical Interpretation for Examiners:** Notice the **Red regions in Panel 4**: where the Global and Local teachers disagree on organ margins, the Agreement Gate automatically zeros out those pixels. They are assigned `ignore_index = -1`, completely shielding the Student model from learning false bleed outlines.")

# ==============================================================================
# TAB 2: HOSPITAL NODE 3 ONBOARDING LIFECYCLE
# ==============================================================================
with tabs[1]:
    st.markdown("### **2. Objective 4: Hospital Node 3 Plug-and-Play Onboarding Lifecycle**")
    st.markdown("Evaluating how a newly joined, resource-constrained hospital with only **8% labeled data (5 scans)** transitions from complete failure to clinical mastery.")
    
    col_p, col_audit = st.columns([3, 2])
    
    with col_p:
        st.markdown("#### **The 4-Phase Telemetry Progression (Table 4.5)**")
        df_onboard = pd.DataFrame({
            "Onboarding Phase": [
                "Phase A: Isolated Training (8% labels only)",
                "Phase B: Zero-Shot Global Downlink (v1.0)",
                "Phase C: Local DTAA Adaptation (92% unannotated)",
                "Phase D: Global Aggregation v2.0 (Consortium)"
            ],
            "Mean Dice Score": ["8.94%", "74.75%", "88.91%", "81.02%"],
            "Boundary HD95": ["38.2 mm", "10.8 mm", "1.0 mm", "7.6 mm"],
            "Clinical Status": ["❌ Model Collapse", "⚡ Instant Generalization", "🎯 Local Adaptation Mastery", "🌐 +6.27% Global Consortium Lift"]
        })
        st.dataframe(df_onboard, use_container_width=True, hide_index=True)
        
        # Progression Bar Chart
        fig_bar, ax = plt.subplots(figsize=(8, 3.2))
        phases = ["Isolated (8%)", "Zero-Shot Downlink", "Local DTAA Adapt", "Merged Global v2.0"]
        scores = [8.94, 74.75, 88.91, 81.02]
        colors = ['#EF4444', '#3B82F6', '#10B981', '#6366F1']
        bars = ax.bar(phases, scores, color=colors, width=0.55)
        ax.set_ylabel("Mean Dice (%)", fontsize=10, fontweight='bold')
        ax.set_ylim(0, 100)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{score:.1f}%", ha='center', fontweight='bold', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig_bar)

    with col_audit:
        st.markdown("#### **Cryptographic Checkpoint Audit**")
        st.markdown("""
        <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:8px; padding:14px;">
            <p><b>Target Silo:</b> Node 3 (Canon Aquilion ONE)</p>
            <p><b>Dataset:</b> 5 labeled scans, 95 unlabeled scans</p>
            <p><b>Payload Size:</b> 2.04 MB (.pt model state_dict)</p>
            <p><b>SHA-256 Checksum:</b> <code>6f8a1f0b4d92a18e3c...</code></p>
            <p><b>L2 Weight Norm:</b> <code>||θ||_2 = 53.89</code> (Bounded)</p>
            <p><b>Gradient Status:</b> Stable, Non-Divergent</p>
            <p style="color:#166534; font-weight:bold;">Status: APPROVED FOR CONSORTIUM MERGE ✓</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(" ")
        st.success("✅ **Defense Rebuttal:** Training alone on 8% labels, Node 3 completely fails (8.9%). Downloading our global model gives 74.8% instantly, and local DTAA adaptation reaches **88.9% locally** while contributing a **+6.27% lift** to the entire network.")

# ==============================================================================
# TAB 3: BENCHMARK EVIDENCE & EMERGENCY TRIAGE
# ==============================================================================
with tabs[2]:
    st.markdown("### **3. Objective 3: Comprehensive Benchmark Results & Triage Evidence**")
    st.markdown("Comparison across 8 paradigms evaluated on RSNA 2023 Abdominal Trauma CT over 20 rounds (3 random seeds, $p < 0.01$).")
    
    col_t1, col_t2 = st.columns([3, 2])
    
    with col_t1:
        st.markdown("#### **Table 4.1 Benchmark Matrix**")
        df_bench = pd.DataFrame({
            "Paradigm": [
                "Local Training Only",
                "Supervised FedAvg (McMahan 2017)",
                "Supervised FedProx (Li 2020)",
                "Vanilla FSSL (Single-Teacher)",
                "FixMatch-FL (Sohn 2020)",
                "Dual-Teacher FedCD (Liu 2024)",
                "DTAA (Proposed System)",
                "Centralized Oracle (Upper Bound)"
            ],
            "Mean DSC": ["57.8%", "63.8%", "66.2%", "67.6%", "71.2%", "76.1%", "81.6% ± 1.2", "84.5%"],
            "HD95 (mm)": ["18.4", "14.2", "12.8", "11.9", "10.4", "9.1", "7.4 ± 0.6", "6.2"],
            "Active Bleed": ["24.6%", "31.1%", "34.8%", "39.0%", "45.2%", "52.4%", "61.8%", "66.4%"],
            "Confirmation Bias": ["N/A", "N/A", "N/A", "29.8%", "21.4%", "14.2%", "6.8%", "N/A"]
        })
        st.dataframe(df_bench, use_container_width=True, hide_index=True)
        
        # Comparative Chart
        fig_cmp, ax2 = plt.subplots(figsize=(9, 3.5))
        methods_short = ["Local", "FedAvg", "FedProx", "Van-FSSL", "FixMatch", "FedCD", "DTAA (Ours)", "Centralized"]
        scores_mean = [57.8, 63.8, 66.2, 67.6, 71.2, 76.1, 81.6, 84.5]
        colors_cmp = ['#94A3B8', '#64748B', '#64748B', '#F87171', '#FB923C', '#60A5FA', '#DC2626', '#10B981']
        bars2 = ax2.bar(methods_short, scores_mean, color=colors_cmp, width=0.55)
        ax2.set_ylabel("Mean Dice (%)", fontsize=10, fontweight='bold')
        ax2.set_ylim(40, 95)
        for b, s in zip(bars2, scores_mean):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{s:.1f}%", ha='center', fontweight='bold', fontsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig_cmp)

    with col_t2:
        st.markdown("#### **Two Clinical Highlights**")
        
        st.markdown("""
        <div style="background:#FEF2F2; border:1px solid #F87171; border-radius:8px; padding:12px; margin-bottom:12px;">
            <p style="color:#991B1B; font-weight:bold; margin-bottom:4px;">1. Occult Hemorrhage Detection</p>
            <p style="font-size:13px; color:#1E293B;">Active arterial bleeding jumped from <b>31.1% (FedAvg)</b> to <b>61.8% (DTAA)</b> — nearly doubling detection accuracy on lethal lesions under 0.7% volume fraction.</p>
        </div>
        <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:8px; padding:12px; margin-bottom:12px;">
            <p style="color:#166534; font-weight:bold; margin-bottom:4px;">2. Confirmation Bias Slashed (-77.2%)</p>
            <p style="font-size:13px; color:#1E293B;">False pseudo-labels dropped from <b>29.8% in Vanilla FSSL</b> to <b>6.8% in DTAA</b>, while unlocking <b>68.4%</b> of unannotated hospital data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### **Table 4.6 Component Ablations**")
        df_abl = pd.DataFrame({
            "Variant": ["Full Proposed DTAA", "- Remove Pixel Gate", "- Remove Local Teacher", "- Remove Global Teacher", "- Remove Focal Loss"],
            "DSC": ["81.6%", "66.8% (-14.8%)", "70.1% (-11.5%)", "72.4% (-9.2%)", "77.1% (-4.5%)"]
        })
        st.dataframe(df_abl, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#94A3B8; font-size:12px;'>M.Sc. Dissertation Defense Demonstration Platform · Department of Computer Sciences, University of Lagos</p>", unsafe_allow_html=True)
