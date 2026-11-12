# Dual-Teacher Adaptive Agreement (DTAA)

PyTorch implementation of the **Dual-Teacher Adaptive Agreement (DTAA)** framework for semi-supervised federated segmentation of multi-organ abdominal trauma CT scans.

## Abstract & Research Motivation

In acute abdominal polytrauma care, contrast-enhanced computed tomography (CT) is vital for rapid triage. Deep learning approaches in distributed hospital settings face three major obstacles:
1. **Severe Label Scarcity:** Expert radiological voxel-level annotations are available for fewer than 15% of scans, while unannotated scans are abundant.
2. **Scanner Distribution Shift (Non-IID Data):** Heterogeneous hospital scanners (Siemens, GE, Philips, Canon) introduce significant domain variation in slice thickness, reconstruction kernels, and noise.
3. **Pseudo-Label Confirmation Bias Index (CBI):** Standard self-training and single-teacher semi-supervised models accumulate erroneous predictions over federated rounds, causing model degradation.

DTAA addresses these challenges by introducing a **Dual-Teacher Triad**: decoupling global anatomical priors from local scanner noise, gating pseudo-labels through a Total Variation Distance consensus metric, and annealing thresholds via a cosine curriculum schedule.

---

## Key Framework Components

- **Trauma U-Net Backbone:** Parameter-efficient 509,238-parameter architecture utilizing Group Normalization ($G=8$) to ensure numerical stability under small clinical batch sizes ($B \in \{2, 4, 8\}$) without BatchNorm drift.
- **Dual-Teacher Consensus Gating:** Evaluates agreement between the Global Anchor Teacher ($\theta_G$) and the Local Scanner EMA Teacher ($\theta_L$):
  - Consensus metric: $\mathcal{A}(v) = 1 - \frac{1}{2} \sum_{c=1}^C |p_g^{(c)}(v) - p_l^{(c)}(v)| \in [0, 1]$
  - Mutual geometric confidence: $\mathcal{C}(v) = \sqrt{\max_c p_g^{(c)}(v) \cdot \max_c p_l^{(c)}(v)}$
  - Agreement gate: accepts pseudo-label only when $\mathcal{A}(v) \cdot \mathcal{C}(v) \ge \tau(t)$
- **Cosine Annealing Curriculum $\tau(t)$:** Dynamically shifts filtering from conservative precision ($\tau_{\max} = 0.95$) to inclusive boundary supervision ($\tau_{\min} = 0.70$).
- **Compound Objective:** Class-weighted Focal Loss combined with Soft Dice Loss and a Kullback-Leibler (KL) manifold consistency regularizer.
- **Dynamic Onboarding Protocol:** 5-step integration protocol for new hospital nodes with cryptographic SHA-256 weight verification and zero catastrophic forgetting.

---

## Empirical Benchmarks (RSNA Trauma Cohort)

Benchmarked across 4 heterogeneous hospital clients with non-IID label ratios (80%, 25%, 15%, 8%):

- **Mean Foreground Dice (DSC):** **81.6%** (vs. 68.4% for supervised FedAvg and 72.3% for FixMatch-FedAvg).
- **Active Extravasation (Arterial Bleed):** **61.8% DSC** (+30.7% absolute improvement over FedAvg's 31.1%).
- **Confirmation Bias Index (CBI):** Reduced by **77.2%** (from 29.8% down to 6.8%).
- **95th Percentile Hausdorff Distance (HD95):** **7.4 mm** (vs. 14.2 mm for FedAvg).
- **Resource Footprint:** 2.04 MB parameter sync payload, 2.1 GB peak VRAM on commodity consumer GPUs.

---

## Repository Structure

```
dtaa-federated-trauma-ct/
├── configs/
│   └── default_config.yaml            # Hyperparameters and curriculum schedules
├── demo/
│   └── app.py                         # Streamlit interactive diagnostic and defense app
├── dtaa/
│   ├── core/                          # Consensus, dual-teacher EMA, losses, and metrics
│   ├── data/                          # HU calibration, 6-feature spatial maps, and partitioners
│   ├── federated/                     # Client, central server, and dynamic onboarding
│   └── models/                        # 509k Trauma U-Net with Group Normalization (G=8)
├── notebooks/
│   ├── DTAA_Architecture_Verification.ipynb
│   └── DTAA_End_to_End_Walkthrough.ipynb
├── tests/
│   └── test_models.py                 # Automated unit tests
└── requirements.txt
```

---

## Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/HarbdhulQuadri/dtaa-federated-trauma-ct.git
cd dtaa-federated-trauma-ct
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python -m unittest tests/test_models.py
```

### 3. Launch Interactive Defense Demo
```bash
streamlit run demo/app.py
```

---

## Citation & Academic Attribution

```bibtex
@mastersthesis{adegbiji2026dtaa,
  title={A Dual-Teacher Adaptive Agreement Strategy for Semi-Supervised Federated Learning in Abdominal Trauma CT Segmentation},
  author={Adegbiji, Abdulquadri Olanrewaju},
  school={University of Lagos, Department of Computer Science},
  year={2026},
  note={Supervisor: Dr. F.O. Alamu}
}
```

* **Candidate:** Abdulquadri Olanrewaju Adegbiji  
* **Supervisor:** Dr. F.O. Alamu  
* **Department:** Department of Computer Science, University of Lagos, Nigeria  
