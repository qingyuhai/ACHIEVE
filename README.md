# ACHIEVE: Dual-Mode Illumination-Aware Perception Framework in Unevenly Illuminated Environments

[![Conference](https://img.shields.io/badge/ICRA-2026-blue)]()
[![Platform](https://img.shields.io/badge/UAV-Embedded-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

ACHIEVE is a **lightweight dual-mode visual SLAM framework** designed for UAVs operating in **unevenly illuminated indoor environments**.  
It dynamically adapts to bright and dark regions using:
- **Geometry-based VO** for well-lit areas.
- **Neural edge-based VO** (RAW input, weak supervision) for dark zones.
- **Graph-based fusion** with assistant keyframes and weighted optimization for seamless transitions.

---

## 🚀 Features
- Illumination-aware VO switching
- RAW-image edge detection network (weakly supervised training)
- Graph-based assistant keyframe fusion
- Lightweight design for embedded devices (Jetson Xavier NX, Orin Nano)
- Real-time performance (~29 FPS) with low memory usage

---

## 📂 Repository Structure
```
ACHIEVE/
│── src/               # Core implementation
│   ├── vo/            # Visual odometry modules
│   ├── fusion/        # Graph-based fusion & keyframe handling
│   ├── network/       # Neural edge detector (training + inference)
│   └── utils/         # Helpers (logging, plotting, configs)
│
│── scripts/           # Running pipelines
│   ├── preprocess.py  # Convert datasets (RAW, darkening)
│   ├── train.py       # Train edge detector
│   ├── test.py        # Evaluate on datasets
│   └── run_slam.py    # Launch ACHIEVE SLAM pipeline
│
│── data/              # Placeholder for datasets (not included)
│   ├── TUM/           # TUM RGB-D dataset (darkened variants)
│   ├── testbed/       # Custom UAV testbed sequences
│   └── LOL/           # Optional low-light datasets (LOL-v1/v2)
│
│── results/           # Experimental logs & output
│   ├── figures/       # Plots & visualizations
│   └── tables/        # Numerical results
│
│── README.md          # Project documentation
│── requirements.txt   # Python dependencies
└── LICENSE
```

---

## ⚙️ Installation
```bash
git clone https://github.com/qingyuhai/ACHIEVE.git
cd ACHIEVE
conda create -n achieve python=3.8
conda activate achieve
pip install -r requirements.txt
```

**Main dependencies**:
- PyTorch ≥ 1.9
- OpenCV ≥ 4.5
- NumPy, SciPy, Matplotlib
- evo (for SLAM evaluation)

---

## 📊 Datasets
1. **TUM RGB-D**  
   - Download from [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset)  
   - Use `scripts/preprocess.py` to generate uneven illumination variants.

   ```bash
   python scripts/preprocess.py --dataset TUM --darken --noise
   ```

2. **Custom UAV Testbed**  
   - Real-world sequences collected in 1:100 scale lab testbed.  
   - Contains bright/dark zones with ground-truth motion capture (Vicon).  

3. **LOL-v1/v2 (optional)**  
   - For additional low-light training pairs.  
   - [LOL dataset link](https://daooshee.github.io/BMVC2018website/)

---

## 🏃 Usage
### Train Edge Detector
```bash
python scripts/train.py --dataset data/TUM --epochs 50 --batch_size 16
```

### Run ACHIEVE SLAM
```bash
python scripts/run_slam.py --dataset data/testbed/sequence_01
```

### Evaluate Results
```bash
python scripts/test.py --pred results/seq01_traj.txt --gt data/testbed/seq01_gt.txt
```

---

## 📈 Evaluation Metrics
- **ATE (Absolute Trajectory Error, m)**
- **RPE (Relative Pose Error, m)**
- **RE (Rotation Error, deg/m)**
- **FPS (runtime speed)**
- **Memory footprint (MB/GB)**

---

## 🔬 Experiments
We report:
- Comparisons against **ORB-SLAM3, DSO, VINS-Mono, DROID-SLAM, DarkSLAM, CCM-SLAM**  
- Ablation studies on:
  - RAW vs RGB input
  - Assistant keyframes
  - Weighted GBA  
- Runtime breakdown (perception, fusion, optimization)

---

## 📌 Citation
If you use this work, please cite:
```bibtex
@inproceedings{ACHIEVE2026,
  title={ACHIEVE: Dual-Mode Illumination-Aware Perception Framework in Unevenly Illuminated Environments},
  author={xxx, xxx, xxx, xxx, et al},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

---

## 📜 License
This project is released under the MIT License.

