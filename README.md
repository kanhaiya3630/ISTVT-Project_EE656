# Interpretable Spatial-Temporal Video Transformer (ISTVT) for Deepfake Detection

## Overview

This repository contains our EE656 course project on **Interpretable Spatial-Temporal Video Transformer (ISTVT)** for Deepfake Detection.

The project explores a transformer-based architecture that jointly models **spatial** and **temporal** inconsistencies in videos while providing interpretable visual explanations of the model's predictions using Layer-wise Relevance Propagation (LRP).

The work is based on the IEEE paper:

> Zhao et al., "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection," IEEE Transactions on Information Forensics and Security, 2023.

---

## Team

- Kanhaiya Kumar
- Manish Gurjar
- Utkarsh Kashyap
- Amrutha Sidharth

Course: **EE656**

Instructor: **Prof. Nishchal K. Verma**

Indian Institute of Technology Kanpur

---

## Project Objectives

- Understand modern Deepfake detection techniques.
- Study Vision Transformers for video analysis.
- Explore interpretable AI methods for video transformers.
- Analyze spatial and temporal inconsistencies in Deepfake videos.
- Present the complete ISTVT architecture and methodology.

---

## Repository Structure

```
.
├── Code/
│   └── ee656/
│       ├── main.py
│       ├── train.py
│       ├── model.py
│       ├── visualization.py
│       └── ...
├── ISTVT Report.pdf
├── ISTVT Presentation.pptx
└── README.md
```

---

## Methodology

The ISTVT framework consists of:

- Xception backbone for feature extraction
- Spatial Self-Attention
- Temporal Self-Attention
- Self-Subtract Mechanism
- MLP Classification Head
- Layer-wise Relevance Propagation (LRP)

The model separates spatial and temporal attention, improving both detection performance and interpretability.

---

## Datasets Studied

- FaceForensics++
- Celeb-DF
- DFDC
- FaceShifter
- DeeperForensics

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib

---

## Repository Contents

- 📄 Detailed Project Report
- 📊 Project Presentation
- 💻 Source Code
- 📚 Literature Review
- 🧠 Model Architecture
- 📈 Experimental Analysis

---

## Results

The study demonstrates how decomposed spatial-temporal attention and the self-subtract mechanism improve Deepfake detection while producing interpretable attention heatmaps.

---

## References

C. Zhao, C. Wang, G. Hu, H. Chen, C. Liu and J. Tang,

**"ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection."**

IEEE Transactions on Information Forensics and Security (TIFS), 2023.

DOI:
https://doi.org/10.1109/TIFS.2023.3239223

---

## License

This repository is intended for educational and research purposes only.
