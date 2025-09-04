# Multi-Modal Accident Risk Prediction (Stereo Vision + YOLOv8 + LSTM)

> End-to-end pipeline that fuses **stereo vision depth**, **YOLOv8 object detection**, **vehicle speed estimation**, and **driver behavior analysis** into an **LSTM-based risk prediction** system. The repo includes modular notebooks for each stage and a lightweight dashboard for visualization.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Datasets & Paths](#datasets--paths)
- [How to Run](#how-to-run)
  - [1) Object Detection, Depth & Speed](#1-object-detection-depth--speed)
  - [2) Driver Behavior Analysis](#2-driver-behavior-analysis)
  - [3) LSTM Risk Modeling](#3-lstm-risk-modeling)
  - [4) Streamlit Dashboard](#4-streamlit-dashboard)
- [Configuration](#configuration)
- [Expected Results (Reference)](#expected-results-reference)
- [Reproducibility Tips](#reproducibility-tips)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project builds a **multi-stage perception pipeline** for road safety:

1. **Object Detection & Depth:** YOLOv8 detects vehicles per frame; **StereoSGBM** computes disparity to triangulate **depth**.  
2. **Speed Estimation:** Per-object **centroid tracking** estimates speed over time.  
3. **Driver Behavior Analysis:** A CNN classifies in-cabin driver state (e.g., attentive, phone usage).  
4. **Temporal Risk Modeling:** An **LSTM** fuses spatial (depth), temporal (speed), and behavioral (driver state probabilities) to output a **continuous risk score** mapped into **5 risk levels**.  
5. **Visualization:** A **Streamlit** dashboard aggregates outputs, visualizes risk levels, and supports quick scenario analysis.  

---

## Key Features

- **YOLOv8 object detection (Ultralytics)**  
- **StereoSGBM depth estimation (OpenCV)**  
- **Centroid-based multi-object tracking & speed estimation**  
- **CNN driver behavior classifier (~90–92% val. accuracy on reference split)**  
- **LSTM sequence model for 5-level risk scoring (0–1 scaled)**  
- **Streamlit dashboard (`my_dashboard.py`) for interactive visualization**  
- Modular, notebook-driven workflow with saved intermediates (CSV/NPY)  

---
