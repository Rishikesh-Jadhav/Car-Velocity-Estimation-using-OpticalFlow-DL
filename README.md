# Optical Flow Analysis and Hardware Implementation on Robot Rover for Car Velocity Estimation

## Project Overview

This project combines classic optical flow algorithms (Lucas-Kanade and Farneback) and a deep learning approach (RAFT) for vehicle speed estimation. Implemented on a robot rover equipped with Raspberry Pi 4, Pi Camera, and a custom-built mobile platform, the project explores the integration of computer vision techniques in real-world scenarios.

## Table of Contents
1. [Introduction](#1-introduction)
2. [Algorithms](#2-algorithms)
    - [Lucas-Kanade](#21-algorithm-overview---lucas-kanade)
    - [Farneback Gunnar](#22-algorithm-overview---farneback-gunnar)
    - [RAFT](#23-raft)
    - [YOLO](#24-yolo)
3. [Results](#3-results)
4. [Hardware](#4-hardware)
6. [Challenges](#6-challenges)
7. [Future Work](#7-future-work)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Introduction

The project addresses the critical need for accurate vehicle speed estimation in autonomous vehicles. Optical flow algorithms, both classic and deep learning-based, are applied to video data from onboard cameras. The focus is on practical implementation, including a hardware setup on a robot rover, making the project applicable to real-world scenarios.

---

## 2. Algorithms

### 2.1 Algorithm Overview - Lucas-Kanade

Lucas-Kanade, a sparse optical flow algorithm, tracks feature points between frames. The paper details steps for car velocity estimation, including feature extraction, optical flow calculation, and unit conversion.

![Lucas-Kanade](outputs%20and%20results/lucas%20kanade%20algo.png)

### 2.2 Algorithm Overview - Farneback Gunnar

Farneback Gunnar, a dense optical flow algorithm, estimates flow for every pixel. The process involves grayscale conversion, derivative computation, structure tensor creation, and optical flow computation.

![Farneback](outputs%20and%20results/farneback%20algo.png)

### 2.3 RAFT

Recurrent All-Pairs Field Transforms (RAFT), a deep network architecture, produces multi-scale 4D correlation volumes for pixel pairs, extracting per-pixel characteristics. The method is divided into three stages: feature extraction, visual similarity computation, and iterative updates.

![RAFT](outputs%20and%20results/raft.png)

### 2.4 YOLO

You Only Look Once (YOLO), a real-time object detection system, is integrated into the pipeline for identifying cars and providing bounding boxes for Region of Interest (ROI) determination.

![YOLO Detection](outputs%20and%20results/yolo%20detection.png)

---

## 3. Results

### 3.1 Our Pipeline for Car Velocity Estimation 

![Pipeline](outputs%20and%20results/pipeline.png)

### 3.2 Test Results for Lukas and Farne

The project showcases results from Lucas-Kanade, Farneback, and RAFT, including optical flow, feature points, and motion vectors. Live demo results and YOLO integration for ROI determination are presented.

![Test Results](outputs%20and%20results/lukas%20plus%20farne.png)

### 3.3 Final Results with Combined YOLO-RAFT Pipeline

![Results](outputs%20and%20results/test%20results.png)

---

## 4. Hardware

A mobile robot, powered by Raspberry Pi 4, Pi Camera, and a custom-built platform, is designed for video recording and analysis. The section covers hardware specifications, video streaming details, and challenges faced during implementation.

![Robot Rover](outputs%20and%20results/robot%20with%20raspi%20cam.png)

---

## 6. Challenges

This project encountered several challenges that required innovative problem-solving and adaptability:

- Depth Data: Obtaining accurate depth information was hindered by the camera's limitations, impacting velocity calculations and necessitating alternative methods.
- Real-time Processing: Achieving real-time processing on the Raspberry Pi 4 posed computational challenges, requiring optimizations to balance performance with limited hardware capabilities.
- Camera Calibration: Challenges in obtaining precise calibration values impacted velocity accuracy, prompting iterative refinement of calibration techniques.
- YOLO and RAFT Integration: Seamless integration of YOLO for object detection and RAFT for optical flow involved addressing compatibility issues and aligning diverse method outputs.
- Network Latency: Overcoming network latency during live video streaming demanded optimizations for maintaining synchronization with real-time processing.
- Lighting Conditions: Varied lighting conditions affected video quality, necessitating system adjustments for robustness under different scenarios.
- Model Training: Challenges in RAFT model training on the KITTI dataset included parameter tuning and ensuring generalization across diverse scenarios.
- Multi-car Velocity Calculation: Extending velocity calculations to multiple cars required modifications for tracking and distinguishing between vehicles.
- Hardware Limitations: The Raspberry Pi 4's computational limitations posed ongoing challenges, necessitating efficient algorithm adaptation while maintaining performance standards.

## 7. Future Work

We suggest potential avenues for future work, including implementing a robot tracking system, improving velocity calculations, modifying RAFT, and making it lightweight for real-time output, as in our case after deployment the model was too heavy for the Raspi cam and computations slowed the process, and extending the method for tracking multiple cars.

---

## 8. Conclusion

This project provides, emphasizing practical implementation on a robot rover. The combination of classic and deep learning-based optical flow algorithms, coupled with a detailed hardware setup, positions this project as a valuable asset for researchers and practitioners in the field.

## 9. References

1. A. Geiger, P. Lenz, R. Urtasun, "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite," Conference on Computer Vision and Pattern Recognition (CVPR), 2012, [Link](https://link.springer.com/article/10.1007/s00138-012-0435-y).
2. N. Sharmin, R. Brad, "Optimal Filter Estimation for Lucas-Kanade Optical Flow," Sensors, vol. 12, no. 9, pp. 12694–12709, Sep. 2012, [DOI](https://doi.org/10.3390/s120912694).
3. Z. Teed, J. Deng, "RAFT: Recurrent all-pairs field transforms for optical flow," [GitHub](https://github.com/princeton-vl/RAFT).
