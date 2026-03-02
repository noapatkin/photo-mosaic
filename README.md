Slit-Scan Panorama Generator

A Python-based tool that transforms video files into multi-perspective panoramas using Slit-Scan (or strip-photography) techniques. 

This project was developed as part of an Image Processing course at the Hebrew University of Jerusalem (HUJI).

📌 Overview

Traditional panoramas (stitching) represent a single moment in time from multiple angles. 
This tool does the opposite: it captures multiple moments in time through a single vertical "slit."

By extracting specific vertical strips from every frame of a video and "sewing" them together, the algorithm creates a visualization of time-space.

View 0: Uses the leftmost strip of every frame (Past/Leading edge).

View N: Uses the rightmost strip of every frame (Future/Trailing edge).

🛠️ Technical FeaturesMotion Estimation: Uses ORB (Oriented FAST and Rotated BRIEF) feature detection and BFMatcher to track camera movement between frames.

Vertical Alignment: Implements a cumulative translation vector (ty) to compensate for hand-shake and vertical drift, ensuring a smooth horizontal horizon.

Motion Smoothing: Applies a Gaussian temporal filter to the y-axis displacement to prevent "staircase" artifacts in the final output.

Dynamic Canvas Mapping: Calculates global dimensions based on cumulative motion to ensure no data is lost during the reconstruction.

🚀 How it WorksPreprocessing: Extracts frames from the input video.Feature Matching: Calculates the Median (dx, dy) shift between consecutive frames.

Slit Extraction: For a given "view index," it crops a strip of width from each frame.

Sequential Stitching: Aligns strips based on their temporal order while adjusting their vertical position to counteract camera shake.

Post-processing: Automatically crops the resulting canvas to remove "dead zones" caused by vertical stabilization.

💻 Setup & Usage

Prerequisites

Python 3.x

OpenCV (cv2)

NumPyPillow (PIL)
