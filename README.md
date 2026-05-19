# Vision-based Crosswalk Risk Warning System

## Project Overview

This project aims to build a real-time computer vision prototype for detecting pedestrian-vehicle risk at crosswalks.

The system analyzes CCTV-style crosswalk video using object detection, object tracking, crosswalk ROI analysis, traffic light recognition, and risk scoring. When a potentially dangerous situation is detected, the system generates visual warnings and simulates an audio warning that could be connected to a speaker installed near a traffic light pole.

The main goal is not only to detect pedestrians and vehicles, but also to understand the crosswalk scene and estimate potential risk in real time.

## Motivation

Crosswalk safety is an important issue in urban environments such as Seoul. Pedestrians, vehicles, bicycles, motorcycles, and traffic lights interact dynamically around crosswalks. A vision-based warning system can help monitor these interactions and provide early warnings when a dangerous situation is likely to occur.

This project is inspired by smart crosswalk systems and AI-based pedestrian safety services. It is designed as an academic prototype for a Computer Vision term project.

## Main Features

- Detect pedestrians and vehicles using YOLO
- Detect crosswalk or define a manual crosswalk ROI
- Track pedestrians and vehicles using DeepSORT or ByteTrack
- Analyze object movement and trajectory near the crosswalk
- Recognize traffic light state such as red, yellow, or green
- Estimate pedestrian-vehicle risk level
- Generate visual warning overlays on video
- Simulate audio warning for high-risk situations
- Save event logs and high-risk screenshots
- Provide demo results through README and output video

## Planned Pipeline

Input video or camera stream  
→ Object detection  
→ Object tracking  
→ Crosswalk ROI analysis  
→ Traffic light state recognition  
→ Trajectory-based risk analysis  
→ Visual/audio warning  
→ Output video and event log

## Planned Technologies

- Python
- OpenCV
- YOLO
- DeepSORT or ByteTrack
- NumPy
- Streamlit
- Google Colab GPU for training or testing if needed

## Expected Output

The final project will include:

- Source code
- README documentation
- Demo video or GIF
- Screenshots of normal and high-risk situations
- Event log file
- Explanation of risk analysis logic
- References to used models, datasets, and open-source code

## Current Status

This repository is created for the Computer Vision term project planning assignment.  
The project name and detailed implementation may be updated during the project development.