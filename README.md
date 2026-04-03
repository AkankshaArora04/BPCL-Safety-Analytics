# Industrial Surveillance AI System

## Overview
This project focuses on building a real-time industrial surveillance system capable of analyzing visual data from images and video streams. The system performs object detection and applies rule-based logic to identify potential safety violations and monitor restricted zones.

The objective is to simulate a practical surveillance setup where automated detection assists in improving safety compliance and situational awareness in industrial environments.

## Key Features
- Object detection on images using a YOLOv8-based model
- Frame-by-frame analysis of video inputs for continuous monitoring
- Rule-based alert generation for safety-related observations
- Zone-based intrusion detection using spatial constraints
- Integration of multiple datasets for comparative analysis of machine learning models

## Methodology
The system processes input data in two stages. First, a pre-trained object detection model identifies relevant entities within the scene. In the second stage, a rule-based layer interprets these detections to generate meaningful alerts.

For video inputs, frames are processed sequentially to simulate real-time monitoring. A predefined region of interest is used to detect potential intrusions based on object position.

## Tech Stack
- Python
- Streamlit (for interface)
- YOLOv8 (Ultralytics)
- OpenCV
- Scikit-learn
- Pandas, NumPy

## Project Structure
- `app/` – Streamlit application interface
- `src/` – Core logic including detection and alert generation
- `data/` – Input datasets
- `results/` – Sample outputs and visual results

## Results
The system successfully detects objects in both static images and video streams. It is capable of highlighting detected entities and generating alerts based on predefined safety conditions and spatial rules.

Sample outputs demonstrating detection and alert behavior are available in the `results/` directory.

## Limitations
The current implementation uses a pre-trained model that does not include specialized classes such as safety helmets. As a result, safety compliance detection is approximated using rule-based logic rather than dedicated classification.

## Future Work
- Training a custom model for safety gear detection (e.g., helmets, vests)
- Incorporating object tracking for persistent monitoring across frames
- Enhancing real-time performance for live surveillance feeds
- Extending the system to handle satellite or aerial imagery

## Author
Akanksha Arora
