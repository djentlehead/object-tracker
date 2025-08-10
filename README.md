# Simplified Object Tracker

A Python-based object tracking tool using OpenCV's tracking algorithms.  
This project implements a hybrid CSRT + MOSSE tracking approach for a balance of **accuracy** and **speed**, with a clean, interactive UI.

## Features
- **Hybrid Tracking**: CSRT for high accuracy, MOSSE for fast intermediate updates.
- **Multiple Tracker Types**: Switch between CSRT, MOSSE, KCF, BOOSTING, MIL, and TLD.
- **Adaptive Confidence Scoring**: Confidence adjusts based on image sharpness.
- **Enhanced UI**:
  - Real-time FPS display
  - Trail visualization of object path
  - Tracking statistics (frames processed, lost count)
  - On-screen controls panel
- **JSON Export**: Optionally save tracking data for later analysis.

## Requirements
- Python 3.8+
- OpenCV (`pip install opencv-contrib-python`)
- NumPy (`pip install numpy`)

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/simplified-object-tracker.git
   cd simplified-object-tracker
