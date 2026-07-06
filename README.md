# AI and Machine Learning: Gait Analysis & Feature Extraction 🚶‍♂️🧠

This repository contains a Machine Learning project built with MATLAB. The project focuses on analyzing human gait (walking patterns) using sensor data, extracting key features from data windows, and utilizing a trained Multilayer Perceptron (MLP) neural network for classification.

## 📂 Project Structure

- **`gait_code.m`**: The main execution script that runs the gait analysis pipeline.
- **`extract_features_from_window.m`**: A helper function that processes time-series windows to extract meaningful statistical features.
- **`make_feature_header.m`**: Generates headers/labels for the extracted features.
- **`trainedMLP_FD_MD.mat`**: The pre-trained Multilayer Perceptron (MLP) Neural Network model used for predictions.
- **`results_FD_MD.mat`**: Stored outputs and results from the model's predictions.
- **`Dataset/`**: Directory containing the raw sensor data in CSV format (e.g., `U1NW_FD.csv`, `U1NW_MD.csv`).

## 🛠️ Technologies Used
- **MATLAB**: Used for all data processing, feature extraction, and neural network deployment.
- **Machine Learning (MLP)**: A feedforward artificial neural network used to classify the gait data.

## 🚀 How to Run

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/dasi005/AI-and-Machine-Learning-project.git
