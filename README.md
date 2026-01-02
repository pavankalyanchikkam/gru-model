# PrognosAI - Predictive Maintenance System 🔧

## Project Overview
PrognosAI is an AI-driven predictive maintenance system that estimates the Remaining Useful Life (RUL) of industrial machinery using multivariate time-series sensor data. Built with NASA CMAPSS datasets, this system enables timely maintenance decisions, minimizes unplanned downtime, and optimizes asset utilization through deep learning techniques.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

### 🎯 Core Capabilities
- **🤖 AI-Powered RUL Prediction**: LSTM/GRU/CNN-LSTM models for accurate remaining useful life estimation
- **🔔 Real-time Risk Assessment**: Automated alert system with warning and critical thresholds
- **📊 Interactive Dashboard**: Professional Streamlit interface with real-time visualizations
- **📁 Multi-Dataset Support**: FD001, FD002, FD003, FD004 CMAPSS datasets
- **🔄 Simulation Mode**: Works without trained models for demonstration

### 📊 Dashboard Features
- **📈 Live Metrics**: Critical/Warning/Normal alerts tracking
- **📊 Interactive Charts**: RUL distribution, trend analysis, error visualization
- **💾 Export Capabilities**: CSV reports, model downloads, configuration exports
- **⚡ Performance Monitoring**: RMSE, MAE, R² scores and accuracy metrics
- **⚙️ Customizable Settings**: Threshold configuration, visualization options

## 🏗️ Project Structure
```text
PrognosAI/
├── 📁 app/                           # Main application
│   ├── main.py                      # Streamlit application entry point
│   ├── config.py                    # Configuration settings
│   └── styles.css                   # Minimal CSS for styling
│
├── 📁 src/                          # Source code modules
│   ├── data_processor.py            # CMAPSS data processing
│   ├── model_handler.py             # Model prediction handler
│   ├── visualizations.py            # Plotly chart generation
│   └── file_handler.py              # File operations and validation
│
├── 📁 notebooks/                    # Jupyter notebooks
│   └── prognos.ipynb                # Complete training pipeline
│
├── 📁 models/                       # Trained model storage (auto-created)
├── 📁 test_data/                    # Sample test files (auto-created)
├── 📁 uploads/                      # User uploads (auto-created)
├── 📁 assets/                       # Static assets (auto-created)
│
├── 📁 docs/                         # Documentation
│   └── AI-PrognosAI.pdf             # Project documentation
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 setup.bat                     # Windows setup script
├── 📄 run.bat                       # Windows run script
└── 📄 README.md                     # This file

