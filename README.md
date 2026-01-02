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

### 🏗️ Architecture Diagrams
## 1. Interface Architecture
<img width="3462" height="7839" alt="interface acrhi" src="https://github.com/user-attachments/assets/8da1bbdf-ce9c-4190-9485-40529461035e" />

## 2. Colab Training Architecture
<img width="4847" height="13698" alt="collab arch" src="https://github.com/user-attachments/assets/cfed6037-d369-4704-ac57-06caf479135c" />

### 3. 🏗️ Project Structure
<img width="1131" height="684" alt="image" src="https://github.com/user-attachments/assets/2cbad8b3-dea4-49e3-8843-bc804f847b78" />


## 📊 Dataset Information

The system supports NASA's CMAPSS datasets:

| Dataset | Description                                          | Complexity | Training Engines | Test Engines |
|---------|------------------------------------------------------|------------|------------------|--------------|
| FD001   | Single Operating Condition, Single Fault Mode        | Low        | 100              | 100          |
| FD002   | Multiple Operating Conditions, Single Fault Mode     | Medium     | 260              | 259          |
| FD003   | Single Operating Condition, Multiple Fault Modes     | Medium     | 100              | 100          |
| FD004   | Multiple Operating Conditions, Multiple Fault Modes  | High       | 249              | 248          |

## 📖 Usage Guide

### 1. Quick Start with Sample Data
- Navigate to **"Try Sample Data"** tab  
- Click **"Generate Sample Data"**  
- Select **FD001** dataset  
- Upload generated files  
- Set alert thresholds  
- Click **"Run Predictive Analysis"**

### 2. Using Your Own Data
Prepare CMAPSS-format files:
- `test_FDXXX.txt` → Test sensor data  
- `RUL_FDXXX.txt` → True RUL values (optional)  

In the application:
- Select dataset type (**FD001–FD004**)  
- Upload your files  
- Configure warning/critical thresholds  
- Run analysis  

### 3. Dashboard Navigation
- 🏠 **Home**: Upload data and configure settings  
- 📊 **Dashboard**: View predictions, alerts, and metrics  
- 📈 **Analysis**: Detailed statistical analysis  
- ⚙️ **Settings**: System configuration and optimization

## 🤖 Model Training  
Using Jupyter Notebook  


    # Open the training notebook
    jupyter notebook notebooks/prognos.ipynb

    # Follow the step-by-step training pipeline
    # 1. Install dependencies
    # 2. Upload CMAPSS datasets
    # 3. Configure model parameters
    # 4. Train LSTM/GRU/CNN-LSTM models
    # 5. Evaluate performance
    # 6. Export trained models


## 🛠️ Training Pipeline Features


- Multiple Architectures: LSTM, GRU, CNN-LSTM with attention mechanisms

- Hyperparameter Tuning: Built-in with Keras Tuner

- Comprehensive Evaluation: RMSE, MAE, R² scores, error analysis

- Production-Ready Export: Models, scalers, configurations in one package

- Visualization Suite: Training curves, prediction plots, error distributions

## 🔧 Configuration  
### Key Settings in config.py 

    # Alert thresholds (customizable)
    DEFAULT_WARNING_THRESHOLD = 50  # cycles
    DEFAULT_CRITICAL_THRESHOLD = 20  # cycles

    # Performance optimization (for i3/8GB)
    MAX_MEMORY_USAGE_PERCENT = 60
    CACHE_SIZE_LIMIT_MB = 500
    SIMULATION_MODE = True  # Fallback when models unavailable

    # Visualization
    COLOR_SCHEME = {
      'critical': '#dc2626',
      'warning': '#f59e0b',
      'normal': '#10b981'
    }

### 📊 Performance Metrics
The system provides comprehensive evaluation:

| Metric          | Description                     | Target Value   |
|-----------------|---------------------------------|----------------|
| RMSE            | Root Mean Square Error          | < 30 cycles    |
| MAE             | Mean Absolute Error             | < 25 cycles    |
| R² Score        | Coefficient of Determination    | > 0.85         |
| Alert Accuracy  | Correct alert classification    | > 90%          |
| Inference Time  | Prediction time per unit        | < 100ms        |

### 📈 Results & Output
### Generated Artifacts
- CSV Reports: Complete prediction data with metadata
- Visualizations: Interactive Plotly charts (HTML/PNG)
- Model Files: Trained .keras models with configurations
- Scalers: Feature normalization parameters
- Logs: Training history and performance metrics

## Sample Output

    📊 Dataset: FD001
    ✅ Units Analyzed: 100
    🔴 Critical Alerts: 15
    🟡 Warning Alerts: 25
    🟢 Normal Status: 60
    📈 RMSE: 18.42 cycles
    🎯 Accuracy (±20 cycles): 87.5%

### 🛡️ Quality Assurance
### Testing Strategy
- Unit Tests: Individual component validation
- Integration Tests: End-to-end workflow testing
- Performance Tests: Memory usage and response time
- Data Validation: CMAPSS format compliance checking

### Code Quality
- Type Hints: Python type annotations throughout
- Docstrings: Comprehensive function documentation
- Error Handling: Graceful degradation and user feedback
- Logging: Structured logging for debugging


### 🙏 Acknowledgments
- NASA: For the CMAPSS datasets
- TensorFlow/Keras: Deep learning frameworks
- Streamlit: Interactive web application framework
- Plotly: Interactive visualization library
- Infosys Mentor: For technical guidance and project supervision

## PrognosAI - Predicting Failures Before They Happen 🔧🤖

## Built for the Infosys Virtual Internship Program
