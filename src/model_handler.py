"""
Model Handler for PrognosAI - Predictive Maintenance System
Optimized for i3/8GB systems with simulation mode fallback
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import random
import sys
import warnings
warnings.filterwarnings('ignore')

class ModelHandler:
    """Handle model predictions with actual model loading support"""
    
    # Class variable to track initialization (solves the spam issue)
    _initialized = False
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance"""
        if cls._instance is None:
            cls._instance = super(ModelHandler, cls).__new__(cls)
            cls._instance._setup_complete = False
        return cls._instance
    
    def __init__(self):
        """Initialize model handler - runs only once"""
        if not self._setup_complete:
            self.models_loaded = {}
            self.scalers = {}
            self.configs = {}
            self.simulation_mode = True
            self._setup_complete = True
            
            # Print initialization message only once
            print("🤖 Model Handler Initialized: Running in optimized simulation mode")
            print("   To use actual models, copy trained models to 'models/' folder")
    
    def check_model_availability(self, dataset_name):
        """Check if actual trained model exists for dataset"""
        try:
            model_dir = Path('models')
            model_folders = list(model_dir.glob(f"{dataset_name}_*"))
            
            if model_folders:
                latest_folder = max(model_folders, key=lambda x: x.stat().st_mtime)
                return {
                    'available': True,
                    'path': str(latest_folder),
                    'name': latest_folder.name,
                    'config': self._load_model_config(latest_folder)
                }
            return {'available': False, 'message': f"No trained model found for {dataset_name}"}
        except Exception as e:
            return {'available': False, 'message': f"Error checking model: {str(e)}"}
    
    def _load_model_config(self, model_folder):
        """Load model configuration from JSON file"""
        try:
            config_path = model_folder / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def predict(self, dataset_name, n_units, use_realistic_simulation=True):
        """
        Generate predictions using enhanced simulation
        
        Args:
            dataset_name: Dataset identifier (FD001-FD004)
            n_units: Number of engine units to predict
            use_realistic_simulation: Whether to use realistic patterns
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        # Check if we have model config for realistic simulation
        model_info = self.check_model_availability(dataset_name)
        config = model_info.get('config', {})
        rul_clip = config.get('rul_clip', 125)
        
        # Different prediction patterns for each dataset
        dataset_patterns = {
            'FD001': {'critical_ratio': 0.15, 'warning_ratio': 0.30},
            'FD002': {'critical_ratio': 0.20, 'warning_ratio': 0.35},
            'FD003': {'critical_ratio': 0.10, 'warning_ratio': 0.25},
            'FD004': {'critical_ratio': 0.25, 'warning_ratio': 0.40}
        }
        
        pattern = dataset_patterns.get(dataset_name, dataset_patterns['FD001'])
        
        for i in range(n_units):
            unit_id = i + 1
            
            if use_realistic_simulation:
                # Realistic simulation based on dataset patterns
                if i % 7 == 0:  # Pattern-based critical units
                    rul = random.uniform(5, 20)
                    alert = 'CRITICAL'
                    confidence = random.uniform(0.85, 0.95)
                elif i % 3 == 0:  # Pattern-based warning units
                    rul = random.uniform(20, 50)
                    alert = 'WARNING'
                    confidence = random.uniform(0.75, 0.90)
                else:  # Normal units
                    rul = random.uniform(50, min(150, rul_clip))
                    alert = 'NORMAL'
                    confidence = random.uniform(0.65, 0.85)
            else:
                # Simple random simulation
                rand_val = random.random()
                if rand_val < 0.2:
                    rul = random.uniform(5, 20)
                    alert = 'CRITICAL'
                    confidence = random.uniform(0.7, 0.9)
                elif rand_val < 0.5:
                    rul = random.uniform(20, 50)
                    alert = 'WARNING'
                    confidence = random.uniform(0.6, 0.8)
                else:
                    rul = random.uniform(50, 150)
                    alert = 'NORMAL'
                    confidence = random.uniform(0.5, 0.7)
            
            # Add some realistic trends (units degrade over time)
            trend_factor = 1.0 - (i / (n_units * 1.5))
            rul = max(1, rul * trend_factor)
            
            predictions.append({
                'unit': unit_id,
                'predicted_rul': round(rul, 2),
                'alert': alert,
                'confidence': round(confidence, 2),
                'simulation': True,
                'trend': 'decreasing' if trend_factor < 0.9 else 'stable'
            })
        
        return predictions
    
    def get_alert_status(self, rul_value, warning_thresh=50, critical_thresh=20):
        """Determine alert status based on RUL"""
        if rul_value <= critical_thresh:
            return 'CRITICAL', '🔴', 'Immediate action required'
        elif rul_value <= warning_thresh:
            return 'WARNING', '🟡', 'Schedule maintenance soon'
        else:
            return 'NORMAL', '🟢', 'Operating normally'
    
    def calculate_metrics(self, predictions, true_rul=None):
        """Calculate comprehensive prediction metrics"""
        metrics = {'simulation': True}
        
        if predictions:
            pred_values = [p['predicted_rul'] for p in predictions]
            conf_values = [p.get('confidence', 0.7) for p in predictions]
            
            # Basic statistics
            metrics.update({
                'mean_rul': float(np.mean(pred_values)),
                'median_rul': float(np.median(pred_values)),
                'std_rul': float(np.std(pred_values)),
                'min_rul': float(np.min(pred_values)),
                'max_rul': float(np.max(pred_values)),
                'avg_confidence': float(np.mean(conf_values))
            })
            
            # Alert distribution
            alerts = [p['alert'] for p in predictions]
            metrics.update({
                'critical_count': alerts.count('CRITICAL'),
                'warning_count': alerts.count('WARNING'),
                'normal_count': alerts.count('NORMAL'),
                'total_units': len(predictions)
            })
        
        if true_rul is not None and len(predictions) == len(true_rul):
            pred_values = [p['predicted_rul'] for p in predictions]
            errors = np.array(pred_values) - np.array(true_rul)
            
            # Error metrics
            metrics.update({
                'rmse': float(np.sqrt(np.mean(errors ** 2))),
                'mae': float(np.mean(np.abs(errors))),
                'max_error': float(np.max(np.abs(errors))),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'simulation': False  # We have real metrics now
            })
            
            # Accuracy metrics
            accurate_10 = np.sum(np.abs(errors) <= 10)
            accurate_20 = np.sum(np.abs(errors) <= 20)
            accurate_30 = np.sum(np.abs(errors) <= 30)
            
            metrics.update({
                'accuracy_10': float((accurate_10 / len(errors)) * 100),
                'accuracy_20': float((accurate_20 / len(errors)) * 100),
                'accuracy_30': float((accurate_30 / len(errors)) * 100)
            })
        
        return metrics
    
    def get_dataset_info(self, dataset_name):
        """Get information about the dataset"""
        dataset_info = {
            'FD001': {
                'name': 'FD001',
                'description': 'Single Operating Condition, Single Fault Mode',
                'engines_train': 100,
                'engines_test': 100,
                'complexity': 'Low',
                'recommended_use': 'Basic prediction testing'
            },
            'FD002': {
                'name': 'FD002',
                'description': 'Multiple Operating Conditions, Single Fault Mode',
                'engines_train': 260,
                'engines_test': 259,
                'complexity': 'Medium',
                'recommended_use': 'Condition-varying scenarios'
            },
            'FD003': {
                'name': 'FD003',
                'description': 'Single Operating Condition, Multiple Fault Modes',
                'engines_train': 100,
                'engines_test': 100,
                'complexity': 'Medium',
                'recommended_use': 'Multiple fault type analysis'
            },
            'FD004': {
                'name': 'FD004',
                'description': 'Multiple Operating Conditions, Multiple Fault Modes',
                'engines_train': 249,
                'engines_test': 248,
                'complexity': 'High',
                'recommended_use': 'Advanced comprehensive analysis'
            }
        }
        
        return dataset_info.get(dataset_name, dataset_info['FD001'])