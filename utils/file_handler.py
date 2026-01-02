"""
File Handler for PrognosAI
Handles file operations, validation, and exports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import io
import json
from datetime import datetime

class FileHandler:
    """Handle all file operations for the application"""
    
    @staticmethod
    def validate_cmapss_file(file_path, expected_columns=26):
        """
        Validate CMAPSS dataset file format
        
        Args:
            file_path: Path to the file
            expected_columns: Expected number of columns (26 for CMAPSS)
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Try reading first few lines
            df = pd.read_csv(file_path, sep=r'\s+', header=None, nrows=10)
            
            # Check column count
            if df.shape[1] != expected_columns:
                return False, f"Expected {expected_columns} columns, got {df.shape[1]}"
            
            # Check for numeric values
            if not np.issubdtype(df.values.dtype, np.number):
                # Try to find which column has non-numeric values
                non_numeric_cols = []
                for col in df.columns:
                    try:
                        pd.to_numeric(df[col])
                    except:
                        non_numeric_cols.append(col)
                
                if non_numeric_cols:
                    return False, f"Non-numeric values in columns: {non_numeric_cols[:3]}"
            
            # Check for reasonable value ranges
            # Sensor values typically in reasonable ranges
            if (df.iloc[:, 2:].abs() > 10000).any().any():
                return False, "Values outside expected range detected"
            
            return True, "File format is valid"
            
        except pd.errors.ParserError as e:
            return False, f"Parsing error: {str(e)}"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    @staticmethod
    def generate_csv_report(predictions, metrics=None, dataset_name=None):
        """
        Generate comprehensive CSV report
        
        Args:
            predictions: List of prediction dictionaries
            metrics: Dictionary of metrics
            dataset_name: Name of the dataset
        
        Returns:
            CSV string
        """
        if not predictions:
            return ""
        
        # Create main predictions dataframe
        pred_df = pd.DataFrame(predictions)
        
        # Add metadata columns
        pred_df['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if dataset_name:
            pred_df['dataset'] = dataset_name
        
        # Add priority score based on RUL
        def calculate_priority(rul, alert):
            if alert == 'CRITICAL':
                return 1
            elif alert == 'WARNING':
                return 2
            else:
                return 3
        
        pred_df['priority'] = pred_df.apply(
            lambda row: calculate_priority(row['predicted_rul'], row['alert']), 
            axis=1
        )
        
        # Add maintenance recommendation
        def get_maintenance_recommendation(rul, alert):
            if alert == 'CRITICAL':
                return "IMMEDIATE maintenance required"
            elif alert == 'WARNING':
                return "Schedule maintenance within 2 weeks"
            else:
                return "Routine check recommended"
        
        pred_df['recommendation'] = pred_df.apply(
            lambda row: get_maintenance_recommendation(row['predicted_rul'], row['alert']), 
            axis=1
        )
        
        # Create summary section
        summary_data = []
        if metrics:
            summary_data.append(["METRICS SUMMARY", ""])
            summary_data.append(["Total Units Analyzed", metrics.get('total_units', 0)])
            summary_data.append(["Critical Alerts", metrics.get('critical_count', 0)])
            summary_data.append(["Warning Alerts", metrics.get('warning_count', 0)])
            summary_data.append(["Normal Status", metrics.get('normal_count', 0)])
            
            if 'rmse' in metrics:
                summary_data.append(["RMSE", f"{metrics.get('rmse', 0):.2f}"])
                summary_data.append(["MAE", f"{metrics.get('mae', 0):.2f}"])
                summary_data.append(["Max Error", f"{metrics.get('max_error', 0):.2f}"])
            
            summary_data.append(["Mean RUL", f"{metrics.get('mean_rul', 0):.2f}"])
            summary_data.append(["Analysis Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        # Combine data
        csv_output = pred_df.to_csv(index=False)
        
        # Add summary section
        if summary_data:
            csv_output = "\n".join([",".join(map(str, row)) for row in summary_data]) + "\n\n" + csv_output
        
        return csv_output
    
    @staticmethod
    def generate_json_report(predictions, metrics=None, dataset_name=None):
        """
        Generate JSON report
        
        Args:
            predictions: List of prediction dictionaries
            metrics: Dictionary of metrics
            dataset_name: Name of the dataset
        
        Returns:
            JSON string
        """
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset': dataset_name,
                'total_predictions': len(predictions)
            },
            'predictions': predictions
        }
        
        if metrics:
            report['metrics'] = metrics
        
        return json.dumps(report, indent=2)
    
    @staticmethod
    def save_uploaded_file(file, upload_dir="uploads"):
        """
        Save uploaded file with proper naming
        
        Args:
            file: Streamlit uploaded file object
            upload_dir: Directory to save file
        
        Returns:
            Path to saved file
        """
        upload_path = Path(upload_dir)
        upload_path.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.name}"
        file_path = upload_path / filename
        
        # Save file
        file_path.write_bytes(file.getbuffer())
        
        return file_path
    
    @staticmethod
    def create_sample_dataset(n_units=10, n_cycles=50):
        """
        Create a sample CMAPSS-format dataset for testing
        
        Args:
            n_units: Number of engine units
            n_cycles: Cycles per unit
        
        Returns:
            DataFrame with sample data
        """
        data = []
        
        for unit in range(1, n_units + 1):
            for cycle in range(1, n_cycles + 1):
                row = [unit, cycle]
                
                # Operation settings (3 columns)
                row.extend([
                    np.random.uniform(-0.01, 0.01),
                    np.random.uniform(-0.01, 0.01),
                    np.random.uniform(-100, 100)
                ])
                
                # Sensors (21 columns) - with some degradation over time
                degradation = (cycle / n_cycles) * np.random.uniform(0.5, 2.0)
                
                for sensor in range(1, 22):
                    if sensor in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]:
                        # Degrading sensors
                        base_value = np.random.uniform(500, 2500)
                        noise = np.random.normal(0, 10)
                        sensor_value = base_value * (1 - degradation * 0.01) + noise
                    else:
                        # Stable sensors
                        sensor_value = np.random.uniform(10, 50)
                    
                    row.append(round(sensor_value, 4))
                
                data.append(row)
        
        # Create column names
        columns = ['unit', 'cycle']
        columns += [f'op_setting_{i}' for i in range(1, 4)]
        columns += [f'sensor_{i}' for i in range(1, 22)]
        
        return pd.DataFrame(data, columns=columns)