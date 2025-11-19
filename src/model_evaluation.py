import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Calculate key regression metrics for financial predictions.
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Convert to numpy if tensors
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy().flatten()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy().flatten()
    else:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
    
    # Correlation (handle zero variance cases)
    with np.errstate(invalid='ignore'):
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            correlation = np.nan
        else:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Directional accuracy
    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    directional_accuracy = (direction_true == direction_pred).mean() * 100
    
    # R² (variance explained)
    r2 = r2_score(y_true, y_pred)
    
    # RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {
        'Directional_Accuracy_%': directional_accuracy,
        'Correlation': correlation,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'N_Predictions': len(y_true)
    }
    
    return metrics

def quick_eval(y_test, y_pred):
    """
    Quick evaluation that prints just the key metrics.
    
    Args:
        y_test: actual values (tensor or array)
        y_pred: predicted values (tensor or array)
    
    Returns:
        dict: metrics dictionary
    """
    metrics = evaluate_model(y_test, y_pred)
    
    print(f"\nDirectional Accuracy: {metrics['Directional_Accuracy_%']:.2f}%")
    print(f"Correlation:          {metrics['Correlation']:.4f}")
    print(f"R² (Variance):        {metrics['R²']:.4f}")
    
    return metrics


class ModelPerformanceTracker:
    def __init__(self, csv_file="model_results.csv"):
        self.output_dir = "/home/lukeditzler/projects/pytorch/examples/Financial_Markets_Model/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, csv_file)
        
        # Try to load existing results from the correct path
        try:
            self.results_df = pd.read_csv(self.csv_path)
            print(f"Loaded existing results from {self.csv_path}")
            print(f"Current number of rows: {len(self.results_df)}\n")
        except FileNotFoundError:
            self.results_df = pd.DataFrame()
            print(f"No existing CSV found. Creating new tracker.\n")
        
    def add_result(self, model_name, num_features, metrics_dict, notes=""):
        """
        Add a model's results to the tracker.
        
        Args:
            model_name: Name/description of model architecture
            num_features: Number of features used
            metrics_dict: Dictionary of metrics from evaluate_model
            notes: Any additional notes about the configuration
        """
        result = {
            'Model_Name': model_name,
            'Num_Features': num_features,
            'Directional_Accuracy_%': metrics_dict['Directional_Accuracy_%'],
            'Correlation': metrics_dict['Correlation'],
            'R²': metrics_dict['R²'],
            'RMSE': metrics_dict['RMSE'],
            'MAE': metrics_dict['MAE'],
            'N_Predictions': metrics_dict['N_Predictions'],
            'Notes': notes
        }
        
        # Append to results
        new_row = pd.DataFrame([result])
        if self.results_df.empty:
            self.results_df = new_row
        else:
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        # Save to CSV (removed the incorrect file_path line)
        self.results_df.to_csv(self.csv_path, index=False)

        
    def print_summary(self):
        """Print a formatted summary table"""
        if self.results_df.empty:
            print("No results yet!")
            return
        
        print("\n" + "="*120)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*120)
        
        # Format for display
        display_df = self.results_df.copy()
        display_df['Directional_Accuracy_%'] = display_df['Directional_Accuracy_%'].apply(lambda x: f"{x:.2f}")
        display_df['Correlation'] = display_df['Correlation'].apply(lambda x: f"{x:.4f}")
        display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.4f}")
        display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        print("="*120 + "\n")