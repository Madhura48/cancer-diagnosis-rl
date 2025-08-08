"""
Data Inspector for Cancer Genomics Project
Analyzes downloaded real cancer datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def inspect_downloaded_data():
    """Inspect all downloaded real cancer datasets"""
    print("=== Cancer Genomics Data Inspection ===\n")
    
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    print("Raw data files:")
    raw_files = list(raw_dir.glob("*"))
    for file in raw_files:
        print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
    
    print(f"\nProcessed data files:")
    processed_files = list(processed_dir.glob("*"))
    for file in processed_files:
        print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
    
    # Analyze each dataset
    datasets_info = {}
    
    for file in processed_files:
        if file.suffix == '.csv':
            try:
                df = pd.read_csv(file)
                datasets_info[file.stem] = {
                    'shape': df.shape,
                    'columns': list(df.columns[:10]),  # First 10 columns
                    'data_types': df.dtypes.value_counts().to_dict(),
                    'missing_values': df.isnull().sum().sum(),
                    'sample_data': df.head(3)
                }
                
                print(f"\n--- {file.stem} ---")
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns[:5])}...")
                print(f"Missing values: {df.isnull().sum().sum()}")
                print(f"Data types: {dict(df.dtypes.value_counts())}")
                
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
    
    return datasets_info

def select_primary_dataset():
    """Select the best dataset for RL training"""
    processed_dir = Path("data/processed")
    
    # Find the largest real dataset
    best_file = None
    max_samples = 0
    
    for file in processed_dir.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            if df.shape[0] > max_samples:
                max_samples = df.shape[0]
                best_file = file
        except:
            continue
    
    if best_file:
        print(f"\n=== Selected Primary Dataset: {best_file.name} ===")
        df = pd.read_csv(best_file)
        
        print(f"Samples: {df.shape[0]}")
        print(f"Features: {df.shape[1]}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Save as primary training dataset
        df.to_csv(processed_dir / "primary_cancer_data.csv", index=False)
        
        return df
    else:
        print("No suitable dataset found!")
        return None

def prepare_for_rl_training(df):
    """Prepare the dataset specifically for RL training"""
    if df is None:
        return None
    
    print("\n=== Preparing for RL Training ===")
    
    # Identify target variable (diagnosis/outcome)
    target_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['diagnosis', 'class', 'target', 'outcome', 'response']):
            target_cols.append(col)
    
    print(f"Potential target columns: {target_cols}")
    
    if target_cols:
        target_col = target_cols[0]
        print(f"Using '{target_col}' as target variable")
        
        # Encode target if needed
        if df[target_col].dtype == 'object':
            unique_vals = df[target_col].unique()
            print(f"Target classes: {unique_vals}")
            
            # Create mapping
            label_map = {val: i for i, val in enumerate(unique_vals)}
            df[f'{target_col}_encoded'] = df[target_col].map(label_map)
            
        print(f"Dataset ready for RL training:")
        print(f"  - Features: {df.shape[1]-1}")
        print(f"  - Samples: {df.shape[0]}")
        print(f"  - Classes: {df[target_col].nunique() if target_col in df.columns else 'Unknown'}")
        
        return df, target_col
    else:
        print("Warning: No clear target variable found")
        return df, None

if __name__ == "__main__":
    # Inspect all data
    datasets_info = inspect_downloaded_data()
    
    # Select primary dataset
    primary_df = select_primary_dataset()
    
    # Prepare for RL
    if primary_df is not None:
        prepared_data, target = prepare_for_rl_training(primary_df)
        print(f"\n✓ Data preparation complete!")
        print(f"Ready to build Dayhoff RL agents with {prepared_data.shape[0]} real cancer samples")
    else:
        print("\n❌ No suitable dataset found. Check download_real_data.py output.")