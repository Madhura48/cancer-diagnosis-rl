"""
Test script to verify all components are working
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import time

def test_gpu():
    """Test GPU connectivity"""
    print("=== Testing GPU ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Test tensor creation
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        print(f"GPU tensor multiplication successful: {z.shape}")
    else:
        print("No GPU available, using CPU")

def test_data():
    """Test data loading"""
    print("\n=== Testing Data ===")
    data_file = "data/processed/primary_cancer_data.csv"
    
    if Path(data_file).exists():
        df = pd.read_csv(data_file)
        print(f"Data loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns[:5])}...")
        print(f"Sample data:\n{df.head(2)}")
        return df
    else:
        print(f"Data file not found: {data_file}")
        return None

def test_simple_training():
    """Test simple neural network training"""
    print("\n=== Testing Simple Training ===")
    
    # Create simple network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)
    ).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Test training loop
    print("Running 10 training steps...")
    for step in range(10):
        # Random data
        x = torch.randn(32, 10).to(device)
        y = torch.randint(0, 2, (32,)).to(device)
        
        # Forward pass
        outputs = net(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    print("Simple training completed successfully!")

def test_framework_components():
    """Test framework components individually"""
    print("\n=== Testing Framework Components ===")
    
    # Test data loading for framework
    df = test_data()
    if df is None:
        print("Cannot test framework - no data")
        return False
    
    # Test environment creation
    print("Testing environment creation...")
    try:
        from dayhoff_framework import CancerGenomicsEnvironment
        env = CancerGenomicsEnvironment("data/processed/primary_cancer_data.csv")
        print(f"Environment created successfully: {env.n_samples} samples, {env.n_features} features")
        
        # Test environment reset
        state = env.reset()
        print(f"Environment reset successful, state keys: {list(state.keys())}")
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        return False
    
    # Test agents
    print("Testing agent creation...")
    try:
        from dayhoff_framework import FeatureAnalysisAgent, DiagnosisAgent
        
        feature_agent = FeatureAnalysisAgent(state_dim=env.n_features, action_dim=env.n_features)
        diagnosis_agent = DiagnosisAgent(state_dim=20)
        
        print("Agents created successfully!")
        
        # Test feature selection
        features, confidence = feature_agent.select_features(state['features'])
        print(f"Feature selection test: {len(features)} features selected, confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Agent test failed: {e}")
        return False

def main():
    print("=== Dayhoff Framework Component Test ===")
    start_time = time.time()
    
    # Run all tests
    test_gpu()
    df = test_data()
    test_simple_training()
    framework_ok = test_framework_components()
    
    end_time = time.time()
    print(f"\n=== Test Summary ===")
    print(f"Test completed in {end_time - start_time:.2f} seconds")
    print(f"Framework components OK: {framework_ok}")
    
    if framework_ok:
        print("\n✅ All tests passed! Ready to run full training.")
        print("You can now run: python dayhoff_framework.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()