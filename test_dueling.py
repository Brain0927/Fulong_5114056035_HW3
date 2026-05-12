#!/usr/bin/env python
"""快速診斷 DuelingQNetwork 問題"""
import sys
print(f"Python version: {sys.version}")

try:
    print("1. Importing TensorFlow...")
    import tensorflow as tf
    from tensorflow.keras import layers
    print(f"   TensorFlow version: {tf.__version__}")
    
    print("\n2. Importing DuelingQNetwork...")
    from dqn.models import DuelingQNetwork
    print("   DuelingQNetwork imported successfully")
    
    print("\n3. Creating DuelingQNetwork(action_size=2, state_size=4)...")
    net = DuelingQNetwork(action_size=2, state_size=4)
    print(f"   Created successfully: {net}")
    print(f"   Action size: {net.action_size}")
    print(f"   State size: {net.state_size}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
