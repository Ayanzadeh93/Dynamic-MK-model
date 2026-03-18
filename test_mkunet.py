import torch
import traceback

try:
    from model_mkunet import create_mkunet_original, create_mkunet_dynamic
    x = torch.randn(2, 3, 224, 224)
    
    print("Testing Original...")
    m1 = create_mkunet_original()
    o1 = m1(x)
    print(f"Original output shape: {o1.shape}")
    
    print("\nTesting Dynamic...")
    m2 = create_mkunet_dynamic()
    o2 = m2(x)
    print(f"Dynamic output shape: {o2.shape}")
    
    print("\nSUCCESS: Both models work!")
except Exception as e:
    traceback.print_exc()
