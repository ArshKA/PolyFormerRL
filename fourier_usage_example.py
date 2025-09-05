#!/usr/bin/env python3
"""
Example of how to use Fourier features with the polygon transformer
"""

def show_usage_examples():
    print("=== How to Use Fourier Features ===\n")
    
    print("1. TRAINING WITH FOURIER FEATURES:")
    print("   Add these flags to your training command:")
    print("   --use-fourier-features")
    print("   --fourier-num-frequencies 10  # Optional, default is 10")
    print()
    
    print("   Example training command:")
    print("   python train.py \\")
    print("       --task refcoco \\")
    print("       --arch polyformer_l \\")
    print("       --use-fourier-features \\")
    print("       --fourier-num-frequencies 15 \\")
    print("       [... other training args ...]")
    print()
    
    print("2. INFERENCE WITH FOURIER FEATURES:")
    print("   In demo.py, set the override:")
    print('   overrides={"use_fourier_features": True, "fourier_num_frequencies": 10}')
    print()
    
    print("3. BACKWARD COMPATIBILITY:")
    print("   - Default behavior unchanged (use_fourier_features=False)")
    print("   - Existing models continue to work without changes")
    print("   - Only new models trained with --use-fourier-features will use the new encoding")
    print()
    
    print("4. COMPARISON:")
    print("   BILINEAR (Original):")
    print("   - 4 token sequences + delta interpolation")
    print("   - Complex but precise")
    print("   - Good spatial understanding")
    print()
    print("   FOURIER (New):")
    print("   - Direct coordinate encoding with sin/cos")
    print("   - Simpler architecture")
    print("   - Popular in modern coordinate-based networks")
    print("   - Multi-scale spatial representation")

def show_technical_details():
    print("\n=== Technical Details ===\n")
    
    print("FOURIER ENCODING:")
    print("- Input: (x, y) coordinates in [0, 1]")
    print("- For each frequency i in [0, num_frequencies):")
    print("  - freq = 2^i")
    print("  - Encode: [sin(freq*π*x), cos(freq*π*x), sin(freq*π*y), cos(freq*π*y)]")
    print("- Output dimension: 4 * num_frequencies")
    print("- Linear projection to embedding dimension")
    print()
    
    print("BENEFITS:")
    print("- Multi-scale representation (low + high frequencies)")
    print("- Infinite precision (no quantization)")
    print("- Proven effective in NeRF and other coordinate networks")
    print("- Simpler than bilinear interpolation")
    print()
    
    print("WHEN TO USE:")
    print("- New models: Try Fourier features first")
    print("- Existing models: Keep bilinear for compatibility")
    print("- Research: Compare both approaches")

if __name__ == "__main__":
    show_usage_examples()
    show_technical_details()
    
    print("\n=== Quick Start ===")
    print("To test Fourier features:")
    print("1. Modify demo.py: set 'use_fourier_features': True in overrides")
    print("2. Run inference to see if it works")
    print("3. For training, add --use-fourier-features to your training script")
    print()
    print("The implementation is ready to use! 🎉")
