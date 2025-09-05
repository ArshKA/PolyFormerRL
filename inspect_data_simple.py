#!/usr/bin/env python3
"""
Simple script to inspect the training data structure without heavy dependencies.
Just examines the raw TSV files to understand the data format.
"""

import os
import csv

def inspect_tsv_data():
    """Inspect the TSV training data"""
    
    print("=== RefCoCo Training Data Inspector ===\n")
    
    # Try to find training data files
    possible_paths = [
        "/data0/arshkon/checkpoints/polyform_rl/datasets/finetune/refcoco+g_train_shuffled.tsv",
        "/data0/arshkon/checkpoints/polyform_rl/datasets/finetune/refcoco_train.tsv",
        "/data0/arshkon/checkpoints/polyform_rl/datasets/finetune/refcocog_train.tsv"
    ]
    
    data_file = None
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if not data_file:
        print("No training data files found. Checked:")
        for path in possible_paths:
            print(f"  {path}")
        print("\nPlease check if the data files exist and update the paths.")
        return
    
    print(f"Found training data: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            print(f"Examining first few rows...\n")
            
            for i, row in enumerate(reader):
                if i >= 3:  # Only look at first 3 rows
                    break
                    
                print(f"--- Row {i} ---")
                print(f"Number of columns: {len(row)}")
                
                # Based on selected_cols=0,5,6,2,4,3,7 from training script
                if len(row) >= 8:
                    print(f"Column 0 (ID): {row[0][:50]}...")  # First 50 chars
                    print(f"Column 2 (Text): {row[2][:100]}...")  # First 100 chars
                    print(f"Column 3 (Region): {row[3][:50]}...")
                    print(f"Column 4 (Polygons): {row[4][:100]}...")  # This is the key one
                    print(f"Column 5 (Image): [base64 data - {len(row[5])} chars]")
                    print(f"Column 6 (Mask): [base64 data - {len(row[6])} chars]")
                    if len(row) > 7:
                        print(f"Column 7 (Extra): {row[7][:50]}...")
                    
                    # Analyze polygon data format
                    polygon_data = row[4]
                    print(f"\nPolygon data analysis:")
                    print(f"Length: {len(polygon_data)} characters")
                    if polygon_data:
                        # Count spaces (separators between coordinates)
                        spaces = polygon_data.count(' ')
                        commas = polygon_data.count(',')
                        print(f"Spaces: {spaces}, Commas: {commas}")
                        
                        # Show first part of polygon data
                        first_100 = polygon_data[:100]
                        print(f"First 100 chars: {first_100}")
                        
                        # Try to estimate number of points
                        if commas > 0:
                            # Format might be: x1,y1 x2,y2 x3,y3 ...
                            estimated_points = commas // 2 + 1 if commas % 2 == 1 else commas // 2
                            print(f"Estimated coordinate points: ~{estimated_points}")
                
                print()
                
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print("=== Key Insights ===")
    print("1. The polygon data in column 4 contains the coordinate sequences")
    print("2. During training, these get converted to token sequences with:")
    print("   - Coordinate tokens (type 0)")
    print("   - Separator tokens (type 1)")  
    print("   - EOS tokens (type 2)")
    print("3. The model learns to predict both token types AND coordinate values")
    print("4. If sequences are long, the model should learn to output EOS when done")

def check_log_for_sequence_lengths():
    """Check training logs for clues about sequence lengths"""
    print("\n=== Checking Training Logs ===")
    
    log_file = "/data0/arshkon/checkpoints/polyform_rl/polyformer_l_logs/100_5e-5_512.log"
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Analyzing log file: {log_file}")
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        print(f"Total log lines: {len(lines)}")
        
        # Look for lines mentioning sequence lengths, tokens, etc.
        relevant_lines = []
        for line in lines[:50]:  # First 50 lines
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['token', 'length', 'ntokens', 'sequence']):
                relevant_lines.append(line)
        
        if relevant_lines:
            print(f"\nRelevant lines from log (first 50 lines):")
            for line in relevant_lines[:5]:  # Show first 5 relevant lines
                print(f"  {line}")
        else:
            print("No obvious sequence length info in first 50 lines")
            
        # Look at a few recent lines
        print(f"\nLast few lines of log:")
        for line in lines[-3:]:
            print(f"  {line.strip()}")
            
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    inspect_tsv_data()
    check_log_for_sequence_lengths()
    
    print("\n=== Summary ===")
    print("The training data contains polygon coordinate sequences.")
    print("The model learns to predict when to stop (EOS token type 2).")
    print("If your model hits max_len=210 every time, either:")
    print("1. The min_len constraint is too high (set min_len=1 or 2)")
    print("2. The model wasn't trained long enough to learn proper EOS prediction")
    print("3. There's a bug in the inference loop EOS detection")
