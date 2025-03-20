"""
Preprocessing script for the SiTunes dataset to generate psychological satisfaction prediction datasets.
This script processes music listening data to predict users' emotional changes (valence) after music 
listening sessions. This is formulated as a 3-class classification problem where:
- Class 0: Significant mood decrease (valence change < -0.125)
- Class 1: No significant change (-0.125 <= valence change <= 0.125)
- Class 2: Significant mood improvement (valence change > 0.125)
"""

import os
import pandas as pd

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
psychological_output_dir = os.path.join(project_root, "datasets", "Psychological_Datasets")

# Create output directory for psychological datasets
os.makedirs(psychological_output_dir, exist_ok=True)

# Process data for experimental settings 2 and 3
for setting in [2, 3]:
    # Generate 10 different random seeds for more comprehensive evaluation
    for seed in range(101, 111):
        data_dir = f"setting{setting}-{seed}"
        data_output_path = os.path.join(psychological_output_dir, data_dir)
        
        os.makedirs(data_output_path, exist_ok=True)
        
        # Process item metadata
        item_file = f"{data_dir}.item"
        item = pd.read_csv(os.path.join(data_output_path, item_file), delimiter='\t')
        item.to_csv(os.path.join(data_output_path, item_file), index=False, sep='\t') 

        # Process interaction data for each stage (train/valid/test)
        for stage in ['train', 'valid', 'test']:
            inter_name = f"{data_dir}.{stage}.inter"
            inter = pd.read_csv(os.path.join(data_output_path, inter_name), delimiter='\t')
            
            # Calculate emotional valence change after music listening (pos delta = mood improvement)
            inter['valence_delta'] = inter['emo_post_valence'] - inter['emo_pre_valence']
            
            # Create psychological satisfaction labels using +- 0.125 thresholds
            inter["mood_improvement:label"] = pd.cut(
                inter.valence_delta, bins=[-2, -0.125, 0.125, 2], labels=False
            )
            inter.drop(columns='valence_delta').to_csv(os.path.join(data_output_path, inter_name), index=False, sep='\t')