"""
Ensemble Learning Module for Psychological Music Preference Classification.

This file implements ensemble learning approaches (Stacking and Voting) for classifying
psychological changes in music listening sessions. 

Supports two ensemble methods:
1. Stacking: Uses logistic regression as meta-learner
2. Voting: Uses soft voting (probability-based) combination
"""

import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from helpers.BaseReader import BaseReader

def parse_args():
    """
    Parse command line arguments for ensemble model configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments including:
            - datadir: Path to psychological datasets
            - model_type: Type of ensemble (stacking/voting)
            - dataname: Name of the dataset to use
            - context_column_group: Group of context features to include
    """
    parser = argparse.ArgumentParser(description="Ensemble model for classification")
    parser.add_argument('--datadir', type=str, default='../datasets/Psychological_Datasets/',
                       help="Directory containing the psychological datasets")
    parser.add_argument('--model_type', type=str, default='stacking', choices=['stacking', 'voting'],
                       help="Type of ensemble model to use")
    parser.add_argument('--dataname', type=str, required=True,
                       help="Name of the dataset to analyze")
    parser.add_argument('--context_column_group', type=str, required=True,
                       help="Group of context columns to include in analysis")
    return parser.parse_args()

def load_data(datadir, dataname, context_column_group):
    """
    Load and preprocess data using BaseReader.
    
    Args:
        datadir (str): Directory containing the datasets
        dataname (str): Name of the dataset to load
        context_column_group (str): Group of context columns to include
        
    Returns:
        tuple: Six elements containing:
            - train_X, train_y: Training features and labels
            - val_X, val_y: Validation features and labels
            - test_X, test_y: Test features and labels
    """
    # Create argument namespace for BaseReader
    args = argparse.Namespace(
        datadir=datadir,
        dataname=dataname,
        load_metadata=1,
        context_column_group=context_column_group
    )
    
    # Initialize and use BaseReader to load data
    reader = BaseReader(args)
    train_X, train_y = reader.train_X, reader.train_y
    val_X, val_y = reader.val_X, reader.val_y
    test_X, test_y = reader.test_X, reader.test_y
    
    # Print dataset shapes to verify
    print(f"Train: {train_X.shape}")
    print(f"Validation: {val_X.shape}")
    print(f"Test: {test_X.shape}")
    
    return train_X, train_y, val_X, val_y, test_X, test_y

def calculate_metrics(predictions, y_true):
    """
    Calculate multiple classification metrics.
    
    Args:
        predictions: Model predictions
        y_true: True labels
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    try:
        # Ensure inputs are numpy arrays
        predictions = np.array(predictions)
        y_true = np.array(y_true)
        
        # Verify shapes match
        if predictions.shape != y_true.shape:
            print(f"Shape mismatch: predictions {predictions.shape}, y_true {y_true.shape}")
            return None
            
        # Calculate metrics with error handling
        accuracy = float(accuracy_score(y_true, predictions))
        macro_f1 = float(f1_score(y_true, predictions, average='macro', labels=np.unique(y_true)))
        micro_f1 = float(f1_score(y_true, predictions, average='micro', labels=np.unique(y_true)))
        
        # Validate metric values
        if not (0 <= accuracy <= 1 and 0 <= macro_f1 <= 1 and 0 <= micro_f1 <= 1):
            print("Invalid metric values detected")
            return None
            
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Macro F1   : {macro_f1:.4f}")
        print(f"Micro F1   : {micro_f1:.4f}")
        
        return {"accuracy": accuracy, "macro_f1": macro_f1, "micro_f1": micro_f1}
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def main():
    """
    Main execution function for ensemble learning.
    """
    # 1. Parse command line arguments
    args = parse_args()
    
    # 2. Load and preprocess data
    train_X, train_y, val_X, val_y, test_X, test_y = load_data(args.datadir, args.dataname, args.context_column_group)

    # 3. Initialize base models with optimized hyperparameters
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_split=5,
        random_state=101
    )
    
    gbdt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        random_state=101
    )
    
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            learning_rate='adaptive',
            learning_rate_init=0.01,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=101
        )
    )

    # 4. Create ensemble model based on specified type
    if args.model_type == 'stacking':
        meta_model = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver='lbfgs',
            random_state=101
        )
        ensemble = StackingClassifier(
            estimators=[('rf', rf), ('gbdt', gbdt), ('mlp', mlp)],
            final_estimator=meta_model,
            cv=5
        )
    else: 
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gbdt', gbdt), ('mlp', mlp)],
            voting='soft',
            weights=[2, 1, 1]
        )

    # 5. Train the ensemble model
    print(f"\nTraining ensemble model for {args.dataname} with context {args.context_column_group}...\n")
    
    try:
        ensemble.fit(train_X, train_y)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return

    # 6. Evaluate model on all datasets
    for X, y, dataset in zip([train_X, val_X, test_X], [train_y, val_y, test_y], ["train", "val", "test"]):
        try:
            predictions = ensemble.predict(X)
            metrics = calculate_metrics(predictions, y)
            
            if metrics is None:
                print(f"Warning: Could not calculate valid metrics for {dataset} set")
                continue
                
            print(f"{'-' * 40}")
            print(f"Results for {dataset} Set - {args.dataname} ({args.context_column_group})")
            print(f"Accuracy   : {metrics['accuracy']:.4f}")
            print(f"Macro F1   : {metrics['macro_f1']:.4f}")
            print(f"Micro F1   : {metrics['micro_f1']:.4f}")
            print(f"{'-' * 40}\n")
            
        except Exception as e:
            print(f"Error evaluating {dataset} set: {str(e)}")

if __name__ == "__main__":
    main()
