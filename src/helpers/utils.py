"""
Contains helper functions for formatting and displaying results.
"""

import numpy as np

def format_metric(result_dict) -> str:
    """
    Format a dictionary of metrics into a human-readable string.
    Handles different numeric types (float, int) with appropriate formatting.
    
    Args:
        result_dict (dict): Dictionary containing metric names and their values
                           Example: {'accuracy': 0.85, 'n_samples': 1000}
    
    Returns:
        str: Formatted string of metrics
             Example: "accuracy:0.8500, n_samples:1000"
    
    Raises:
        TypeError: If input is not a dictionary
    """
    if not isinstance(result_dict, dict):
        raise TypeError("Error, need dictionary.")

    format_str = []

    # Sort metrics alphabetically for consistency
    metrics = sorted(result_dict.keys())

    for metric in metrics:
        value = result_dict[metric]

        # Format floating point numbers to 4 decimal places
        if isinstance(value, (float, np.float32, np.float64)):
            format_str.append(f"{metric}:{value:.4f}")

        # Format integers without decimal places
        elif isinstance(value, (int, np.int32, np.int64)):
            format_str.append(f"{metric}:{value}")
            
        # Handle any other types by converting to string
        else:
            format_str.append(f"{metric}:{str(value)}")

    return ', '.join(format_str)
