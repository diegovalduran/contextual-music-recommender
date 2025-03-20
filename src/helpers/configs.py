"""
Configuration constants for SiTunes dataset feature processing.
Defines column names and feature groups used in the psychological satisfaction prediction task.
"""

# Core identifier columns
UID = 'user_id' # User identifier column
IID = 'item_id' # Item (song) identifier column
LABEL = 'mood_improvement:label' # Target variable for mood change prediction

# Subjective context features: User's emotional state before listening
CONTEXT_sub = [
    'emo_pre_valence', # Initial emotional valence (positive/negative feeling)
    'emo_pre_arousal'  # Initial emotional arousal (energy level)
]

# Objective context features: Environmental and physiological measurements
CONTEXT_obj = [
    # Temporal features
    'time_1', 'time_2', 'time_3',  
    
    # Physiological features
    'relative_HB_mean',         # Mean relative heart beat
    'activity_intensity_mean',  # Mean activity intensity
    'activity_step_mean',       # Mean step count
    'relative_HB_std',          # STD of relative heart beat
    'activity_intensity_std',   # STD of activity intensity
    'activity_step_std',        # STD of step count
    
    # Activity type one-hot encodings
    'activity_type_0.0',        
    'activity_type_1.0',
    'activity_type_2.0',
    'activity_type_3.0', 
    'activity_type_4.0',
    
    # Weather conditions
    'weather1_0', 'weather1_1', 'weather1_2',  
    'weather2', # Temperature
    'weather3', # Humidity
    'weather4', # Air pressure
    
    # Location features
    'GPS1', 'GPS2', 'GPS3', # GPS coordinates/location context
    'timestamp'             # Timestamp of the interaction
]

# Combined feature set: both subjective and objective contexts
CONTEXT_all = CONTEXT_sub + CONTEXT_obj

