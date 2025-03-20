# Emotion-Aware Music Recommendation System

A machine learning system that predicts emotional changes to music listening sessions and provides personalized recommendations based on lyrical sentiment analysis and user emotional states.

## Project Overview

This project addresses limitations in traditional music recommendation systems by incorporating:

1. **Emotional Context Awareness**: Predicts how songs will affect users' emotional states using situational and psychological data
2. **Lyrical Sentiment Analysis**: Extracts semantic features from lyrics using a fine-tuned LLaMA model
3. **User Preference Clustering**: Groups users based on emotional responses and lyrical preferences

The system achieves **87.8% accuracy** in predicting emotional shifts post-music listening and delivers personalized recommendations through ensemble-based mood prediction.

## Key Technical Components

### 1. Mood Prediction Models

Various classification algorithms predict emotional changes after music listening:

- **Gradient Boosted Decision Trees (GBDT)**: Best performer with 87.8% accuracy
- **Random Forest (RF)**: Strong performance with interpretable results
- **Multi-Layer Perceptron (MLP)**: Neural network approach for complex patterns
- **Ensemble Model**: Stacking classifier combining multiple models

### 2. Lyrical Sentiment Analysis

A fine-tuned LLaMA 3.2-3B model extracts semantic features from song lyrics:

- **Narrative Complexity**: Measures structural sophistication of lyrics
- **Emotional Sophistication**: Captures depth and subtlety of emotional content
- **Thematic Elements**: Identifies love, life, social themes, etc.
- **Temporal Focus**: Measures emphasis on past, present, or future events

### 3. User Preference Learning & Clustering

Hierarchical clustering groups users based on:

- **Rating-Based Features**: User interaction patterns with songs
- **Emotional Preferences**: Valence and arousal responses to music
- **Lyrical Affinities**: Preferences for narrative styles and themes

## Project Structure
```
.
├── src/
│   ├── models/
│   │   ├── LR.py         # Logistic Regression implementation
│   │   ├── MLP.py        # Multi-Layer Perceptron implementation
│   │   ├── RF.py         # Random Forest implementation
│   │   └── GBDT.py       # Gradient Boosting implementation
│   ├── recs/
│   │   └── learn_preference.py  # User preference learning system
│   ├── main.py           # Main training pipeline
│   └── ensemble.py       # Ensemble model implementation
├── script/
│   ├── LR.sh            # Logistic Regression experiments
│   ├── MLP.sh           # MLP experiments
│   ├── RF.sh            # Random Forest experiments
│   ├── GBDT.sh          # Gradient Boosting experiments
│   └── ensemble.sh      # Ensemble experiments
├── datasets/
│   └── Psychological_Datasets/  # Dataset directory
├── logs/                # Experiment logs
├── models/              # Saved model checkpoints
└── requirements.txt     # Project dependencies
```

## Dataset

This project utilizes the **SiTunes dataset** [1], which includes:

- **Situational Data (Obj.)**: Environmental factors (weather, location) and physiological features (heart rate, activity)
- **Emotional State Data (Sub.)**: User-reported emotional annotations (valence, arousal)
- **Song Metadata**: Basic information about tracks
- **Lyrics**: Web-scraped lyrics for sentiment analysis

The SiTunes dataset is available at: https://github.com/JiayuLi-997/SiTunes_dataset/

## Results

### Mood Prediction Performance

| Model | Feature Set | Accuracy | Macro F1 | Micro F1 |
|-------|-------------|----------|----------|----------|
| GBDT  | Obj. + Sub. | 87.8%    | 0.8646   | 0.8777   |
| Ensemble | Obj. + Sub. | 75.7% | 0.5855   | 0.7565   |
| RF    | Obj. + Sub. | 65.1%    | 0.5216   | 0.6511   |
| LR    | Obj. + Sub. | 59.2%    | 0.4593   | 0.5919   |

### User Clustering

Four distinct user clusters were identified based on emotional responses and lyrical preferences:

- **Cluster 0**: High emotional variability, prefers complex narratives
- **Cluster 1**: Balanced feature profile, favors socially themed songs
- **Cluster 2**: Favors highly rated tracks with moderate emotional content
- **Cluster 3**: Greater valence range, higher emotional sophistication

### Recommendation Quality

Recommendations achieved similarity scores ≥0.95, indicating strong alignment with user preferences.

## Transferability to Video Content

While developed for music, the core techniques can be applied to video content:

- **Emotional Response Prediction**: Predict viewer emotional responses to videos
- **Content Analysis**: Extract thematic elements from video metadata/transcripts
- **User Preference Clustering**: Group viewers based on emotional preferences
- **Contextual Recommendations**: Deliver recommendations based on viewing situation

## Setup Instructions

1. **Environment Setup**:
   ```bash
   # Create and activate a virtual environment
   python -m venv env
   source env/bin/activate 
   
   # Install dependencies
   pip install -r requirements.txt

2. **Data Preparation**:
   - Place your psychological music response datasets in the `datasets/Psychological_Datasets/` directory
   - Ensure the data follows the expected format (see Data Format section below)

3. **Configuration**:
   - Each model's hyperparameters can be configured in their respective shell scripts
   - Logging settings can be adjusted in `main.py`

## Running Experiments

### Individual Models

1. **Logistic Regression**:
   ```bash
   chmod +x script/LR.sh
   ./script/LR.sh
   ```
   - Settings:
     - Setting2: l1 regularization, C=10
     - Setting3: l1 regularization, C=10

2. **Multi-Layer Perceptron**:
   ```bash
   chmod +x script/MLP.sh
   ./script/MLP.sh
   ```
   - Settings:
     - Setting2: lr=5e-3, batch_size=256
     - Setting3: lr=1e-3, batch_size=512

3. **Random Forest**:
   ```bash
   chmod +x script/RF.sh
   ./script/RF.sh
   ```
   - Settings:
     - Setting2: max_depth=3, n_estimators=300
     - Setting3: max_depth=5, n_estimators=200

4. **Gradient Boosting**:
   ```bash
   chmod +x script/GBDT.sh
   ./script/GBDT.sh
   ```
   - Settings:
     - Setting2: lr=0.1, n_estimators=200
     - Setting3: lr=0.05, n_estimators=100

### Ensemble Model

```bash
chmod +x script/ensemble.sh
./script/ensemble.sh
```
- Uses stacking classifier combining all individual models

## Data Format

The dataset should be organized as follows:
- Features should include musical content descriptors
- Target variables should represent psychological responses
- Context groups: CONTEXT_all, CONTEXT_sub, CONTEXT_obj

## Experiment Settings

Each experiment runs with:
- 10 different random seeds (101-110)
- 3 context groups (all, sub, obj)
- 2 settings (Setting2, Setting3)
- Total of 60 experiments per model

### Metrics

The following metrics are computed for each experiment:
- Accuracy
- Macro F1 Score
- Micro F1 Score

## Output and Logging

- Results are saved in the `logs/` directory
- Model checkpoints are saved in the `models/` directory
- Each experiment displays progress and results in real-time
- Final averages are computed across all runs

## Troubleshooting

Common issues and solutions:

1. **Permission Denied for Shell Scripts**:
   ```bash
   chmod +x script/*.sh
   ```

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Invalid Metrics**:
   - Check if the dataset is properly formatted
   - Ensure all required columns are present
   - Verify the context group specifications

## License

MIT License

## Author Contributions

This project was developed solely by Diego Valdez Duran as part of CS229 - Machine Learning at Stanford University. All code components were written by the author except for:

- **Libraries**: Standard ML libraries (scikit-learn, PyTorch) were used
- **LLaMA Model**: The base model is from Meta AI, with custom fine-tuning implementation
- **Dataset**: SiTunes dataset [1]

## Citation

This work uses the SiTunes dataset [1]:

```bibtex
@inproceedings{10.1145/3627508.3638343,
    author = {Grigorev, Vadim and Li, Jiayu and Ma, Weizhi and He, Zhiyu and Zhang, Min and Liu, Yiqun and Yan, Ming and Zhang, Ji},
    title = {SiTunes: A Situational Music Recommendation Dataset with Physiological and Psychological Signals},
    year = {2024},
    isbn = {9798400704345},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3627508.3638343},
    doi = {10.1145/3627508.3638343},
    booktitle = {Proceedings of the 2024 Conference on Human Information Interaction and Retrieval},
    pages = {417–421},
    numpages = {5},
    location = {Sheffield, United Kingdom},
    series = {CHIIR '24}
}
```

## References

[1] Grigorev, V., Li, J., Ma, W., He, Z., Zhang, M., Liu, Y., Yan, M., & Zhang, J. (2024). SiTunes: A Situational Music Recommendation Dataset with Physiological and Psychological Signals. In Proceedings of the 2024 Conference on Human Information Interaction and Retrieval (CHIIR '24). ACM, New York, NY, USA.

## Contact

diegoval@stanford.edu 