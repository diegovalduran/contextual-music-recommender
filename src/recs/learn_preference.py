import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats

"""
User Preference Learning System for Music Recommendations.

This file implements a user preference learning system that:
1. Analyzes user interactions with songs
2. Extracts lyrical and emotional features
3. Clusters users based on their preferences
4. Generates personalized recommendations

Hierarchical clustering is used to group users with similar tastes
and uses both interaction data and lyrical analysis for recommendations.
"""

class UserPreferenceLearner:
    """
    A class that learns and models user preferences for music recommendations.
    
    The learner combines multiple aspects of music preference:
    - Interaction patterns (ratings, emotional responses)
    - Lyrical content analysis (narrative, emotional sophistication)
    - Temporal patterns in listening behavior
    """
    
    def __init__(self, max_users=30, min_clusters=4, max_clusters=6):
        """
        Initialize the preference learner with clustering parameters.
        
        Args:
            max_users (int): Maximum number of users to analyze
            min_clusters (int): Minimum number of preference clusters
            max_clusters (int): Maximum number of preference clusters
        """
        self.max_users = max_users
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.distance_threshold = None
        self.clustering = None
        self.user_preferences = {}
        self.scaler = StandardScaler()
        
    def learn_from_interactions(self, interactions_df, songs_df):
        """
        Learn user preferences from interaction data and song features.

        Args:
            interactions_df (pd.DataFrame): User-song interaction data
            songs_df (pd.DataFrame): Song metadata and features
        """
        self.songs_df = songs_df
        
        # Load lyrical features from JSON
        with open('../../data/lyrical_features.json', 'r') as f:
            self.lyrics_features = json.load(f)
        
        # Map song IDs to their lyrical features
        item_to_lyrics = {}
        for _, song_row in songs_df.iterrows():
            song_key = f"{song_row['music']} - {song_row['singer']}"
            if song_key in self.lyrics_features:
                item_to_lyrics[song_row['i_id_c']] = self.lyrics_features[song_key]['features']
        
        # Merge interactions with song metadata
        merged_df = interactions_df.merge(
            songs_df,
            left_on='item_id',
            right_on='i_id_c',
            how='left'
        )
        
        # Process users and extract features
        unique_users = sorted(merged_df['user_id'].unique())[:self.max_users]
        user_features = []
        user_ids = []
        
        for user_id in unique_users:
            user_data = merged_df[merged_df['user_id'] == user_id]
            features = self.extract_features(user_data, item_to_lyrics)
            user_features.append(list(features.values()))
            user_ids.append(user_id)
        
        # Scale features and perform clustering
        self.user_features = np.array(user_features)
        self.user_features_scaled = self.scaler.fit_transform(self.user_features)
        self.distance_threshold, self.clustering = self.find_optimal_threshold(self.user_features_scaled)
        self.clusters = self.clustering.fit_predict(self.user_features_scaled)
        
        # Build detailed preference profiles for each user
        for idx, user_id in enumerate(unique_users):
            user_data = merged_df[merged_df['user_id'] == user_id]
            lyrical_stats = defaultdict(list)
            favorite_songs = []
            
            # Process highly rated songs
            for _, row in user_data[user_data['rating'] >= 4].iterrows():
                song_info = {
                    'item_id': row['item_id'],
                    'title': row['music'],
                    'artist': row['singer'],
                    'rating': row['rating']
                }
                
                if row['item_id'] in item_to_lyrics:
                    features = item_to_lyrics[row['item_id']]
                    song_info['lyrical_features'] = features
                    
                    # Collect lyrical statistics
                    lyrical_stats['narrative_complexity'].append(features['narrative_complexity'])
                    lyrical_stats['emotional_sophistication'].append(features['emotional_sophistication'])
                    
                    for theme, value in features['thematic_elements'].items():
                        lyrical_stats[f'thematic_elements_{theme}'].append(value)
                    
                    for focus, value in features['temporal_focus'].items():
                        lyrical_stats[f'temporal_focus_{focus}'].append(value)
                
                favorite_songs.append(song_info)
            
            # Calculate preference statistics
            lyrical_preferences = {}
            for feature, values in lyrical_stats.items():
                if values: 
                    lyrical_preferences[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values) if len(values) > 1 else 0
                    }

            # Store user preferences
            self.user_preferences[user_id] = {
                'favorite_songs': favorite_songs,
                'lyrical_preferences': lyrical_preferences,
                'cluster': int(self.clusters[idx])
            }
        
        # Update user features and cluster assignments
        self.user_features = np.array(user_features)
        self.user_ids = np.array(user_ids)
        self.user_features_scaled = self.scaler.fit_transform(self.user_features)
        self.clusters = self.clustering.fit_predict(self.user_features_scaled)
        
        # Ensure all users have cluster assignments
        for idx, user_id in enumerate(self.user_ids):
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}
            self.user_preferences[user_id]['cluster'] = int(self.clusters[idx])

    def get_recommendations(self, user_id, songs_df, n_recommendations=10):
        """
        Generate personalized song recommendations for a user.
        
        Args:
            user_id: Target user ID
            songs_df (pd.DataFrame): Song metadata
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            list: Ranked song recommendations with similarity scores
        """
        if user_id not in self.user_preferences:
            return []
            
        # Get user preferences
        user_prefs = self.user_preferences[user_id]
        preferred_features = {
            'narrative_complexity': user_prefs['lyrical_preferences'].get('narrative_complexity', {}).get('mean', 0),
            'emotional_sophistication': user_prefs['lyrical_preferences'].get('emotional_sophistication', {}).get('mean', 0),
            'theme_love': user_prefs['lyrical_preferences'].get('thematic_elements_love', {}).get('mean', 0),
            'theme_life': user_prefs['lyrical_preferences'].get('thematic_elements_life', {}).get('mean', 0),
            'theme_social': user_prefs['lyrical_preferences'].get('thematic_elements_social', {}).get('mean', 0),
            'focus_past': user_prefs['lyrical_preferences'].get('temporal_focus_past', {}).get('mean', 0),
            'focus_present': user_prefs['lyrical_preferences'].get('temporal_focus_present', {}).get('mean', 0),
            'focus_future': user_prefs['lyrical_preferences'].get('temporal_focus_future', {}).get('mean', 0)
        }
        
        # Get already listened songs
        listened_songs = set(song['item_id'] for song in user_prefs['favorite_songs'])
        
        # Score candidate songs
        song_scores = []
        for _, song_row in songs_df.iterrows():
            if song_row['i_id_c'] in listened_songs:
                continue
                
            song_key = f"{song_row['music']} - {song_row['singer']}"
            if song_key not in self.lyrics_features:
                continue
                
            lyric_features = self.lyrics_features[song_key]['features']
            score = self._calculate_similarity(preferred_features, lyric_features)
            
            song_scores.append({
                'item_id': song_row['i_id_c'],
                'title': song_row['music'],
                'artist': song_row['singer'],
                'score': score,
                'features': lyric_features
            })
        
        # Return top-N recommendations
        recommendations = sorted(song_scores, key=lambda x: x['score'], reverse=True)[:n_recommendations]
        return recommendations
        
    def _calculate_similarity(self, user_prefs, song_features):
        """
        Calculate cosine similarity between user preferences and song features.
        
        Args:
            user_prefs (dict): User's preferred feature values
            song_features (dict): Song's lyrical features
            
        Returns:
            float: Similarity score between 0 (dissimilar) and 1 (identical)
        """
        # Construct user preference vector in 8-dimensional feature space
        user_vector = np.array([
            user_prefs['narrative_complexity'],
            user_prefs['emotional_sophistication'],
            user_prefs['theme_love'],
            user_prefs['theme_life'],
            user_prefs['theme_social'],
            user_prefs['focus_past'],           
            user_prefs['focus_present'],
            user_prefs['focus_future']
        ])
        
        # Construct song feature vector in same dimensional space
        song_vector = np.array([
            song_features['narrative_complexity'],
            song_features['emotional_sophistication'],
            song_features['thematic_elements']['love'],
            song_features['thematic_elements']['life'],
            song_features['thematic_elements']['social'],
            song_features['temporal_focus']['past'],
            song_features['temporal_focus']['present'],
            song_features['temporal_focus']['future']
        ])
        
        # Calculate cosine similarity and extract scalar value from resulting matrix
        return cosine_similarity(user_vector.reshape(1, -1), song_vector.reshape(1, -1))[0][0]
        
    def plot_dendrogram(self, X, save_path='dendrogram.png'):
        """
        Generate and save a dendrogram visualization of user clusters.
        
        Args:
            X (np.array): Scaled user features matrix where each row represents a user
                         and each column represents a normalized feature
            save_path (str): Path where the dendrogram plot will be saved,
                           defaults to 'dendrogram.png'
        """
        plt.figure(figsize=(12, 8))
        
        # Generate dendrogram using Ward's method for hierarchical clustering
        dendrogram = shc.dendrogram(shc.linkage(X, method='ward'))
        
        # Add descriptive labels and title
        plt.title('User Preferences Dendrogram')
        plt.xlabel('User Index') 
        plt.ylabel('Distance')   
        
        plt.savefig(save_path)
        plt.close() 
        
    def get_user_summary(self, user_id):
        """
        Generate a text summary of a user's preferences and recommendations.
        
        Args:
            user_id: Target user ID to generate summary for
            
        Returns:
            str: A formatted multi-line string containing the user's preference summary
                 If no data exists for the user, returns an error message
        """
        # Check if we have data for this user
        if user_id not in self.user_preferences:
            return f"No preference data for user {user_id}"
        
        prefs = self.user_preferences[user_id]
        
        # Initialize summary with user ID and cluster
        summary = f"User {user_id} Preference Summary:\n"
        summary += f"- Cluster: {prefs['cluster']}\n"
        
        # Add favorite songs section if available
        if 'favorite_songs' in prefs:
            summary += f"\n- Favorite songs ({len(prefs['favorite_songs'])}):\n"
            # Show top 5 songs with their lyrical features
            for song in prefs['favorite_songs'][:5]:  
                lyrical_info = ""
                if 'lyrical_features' in song:
                    # Extract narrative and emotional complexity scores
                    narrative = song['lyrical_features']['narrative_complexity']
                    emotional = song['lyrical_features']['emotional_sophistication']
                    lyrical_info = f" [Narrative: {narrative:.2f}, Emotional: {emotional:.2f}]"
                summary += f"  * {song['title']} by {song['artist']} ({song['rating']} stars){lyrical_info}\n"
    
        # Add lyrical preferences section if available
        if 'lyrical_preferences' in prefs and prefs['lyrical_preferences']:
            summary += "\n- Lyrical Preferences:\n"
            lyric_prefs = prefs['lyrical_preferences']
            
            # Add narrative and emotional complexity metrics
            if 'narrative_complexity' in lyric_prefs:
                summary += f"  * Narrative Complexity: {lyric_prefs['narrative_complexity']['mean']:.2f} "
                summary += f"(±{lyric_prefs['narrative_complexity']['std']:.2f})\n"
            if 'emotional_sophistication' in lyric_prefs:
                summary += f"  * Emotional Sophistication: {lyric_prefs['emotional_sophistication']['mean']:.2f} "
                summary += f"(±{lyric_prefs['emotional_sophistication']['std']:.2f})\n"
            
            # Add thematic preferences (love, life, social)
            summary += "  * Thematic preferences:\n"
            for theme in ['thematic_elements_love', 'thematic_elements_life', 'thematic_elements_social']:
                if theme in lyric_prefs:
                    theme_name = theme.split('_')[-1].capitalize()
                    summary += f"    - {theme_name}: {lyric_prefs[theme]['mean']:.2f} "
                    summary += f"(±{lyric_prefs[theme]['std']:.2f})\n"
            
            # Add temporal focus preferences (past, present, future)
            summary += "  * Temporal focus:\n"
            for focus in ['temporal_focus_past', 'temporal_focus_present', 'temporal_focus_future']:
                if focus in lyric_prefs:
                    focus_name = focus.split('_')[-1].capitalize()
                    summary += f"    - {focus_name}: {lyric_prefs[focus]['mean']:.2f} "
                    summary += f"(±{lyric_prefs[focus]['std']:.2f})\n"
                    
        # Try to add personalized recommendations
        try:
            recommendations = self.get_recommendations(user_id, self.songs_df)
            if recommendations:
                summary += "\n- Top Recommendations:\n"
                # Show top 5 recommendations with similarity scores and features
                for i, rec in enumerate(recommendations[:5], 1):
                    summary += f"  {i}. {rec['title']} by {rec['artist']} "
                    summary += f"(Similarity: {rec['score']:.2f})\n"
                    summary += f"     [Narrative: {rec['features']['narrative_complexity']:.2f}, "
                    summary += f"Emotional: {rec['features']['emotional_sophistication']:.2f}]\n"
        except Exception as e:
            summary += f"\nNote: Could not generate recommendations ({str(e)})\n"
        
        return summary

    def extract_features(self, user_data, item_to_lyrics):
        """
        Extract feature vector for a user's listening history.
        
        Args:
            user_data (pd.DataFrame): User's interaction history
            item_to_lyrics (dict): Mapping of songs to lyrical features
            
        Returns:
            dict: Extracted feature vector
        """
        if user_data.empty:
            return {
                'avg_rating': 0,
                'rating_std': 0,
                'rating_range': 0,
                'high_ratings_ratio': 0,
                'low_ratings_ratio': 0,
                'avg_valence': 0,
                'valence_std': 0,
                'avg_arousal': 0,
                'arousal_std': 0,
                'valence_range': 0,
                'arousal_range': 0,
                'emotional_variability': 0,
                'narrative_complexity': 0,
                'emotional_sophistication': 0,
                'theme_love': 0,
                'theme_life': 0,
                'theme_social': 0,
                'focus_past': 0,
                'focus_present': 0,
                'focus_future': 0
            }
        
        # Calculate rating and emotional statistics
        features = {
            'avg_rating': user_data['rating'].mean(),
            'rating_std': user_data['rating'].std(),
            'rating_range': user_data['rating'].max() - user_data['rating'].min(),
            'high_ratings_ratio': len(user_data[user_data['rating'] >= 4]) / len(user_data),
            'low_ratings_ratio': len(user_data[user_data['rating'] <= 2]) / len(user_data),
            'avg_valence': user_data['emo_valence'].mean(),
            'valence_std': user_data['emo_valence'].std(),
            'avg_arousal': user_data['emo_arousal'].mean(),
            'arousal_std': user_data['emo_arousal'].std(),
            'valence_range': user_data['emo_valence'].max() - user_data['emo_valence'].min(),
            'arousal_range': user_data['emo_arousal'].max() - user_data['emo_arousal'].min(),
            'emotional_variability': np.sqrt(user_data['emo_valence'].var() + user_data['emo_arousal'].var()),
            'narrative_complexity': 0.0,
            'emotional_sophistication': 0.0,
            'theme_love': 0.0,
            'theme_life': 0.0,
            'theme_social': 0.0,
            'focus_past': 0.0,
            'focus_present': 0.0,
            'focus_future': 0.0
        }
        
        # Aggregate lyrical features
        lyrical_songs = 0
        for _, row in user_data.iterrows():
            if row['item_id'] in item_to_lyrics:
                lyrical_songs += 1
                lyric_features = item_to_lyrics[row['item_id']]
                features['narrative_complexity'] += lyric_features['narrative_complexity']
                features['emotional_sophistication'] += lyric_features['emotional_sophistication']
                features['theme_love'] += lyric_features['thematic_elements']['love']
                features['theme_life'] += lyric_features['thematic_elements']['life']
                features['theme_social'] += lyric_features['thematic_elements']['social']
                features['focus_past'] += lyric_features['temporal_focus']['past']
                features['focus_present'] += lyric_features['temporal_focus']['present']
                features['focus_future'] += lyric_features['temporal_focus']['future']
        
        # Average lyrical features if songs exist
        if lyrical_songs > 0:
            for key in ['narrative_complexity', 'emotional_sophistication', 
                       'theme_love', 'theme_life', 'theme_social',
                       'focus_past', 'focus_present', 'focus_future']:
                features[key] /= lyrical_songs
        
        return features

    def plot_clusters(self, save_path='cluster_visualization.png'):
        """
        Generate 2D visualization of user clusters using PCA dimensionality reduction.
        
        Args:
            save_path (str): Path where the visualization will be saved,
                           defaults to 'cluster_visualization.png'
        """
        # Reduce high-dimensional feature space to 2D using PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.user_features_scaled)
        
        # Create DataFrame for plotting with seaborn
        plot_df = pd.DataFrame({
            'PC1': features_2d[:, 0],
            'PC2': features_2d[:, 1],
            'Cluster': self.clusters,
            'User_ID': self.user_ids
        })
        
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Create scatter plot
        sns.scatterplot(
            data=plot_df,
            x='PC1',
            y='PC2',
            hue='Cluster',
            style='Cluster',
            s=100,
            palette='deep'
        )
        
        # Add user ID labels to each point
        for idx, row in plot_df.iterrows():
            plt.annotate(
                f'User {int(row["User_ID"])}',
                (row['PC1'], row['PC2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        plt.title('User Clusters in 2D Space')
        plt.xlabel(f'First Principal Component (Variance Explained: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Variance Explained: {pca.explained_variance_ratio_[1]:.2%})')
        plt.savefig(save_path)
        plt.close()

    def plot_feature_importance(self, save_path='feature_importance.png'):
        """
        Visualize feature importance through PCA loadings with a heatmap.
        
        Args:
            save_path (str): Path where the heatmap will be saved,
                           defaults to 'feature_importance.png'
        """
        # Perform PCA and get feature loadings
        pca = PCA(n_components=2)
        pca.fit(self.user_features_scaled)
        feature_names = list(self.extract_features(pd.DataFrame(), {}).keys())
        
        # Create DataFrame of PCA loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=feature_names
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            loadings,
            cmap='RdBu',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Loading Strength'}
        )
        
        plt.title('Feature Importance in Principal Components')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_cluster_profiles(self, save_path='cluster_profiles.png'):
        """
        Create heatmap visualization of average feature values for each cluster.

        Args:
            save_path (str): Path where the heatmap will be saved,
                           defaults to 'cluster_profiles.png'
        """
        # Get unique clusters and initialize feature names
        unique_clusters = np.unique(self.clusters)
        n_clusters = len(unique_clusters)
        feature_names = list(self.extract_features(pd.DataFrame(), {}).keys())
        
        # Calculate mean feature values for each cluster
        cluster_means = []
        for cluster in unique_clusters: 
            cluster_mask = self.clusters == cluster
            cluster_means.append(np.mean(self.user_features[cluster_mask], axis=0))
        
        # Create DataFrame with cluster profiles
        cluster_profiles = pd.DataFrame(
            cluster_means,
            columns=feature_names,
            index=[f'Cluster {i}' for i in range(n_clusters)]
        )
        
        # Create heatmap figure
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            cluster_profiles,
            cmap='YlOrRd',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Average Value'}
        )
        
        plt.title('Cluster Profiles: Average Feature Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def find_optimal_threshold(self, X):
        """
        Find optimal clustering threshold using binary search algorithm.
        
        Args:
            X (np.array): Scaled user features matrix where each row is a user
                         and each column is a normalized feature
            
        Returns:
            tuple: (threshold, clustering_model) where:
                  - threshold: The optimal distance threshold found
                  - clustering_model: Configured AgglomerativeClustering model
        """
        left, right = 0.1, 10.0  # Distance threshold range
        
        # Binary search for optimal threshold
        while (right - left) > 0.1:
            mid = (left + right) / 2
            
            # Create clustering model with current threshold
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=mid,
                linkage='ward'
            )
            
            n_clusters = len(np.unique(clustering.fit_predict(X)))
            
            # Adjust search bounds based on cluster count
            if n_clusters > self.max_clusters:
                left = mid
            elif n_clusters < self.min_clusters:
                right = mid
            else:
                return mid, clustering
                
        return mid, clustering

    def analyze_cluster_differences(self):
        """
        Perform one-way ANOVA to identify statistically significant features
        that distinguish between clusters.
        
        Returns:
            dict: Features with significant differences between clusters, where:
                 - Keys are feature names
                 - Values are dicts containing:
                   * 'f_statistic': F-statistic from ANOVA
                   * 'p_value': Statistical significance (p < 0.05 is significant)
        """
        feature_names = list(self.extract_features(pd.DataFrame(), {}).keys())
        significant_features = {}
        
        # Test each feature using one-way ANOVA
        for feature_idx, feature_name in enumerate(feature_names):
            f_stat, p_value = stats.f_oneway(*[
                self.user_features[self.clusters == c][:, feature_idx]
                for c in np.unique(self.clusters)
            ])
            
            # Store results for significant features (p < 0.05)
            if p_value < 0.05: 
                significant_features[feature_name] = {
                    'f_statistic': f_stat,
                    'p_value': p_value
                }
        return significant_features

    def analyze_temporal_patterns(self, interactions_df):
        """
        Analyze how user preferences evolve over time by dividing each user's
        interaction history into three periods (early, middle, late).
        
        Args:
            interactions_df (pd.DataFrame): User-song interaction data with timestamps
            
        Returns:
            dict: Temporal evolution of user preferences where:
                 - Keys are user IDs
                 - Values are dicts mapping periods to feature vectors
                 - Periods are: 'early', 'middle', 'late'
        """
        # Convert timestamp column to datetime
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        temporal_patterns = {}
        # Analyze each user's temporal patterns
        for user_id in self.user_preferences:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            if len(user_interactions) > 0:
                # Divide interactions into three equal-sized periods
                user_interactions['period'] = pd.qcut(
                    user_interactions['timestamp'], 
                    q=3,
                    labels=['early', 'middle', 'late']
                )
                
                # Calculate feature vectors for each period
                temporal_patterns[user_id] = {
                    period: self.extract_features(group, self.item_to_lyrics)
                    for period, group in user_interactions.groupby('period')
                }
        return temporal_patterns

def main():
    """
    Main execution function for the user preference learning system.
    """
    # 1. Load data
    interactions_df = pd.read_csv('../../data/STAGE1/interactions.csv')
    songs_df = pd.read_csv('../../data/music_metadata.csv')
    
    # 2. Initialize and train learner
    learner = UserPreferenceLearner(max_users=30)
    learner.learn_from_interactions(interactions_df, songs_df)
    
    # 3. Generate visualizations
    learner.plot_dendrogram(learner.user_features_scaled, 'user_preferences_dendrogram.png')
    learner.plot_clusters('user_clusters_2d.png')
    learner.plot_feature_importance('feature_importance.png')
    learner.plot_cluster_profiles('cluster_profiles.png')
    
    # Print user summaries
    for user_id in learner.user_preferences.keys():
        print("\n" + "-"*50)
        print(learner.get_user_summary(user_id))
    
    # 4. Save preferences to JSON
    preferences_dict = {
        int(user_id): prefs 
        for user_id, prefs in learner.user_preferences.items()
    }
    
    with open('user_preferences.json', 'w') as f:
        json.dump(preferences_dict, f, indent=2)

if __name__ == "__main__":
    main()