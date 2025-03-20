"""
Base data reader for the SiTunes dataset that handles:
- Loading and preprocessing interaction data
- Feature normalization and encoding
- Data splitting and imputation
- Context feature management for different experimental settings

Note: This code is adapted from the original SiTunes code.
"""

import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from models import *
from helpers.configs import *

class BaseReader:
    @staticmethod
    def parse_data_args(parser):
        """
        Add data-related command line arguments to the parser.
        
        Args:
            parser: ArgumentParser object
        Returns:
            parser: Updated ArgumentParser with data arguments
        """
        parser.add_argument('--datadir', type=str, default='../data/')
        parser.add_argument('--dataname', type=str, default='basedata')
        parser.add_argument('--load_metadata', type=int, default=1)
        parser.add_argument('--context_column_group', type=str, default='CONTEXT_all')
        return parser

    def __init__(self, args, normalize=True):
        """
        Initialize the data reader with specified arguments.
        
        Args:
            args: Parsed command line arguments
            normalize: Whether to normalize numerical features (default: True)
        """
        self.datadir = args.datadir
        self.dataname = args.dataname
        self.context_column_group = args.context_column_group
        self.normalize = normalize

        # Define context feature groups for different experimental settings:
        self.context_columns = {
            # CONTEXT_all: All available features including objective and subjective
            'CONTEXT_all': [
                'user_id', 'item_id', 'mood_improvement:label', 'emo_pre_valence',
                'emo_pre_arousal', 'time_1', 'time_2', 'time_3', 'relative_HB_mean',
                'activity_intensity_mean', 'activity_step_mean', 'relative_HB_std',
                'activity_intensity_std', 'activity_step_std', 'activity_type_0.0',
                'activity_type_1.0', 'activity_type_2.0', 'activity_type_3.0',
                'activity_type_4.0', 'weather1_0', 'weather1_1', 'weather1_2',
                'weather2', 'weather3', 'weather4', 'GPS1', 'GPS2', 'GPS3', 'timestamp'
            ],
            # CONTEXT_obj: Only objective features (time, activity, weather, location)
            'CONTEXT_obj': [
                'user_id', 'item_id', 'mood_improvement:label', 'time_1', 'time_2',
                'time_3', 'relative_HB_mean', 'activity_intensity_mean', 'activity_step_mean',
                'relative_HB_std', 'activity_intensity_std', 'activity_step_std',
                'activity_type_0.0', 'activity_type_1.0', 'activity_type_2.0', 
                'activity_type_3.0', 'activity_type_4.0', 'weather1_0', 'weather1_1',
                'weather1_2', 'weather2', 'weather3', 'weather4', 'GPS1', 'GPS2', 'GPS3'
            # CONTEXT_sub: Only subjective features (pre-listening emotional state)
            ],
            'CONTEXT_sub': [
                'user_id', 'item_id', 'mood_improvement:label', 'emo_pre_valence',
                'emo_pre_arousal'
            ]
        }

        self.load_columns = self.context_columns[self.context_column_group]
        self._load_inter_data() 
        if args.load_metadata:
            self._load_itemmeta() 
        self._split_data()
        
    def _load_inter_data(self):
        """
        Load interaction data from train/valid/test splits.
        Only loads columns that are present in all three splits.
        Applies normalization if enabled.
        """
        # Check available columns in each split
        train_columns = pd.read_csv(
            os.path.join(self.datadir, self.dataname, self.dataname + ".train.inter"),
            sep='\t', nrows=1
        ).columns
        val_columns = pd.read_csv(
            os.path.join(self.datadir, self.dataname, self.dataname + ".valid.inter"),
            sep='\t', nrows=1
        ).columns
        test_columns = pd.read_csv(
            os.path.join(self.datadir, self.dataname, self.dataname + ".test.inter"),
            sep='\t', nrows=1
        ).columns

        # Only load columns present in all splits
        available_columns = [
            col for col in self.load_columns
            if col in train_columns and col in val_columns and col in test_columns
        ]

        # Load data splits
        self.train = pd.read_csv(
            os.path.join(self.datadir, self.dataname, self.dataname + ".train.inter"),
            sep='\t', usecols=available_columns
        )
        self.val = pd.read_csv(
            os.path.join(self.datadir, self.dataname, self.dataname + ".valid.inter"),
            sep='\t', usecols=available_columns
        )
        self.test = pd.read_csv(
            os.path.join(self.datadir, self.dataname, self.dataname + ".test.inter"),
            sep='\t', usecols=available_columns
        )

        if self.normalize:
            self._normalize_features(self.train, [self.val, self.test])

    def _load_itemmeta(self):
        """Load and merge item metadata with interaction data."""
        self.item_meta = pd.read_csv(os.path.join(self.datadir, self.dataname, self.dataname + ".item"), sep='\t')
        if self.normalize:
            self._normalize_features(self.item_meta, [])

        # Merge metadata with interaction data
        self.train = self.train.merge(self.item_meta, left_on='item_id', right_on='i_id_c', how="left")
        self.val = self.val.merge(self.item_meta, left_on='item_id', right_on='i_id_c', how="left")
        self.test = self.test.merge(self.item_meta, left_on='item_id', right_on='i_id_c', how="left")

    def _normalize_features(self, fit_df, transform_dfs):
        """
        Normalize and encode features:
        - StandardScaler for numerical features (suffix ':float')
        - OneHotEncoder for categorical features (suffix ':token')
        
        Args:
            fit_df: DataFrame to fit transformers on
            transform_dfs: List of DataFrames to apply transformations to
        """
        ss_features, enc_features = [], []
        for col in fit_df:
            if col.split(":")[-1] == 'float':
                ss_features.append(col)
            elif col.split(":")[-1] == 'token' and col not in [UID, IID]:
                enc_features.append(col)
        
        # Standardize numerical features
        if ss_features:
            scaler = preprocessing.StandardScaler().fit(fit_df[ss_features])
            fit_df[ss_features] = scaler.transform(fit_df[ss_features])
            for df in transform_dfs:
                df[ss_features] = scaler.transform(df[ss_features])
        
        # One-hot encode categorical features
        if enc_features:
            enc = preprocessing.OneHotEncoder().fit(fit_df[enc_features])
            out_shape = enc.transform(fit_df[enc_features]).shape[1]
            fit_df[['enc_%d' % i for i in range(out_shape)]] = enc.transform(fit_df[enc_features]).toarray()
            fit_df.drop(columns=enc_features, inplace=True)
            for df in transform_dfs:
                df[['enc_%d' % i for i in range(out_shape)]] = enc.transform(df[enc_features]).toarray()
                df.drop(columns=enc_features, inplace=True)

    def _split_data(self):
        """
        Prepare final X, y splits for training:
        - Selects numerical features for X
        - Uses mood_improvement:label as y
        - Applies median imputation for missing values
        """
        X_columns = self.train.drop(columns=['mood_improvement:label']).select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        
        # Prepare train split
        self.train_X = imputer.fit_transform(self.train[X_columns])
        self.train_y = self.train['mood_improvement:label']
        
        # Prepare validation split
        self.val_X = imputer.transform(self.val[X_columns])
        self.val_y = self.val['mood_improvement:label']
        
        # Prepare test split
        self.test_X = imputer.transform(self.test[X_columns])
        self.test_y = self.test['mood_improvement:label']
        
        print("Train:", self.train_X.shape)
        print("Validation:", self.val_X.shape)
        print("Test:", self.test_X.shape)
