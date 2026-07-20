"""
Enhanced SOZ Framework with 10-Fold Cross-Validation, SMOTE, and Configurable Time Bins
- 10-fold subject-level cross-validation (each subject appears only once in a fold)
- SMOTE for handling class imbalance (11:1 ratio)
- Use all Engel class subjects (not just Class 1)
- Early normalization on n_metrics dimension before temporal aggregation
- Hyperparameter search with stratified 5-fold CV for imbalanced data
- Configurable time bins for temporal aggregation
- Save results to Excel files
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
from pathlib import Path
import time
from collections import defaultdict
warnings.filterwarnings('ignore')

class EnhancedSOZFramework:
    def __init__(self, data_path, soz_path, participants_file, resection_path=None, time_bins=10):
        self.data_path = data_path
        self.soz_path = soz_path
        self.participants_file = participants_file
        self.resection_path = resection_path
        
        # Configurable temporal aggregation parameters
        self.time_bins = time_bins  # Number of temporal bins for aggregation
        
        self.subjects_data = {}
        self.soz_indices = None
        self.participants_info = None
        self.engel_scores = None
        self.resection_indices = None
        
        self.optimal_threshold = 0.5
        self.results_method1 = {}
        
        # 10-fold cross-validation configuration
        self.n_folds = 10
        self.use_smote = True
        
        # Hyperparameter search configuration
        self.use_hyperparameter_search = True
        self.n_iter_search = 20
        self.cv_folds = 5  # Inner CV for hyperparameter search
        self.search_cv_scoring = 'roc_auc'

        self.sampling_method = 'smote'  # Options: 'smote', 'undersample', 'none'
        self.use_smote = True  # Keep for backward compatibility
        
        print(f"Framework initialized with {self.time_bins} temporal bins")
        
    def get_hyperparameter_search_space(self):
        """Define hyperparameter search space for Random Forest"""
        param_dist = {
            'n_estimators': [20, 50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'bootstrap': [True, False]
        }
        return param_dist
    
    def normalize_metrics_early(self, data):
        """
        Early Min-Max normalization on n_metrics dimension
        Input: (n_channels, n_windows, n_eig, n_metrics)
        Output: (n_channels, n_windows, n_eig, n_metrics) with normalized metrics
        """
        n_channels, n_windows, n_eig, n_metrics = data.shape
        normalized_data = np.copy(data)
        
        # Normalize each metric across all channels, windows, and eigenvectors
        for metric_idx in range(n_metrics):
            metric_data = data[:, :, :, metric_idx]
            
            # Flatten for normalization
            flat_data = metric_data.flatten()
            valid_mask = ~np.isnan(flat_data)
            
            if np.sum(valid_mask) > 0:
                valid_data = flat_data[valid_mask]
                data_min = np.min(valid_data)
                data_max = np.max(valid_data)
                data_range = data_max - data_min
                
                if data_range > 0:
                    # Normalize and reshape back
                    normalized_flat = (flat_data - data_min) / data_range
                    normalized_flat = np.clip(normalized_flat, 0, 1)
                    normalized_data[:, :, :, metric_idx] = normalized_flat.reshape(n_channels, n_windows, n_eig)
                else:
                    # If all values are the same, set to 0.5
                    normalized_data[:, :, :, metric_idx] = 0.5
        
        return normalized_data
        
    def load_data(self):
        """Load all required data"""
        print("Loading data...")
        
        # Load SOZ indices
        try:
            with open(self.soz_path, 'rb') as f:
                self.soz_indices = pickle.load(f)
            print(f"Successfully loaded SOZ indices for {len(self.soz_indices)} subjects")
        except Exception as e:
            print(f"Failed to load SOZ indices: {str(e)}")
            return False
        
        # Load participant information and Engel scores
        self._load_participants_and_engel()
        
        # Load resection indices if available
        if self.resection_path and os.path.exists(self.resection_path):
            self._load_resection_indices()
        
        # Load graph data
        return self._load_graph_data()
    
    def _load_participants_and_engel(self):
        """Load participant information and Engel scores"""
        try:
            self.participants_info = pd.read_excel(self.participants_file)
            print(f"Successfully loaded participant information")
            
            if 'engel' in self.participants_info.columns:
                self.engel_scores = {}
                for _, row in self.participants_info.iterrows():
                    participant_id = str(row['participant_id'])
                    possible_ids = [
                        participant_id, f"subject_{participant_id}",
                        participant_id.replace('HUP', ''), f"sub-{participant_id}"
                    ]
                    
                    engel_score = row['engel']
                    if pd.notna(engel_score):
                        try:
                            if isinstance(engel_score, str):
                                engel_numeric = int(''.join(filter(str.isdigit, engel_score)))
                            else:
                                engel_numeric = int(engel_score)
                            
                            for id_format in possible_ids:
                                self.engel_scores[id_format] = engel_numeric
                        except (ValueError, TypeError):
                            continue
                
                print(f"Successfully loaded Engel scores")
        except Exception as e:
            print(f"Failed to load participant information: {str(e)}")
    
    def _load_resection_indices(self):
        """Load resection indices"""
        try:
            with open(self.resection_path, 'rb') as f:
                self.resection_indices = pickle.load(f)
            print(f"Successfully loaded resection indices")
        except Exception as e:
            print(f"Failed to load resection indices: {str(e)}")
            self.resection_indices = None
    
    def _load_graph_data(self):
        """Load graph theory data"""
        try:
            data_folders = [f for f in os.listdir(self.data_path) if f.startswith('HUP')]#'subject_'
            print(f"Found {len(data_folders)} subject folders")
            
            valid_subjects = 0
            for subject_id in data_folders:
                subject_path = os.path.join(self.data_path, subject_id)
                if os.path.isdir(subject_path):
                    pkl_files = [f for f in os.listdir(subject_path) if f.endswith('.pkl') and 'run' in f.lower()]
                    if pkl_files:
                        subject_data = {}
                        for pkl_file in pkl_files:
                            pkl_path = os.path.join(subject_path, pkl_file)
                            try:
                                with open(pkl_path, 'rb') as f:
                                    run_data = pickle.load(f)
                                processed_data = self._process_data(run_data)
                                if processed_data is not None:
                                    run_key = pkl_file.replace('.pkl', '')
                                    subject_data[run_key] = processed_data
                            except:
                                continue
                        
                        if subject_data:
                            self.subjects_data[subject_id] = subject_data
                            valid_subjects += 1
            
            print(f"Successfully loaded graph data for {valid_subjects} subjects")
            return True
            
        except Exception as e:
            print(f"Failed to load graph data: {str(e)}")
            return False
    

    def _process_data(self, raw_data):
        """Process raw data to required format with early normalization"""
        try:
            if isinstance(raw_data, np.ndarray) and len(raw_data.shape) == 4:
                n_windows, n_channels, n_eig, n_metrics = raw_data.shape
                if n_eig == 10 and n_metrics == 4:
                    # Transpose to (n_channels, n_windows, n_eig, n_metrics)
                    # data = np.transpose(raw_data, (1, 0, 2, 3))
                    data = raw_data
                    # Apply early normalization on metrics
                    normalized_data = self.normalize_metrics_early(data)
                    return normalized_data
            return None
        except:
            return None
    
    def temporal_aggregation_method1(self, data):
        """
        Method 1: Average 240 windows into configurable number of bins
        Output: (n_channels, time_bins, 10, 4) -> (n_channels, time_bins*10*4) features
        Data is already normalized in early processing
        """
        n_channels, n_timepoints, n_eigenvectors, n_metrics = data.shape
        
        # Divide 240 time windows into self.time_bins
        bin_size = n_timepoints // self.time_bins
        aggregated_data = np.zeros((n_channels, self.time_bins, n_eigenvectors, n_metrics))
        
        for i in range(self.time_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size
            if i == self.time_bins - 1:  # Last bin takes remaining windows
                end_idx = n_timepoints
            
            # Average over the time windows in this bin
            aggregated_data[:, i, :, :] = np.mean(data[:, start_idx:end_idx, :, :], axis=1)
        
        # Reshape to (n_channels, time_bins*10*4)
        features = aggregated_data.reshape(n_channels, -1)
        
        print(f"Temporal aggregation: {n_timepoints} windows -> {self.time_bins} bins, "
              f"Features shape: {features.shape}")
        
        return features
    
    def filter_subjects_by_engel(self, subject_ids, engel_classes=None):
        """Filter subjects by Engel class"""
        if engel_classes is None:
            # Include all subjects with valid Engel scores
            valid_subjects = []
            for subject_id in subject_ids:
                engel_score = self.get_subject_engel_score(subject_id)
                if engel_score is not None:
                    valid_subjects.append(subject_id)
            print(f"Using all subjects with valid Engel scores: {len(valid_subjects)} out of {len(subject_ids)}")
            return valid_subjects
        else:
            # Filter by specific Engel classes
            filtered_subjects = []
            for subject_id in subject_ids:
                engel_score = self.get_subject_engel_score(subject_id)
                if engel_score in engel_classes:
                    filtered_subjects.append(subject_id)
            
            print(f"Filtered to Engel Class {engel_classes} subjects: {len(filtered_subjects)} out of {len(subject_ids)}")
            return filtered_subjects
    
    def create_subject_level_cv_folds(self, subject_ids):
        """
        Create 10-fold cross-validation splits at subject level
        Each subject appears in exactly one fold
        """
        n_subjects = len(subject_ids)
        
        # If we have fewer subjects than folds, reduce number of folds
        n_folds = min(self.n_folds, n_subjects)
        if n_folds < self.n_folds:
            print(f"Warning: Reducing folds to {n_folds} due to limited subjects ({n_subjects})")
        
        # Shuffle subjects for random assignment to folds
        np.random.seed(42)  # For reproducibility
        shuffled_subjects = np.array(subject_ids.copy())
        np.random.shuffle(shuffled_subjects)
        
        # Create folds
        fold_size = n_subjects // n_folds
        remainder = n_subjects % n_folds
        
        folds = []
        start_idx = 0
        
        for fold_idx in range(n_folds):
            # Add one extra subject to first 'remainder' folds
            current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
            end_idx = start_idx + current_fold_size
            
            fold_subjects = shuffled_subjects[start_idx:end_idx].tolist()
            folds.append(fold_subjects)
            
            start_idx = end_idx
        
        # Verify all subjects are assigned
        all_fold_subjects = [subj for fold in folds for subj in fold]
        assert len(all_fold_subjects) == n_subjects, "Not all subjects assigned to folds"
        assert len(set(all_fold_subjects)) == n_subjects, "Duplicate subjects in folds"
        
        print(f"Created {n_folds} folds with sizes: {[len(fold) for fold in folds]}")
        return folds
    
    def apply_sampling_method(self, X_train, y_train):
        """Apply the selected sampling method to balance training data"""
        # Print class distribution before sampling
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        print(f"Before sampling: {dict(zip(unique_classes, class_counts))}")
        
        if self.sampling_method == 'smote':
            return self.apply_smote_to_training_data(X_train, y_train)
        elif self.sampling_method == 'undersample':
            return self.balance_training_data(X_train, y_train)
        elif self.sampling_method == 'none':
            print("No sampling applied, using original class distribution")
            return X_train, y_train
        else:
            print(f"Unknown sampling method: {self.sampling_method}, using original data")
            return X_train, y_train
        

    def apply_smote_to_training_data(self, X_train, y_train):
        """
        Apply SMOTE to balance SOZ and non-SOZ nodes in training data
        """
        try:
            # Check if we have both classes
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            
            if len(unique_classes) < 2:
                print(f"Warning: Only one class present, skipping SMOTE")
                return X_train, y_train
            
            # Check minimum class size for SMOTE
            min_samples = np.min(class_counts)
            if min_samples < 2:
                print(f"Warning: Minimum class has only {min_samples} samples, cannot apply SMOTE")
                return X_train, y_train
            
            print(f"Before SMOTE: {dict(zip(unique_classes, class_counts))}")
            
            # Apply SMOTE with appropriate k_neighbors
            k_neighbors = min(5, min_samples - 1)
            if k_neighbors < 1:
                k_neighbors = 1
            
            smote = SMOTE(
                random_state=42,
                k_neighbors=k_neighbors,
                sampling_strategy='auto'  # Balance to majority class
            )
            
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            # Check results
            unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
            print(f"After SMOTE: {dict(zip(unique_resampled, counts_resampled))}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"Error applying SMOTE: {str(e)}")
            print("Falling back to original data")
            return X_train, y_train
        
    def balance_training_data(self, X_train, y_train):
        """Balance SOZ and non-SOZ nodes in training data"""
        soz_indices = np.where(y_train == 1)[0]
        nonsoz_indices = np.where(y_train == 0)[0]
        
        n_soz = len(soz_indices)
        if n_soz == 0:
            return X_train, y_train
        
        # Sample non-SOZ nodes equal to SOZ nodes (with replacement if needed)
        if len(nonsoz_indices) >= n_soz:
            selected_nonsoz = np.random.choice(nonsoz_indices, size=n_soz, replace=False)
        else:
            selected_nonsoz = np.random.choice(nonsoz_indices, size=n_soz, replace=True)
        
        # Combine balanced indices
        balanced_indices = np.concatenate([soz_indices, selected_nonsoz])
        np.random.shuffle(balanced_indices)

        # print(f"Random Undersampling: {n_soz} SOZ nodes, {n_soz} non-SOZ nodes (sampled from {len(nonsoz_indices)})")
        
        return X_train[balanced_indices], y_train[balanced_indices]
    
    def create_stratified_cv_splits(self, X, y, n_splits=5):
        """
        Create stratified CV splits for hyperparameter search
        """
        try:
            # Check if we have both classes
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                print(f"Warning: Only one class present in hyperparameter search data")
                return None
            
            # Calculate minimum samples per class
            class_counts = np.bincount(y.astype(int))
            min_samples = np.min(class_counts[class_counts > 0])
            
            # If we don't have enough samples for stratification, reduce n_splits
            if min_samples < n_splits:
                n_splits = max(2, min_samples)
                print(f"Reducing inner CV folds to {n_splits} due to limited samples")
            
            # Create stratified splits
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_splits = list(skf.split(X, y))
            
            # Verify each split has both classes
            valid_splits = []
            for train_idx, val_idx in cv_splits:
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]
                
                # Check if both train and validation have both classes
                if len(np.unique(y_train_fold)) == 2 and len(np.unique(y_val_fold)) == 2:
                    valid_splits.append((train_idx, val_idx))
                else:
                    print(f"Skipping fold with insufficient class representation")
            
            if len(valid_splits) < 2:
                print(f"Warning: Only {len(valid_splits)} valid inner CV folds")
                return None
            
            return valid_splits
            
        except Exception as e:
            print(f"Error creating stratified CV splits: {str(e)}")
            return None
    
    def train_random_forest_with_search(self, X_train, y_train):
        """
        Train Random Forest with selected sampling method and hyperparameter search
        """
        start_time = time.time()
        
        # Apply selected sampling method for class balancing
        X_balanced, y_balanced = self.apply_sampling_method(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_balanced)
        
        if not self.use_hyperparameter_search:
            # Use default parameters if search is disabled
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_scaled, y_balanced)
            return rf, scaler, None
        
        # Create stratified CV splits for hyperparameter search
        cv_splits = self.create_stratified_cv_splits(X_scaled, y_balanced, self.cv_folds)
        
        if cv_splits is None or len(cv_splits) < 2:
            print("Falling back to default parameters due to insufficient CV splits")
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_scaled, y_balanced)
            return rf, scaler, None
        
        try:
            # Set up the random search
            rf_base = RandomForestClassifier(
                random_state=42,
                n_jobs=1,  # Use 1 to allow RandomizedSearchCV to parallelize
                class_weight='balanced'
            )
            
            param_dist = self.get_hyperparameter_search_space()
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                estimator=rf_base,
                param_distributions=param_dist,
                n_iter=self.n_iter_search,
                cv=cv_splits,
                scoring=self.search_cv_scoring,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            print(f"Starting hyperparameter search with {len(cv_splits)} inner CV folds...")
            random_search.fit(X_scaled, y_balanced)
            
            search_time = time.time() - start_time
            print(f"Hyperparameter search completed in {search_time:.2f}s")
            print(f"Best CV score: {random_search.best_score_:.4f}")
            print(f"Best parameters: {random_search.best_params_}")
            
            # Train final model with best parameters
            best_rf = random_search.best_estimator_
            
            return best_rf, scaler, random_search.best_params_
            
        except Exception as e:
            print(f"Error in hyperparameter search: {str(e)}")
            print("Falling back to default parameters")
            
            # Fallback to default parameters
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_scaled, y_balanced)
            return rf, scaler, None
    
    def ten_fold_cross_validation(self, aggregation_method='method1', engel_classes=None):
        """
        10-fold subject-level cross-validation with SMOTE
        Each subject appears in exactly one fold
        """
        print(f"Starting 10-fold CV with {aggregation_method} (All Engel classes)")
        print(f"Time bins: {self.time_bins}")
        print(f"Sampling method: {self.sampling_method}")
        print(f"Hyperparameter search: {'Enabled' if self.use_hyperparameter_search else 'Disabled'}")
        
        subject_ids = list(self.subjects_data.keys())
        valid_subject_ids = [subject_ids[i] for i in range(min(len(subject_ids), len(self.soz_indices)))]
        
        # Filter subjects by Engel class (use all if engel_classes is None)
        engel_subjects = self.filter_subjects_by_engel(valid_subject_ids, engel_classes)
        
        if len(engel_subjects) < 2:
            print(f"Not enough subjects for cross-validation: {len(engel_subjects)}")
            return {}, [], []
        
        # Create subject-level folds
        subject_folds = self.create_subject_level_cv_folds(engel_subjects)
        n_folds_actual = len(subject_folds)
        
        results = {}
        fold_metrics = []  # Store metrics for each fold
        subject_metrics = []  # Store metrics for each subject
        run_metrics = []      # Store metrics for each run
        hyperparameter_log = []  # Store hyperparameter search results
        
        for fold_idx, test_subjects in enumerate(subject_folds):
            print(f"\nProcessing fold {fold_idx + 1}/{n_folds_actual}")
            print(f"Test subjects: {test_subjects}")
            
            try:
                # Prepare training data (from all other folds)
                train_subjects = []
                for other_fold_idx, other_fold_subjects in enumerate(subject_folds):
                    if other_fold_idx != fold_idx:
                        train_subjects.extend(other_fold_subjects)
                
                print(f"Training subjects: {len(train_subjects)}")
                
                X_train_list, y_train_list = [], []
                
                # Collect training data from all training subjects
                for train_subject in train_subjects:
                    if train_subject not in engel_subjects:
                        continue
                        
                    train_idx = valid_subject_ids.index(train_subject)
                    subject_data = self.subjects_data[train_subject]
                    soz_nodes = self.soz_indices[train_idx]
                    
                    for run_name, run_data in subject_data.items():
                        # Apply aggregation method
                        if aggregation_method == 'method1':
                            features = self.temporal_aggregation_method1(run_data)
                        else:
                            print(f"Unknown aggregation method: {aggregation_method}")
                            continue
                        
                        n_channels = features.shape[0]
                        labels = np.zeros(n_channels)
                        if isinstance(soz_nodes, (list, np.ndarray)) and len(soz_nodes) > 0:
                            valid_soz = [idx for idx in soz_nodes if idx < n_channels]
                            if valid_soz:
                                labels[valid_soz] = 1
                        
                        X_train_list.append(features)
                        y_train_list.append(labels)
                
                if not X_train_list:
                    print(f"No training data for fold {fold_idx + 1}, skipping")
                    continue
                
                X_train = np.concatenate(X_train_list, axis=0)
                y_train = np.concatenate(y_train_list, axis=0)
                
                if len(np.unique(y_train)) < 2:
                    print(f"Skipping fold {fold_idx + 1}: insufficient class diversity")
                    continue
                
                print(f"Training data: {len(y_train)} samples, {np.sum(y_train)} SOZ nodes")
                
                # Train classifier with SMOTE and hyperparameter search
                classifier, scaler, best_params = self.train_random_forest_with_search(X_train, y_train)
                
                # Log hyperparameter search results
                hyperparameter_log.append({
                    'fold': fold_idx + 1,
                    'test_subjects': test_subjects,
                    'method': aggregation_method,
                    'time_bins': self.time_bins,
                    'best_params': best_params,
                    'training_samples': len(y_train),
                    'soz_samples': int(np.sum(y_train))
                })
                
                # Test on held-out subjects
                fold_predictions = []
                fold_subject_metrics = []
                
                for test_subject in test_subjects:
                    if test_subject not in engel_subjects:
                        continue
                        
                    test_subject_data = self.subjects_data[test_subject]
                    test_idx = valid_subject_ids.index(test_subject)
                    test_soz_nodes = self.soz_indices[test_idx]
                    
                    run_predictions = []
                    run_metrics_subject = []
                    
                    for run_name, run_data in test_subject_data.items():
                        # Apply same aggregation method
                        if aggregation_method == 'method1':
                            features = self.temporal_aggregation_method1(run_data)
                        else:
                            continue
                        
                        X_test_scaled = scaler.transform(features)
                        y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
                        y_pred = (y_pred_proba > self.optimal_threshold).astype(int)
                        
                        n_channels = features.shape[0]
                        y_true = np.zeros(n_channels)
                        if isinstance(test_soz_nodes, (list, np.ndarray)) and len(test_soz_nodes) > 0:
                            valid_soz = [idx for idx in test_soz_nodes if idx < n_channels]
                            if valid_soz:
                                y_true[valid_soz] = 1
                        
                        # Calculate run-level metrics
                        run_acc, run_sens, run_spec, run_auc = self.calculate_metrics(y_true, y_pred, y_pred_proba)
                        
                        engel_score = self.get_subject_engel_score(test_subject)
                        
                        run_metrics_subject.append({
                            'fold': fold_idx + 1,
                            'subject': test_subject,
                            'run': run_name,
                            'accuracy': run_acc,
                            'sensitivity': run_sens, 
                            'specificity': run_spec,
                            'auc': run_auc,
                            'engel': engel_score,
                            'method': aggregation_method,
                            'time_bins': self.time_bins
                        })
                        
                        run_predictions.append({
                            'y_true': y_true, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba
                        })
                    
                    if not run_predictions:
                        continue
                    
                    # Aggregate predictions across runs for subject-level result
                    if len(run_predictions) > 1:
                        avg_proba = np.mean([pred['y_pred_proba'] for pred in run_predictions], axis=0)
                        final_pred = (avg_proba > self.optimal_threshold).astype(int)
                    else:
                        avg_proba = run_predictions[0]['y_pred_proba']
                        final_pred = run_predictions[0]['y_pred']
                    
                    y_true_final = run_predictions[0]['y_true']
                    
                    # FP-resected analysis
                    fp_resected_analysis = self.analyze_fp_resected(
                        y_true_final, final_pred, self.get_subject_resection_indices(test_subject)
                    )
                    
                    # Feature importance analysis
                    feature_analysis = self.analyze_feature_importance(classifier.feature_importances_)
                    
                    # Calculate subject-level metrics
                    subj_acc, subj_sens, subj_spec, subj_auc = self.calculate_metrics(
                        y_true_final, final_pred, avg_proba)
                    
                    print(f"  {test_subject} - AUC: {subj_auc:.4f}, Acc: {subj_acc:.4f}")
                    
                    # Store results
                    results[f"fold_{fold_idx+1}_{test_subject}"] = {
                        'fold': fold_idx + 1,
                        'subject': test_subject,
                        'y_true': y_true_final,
                        'y_pred': final_pred,
                        'y_pred_proba': avg_proba,
                        'soz_nodes': test_soz_nodes,
                        'engel_score': engel_score,
                        'resection_indices': self.get_subject_resection_indices(test_subject),
                        'feature_importances': classifier.feature_importances_,
                        'run_metrics': run_metrics_subject,
                        'subject_metrics': {
                            'accuracy': subj_acc, 'sensitivity': subj_sens,
                            'specificity': subj_spec, 'auc': subj_auc
                        },
                        'time_bins': self.time_bins,
                        'best_hyperparams': best_params,
                        'fp_resected_analysis': fp_resected_analysis,
                        'feature_analysis': feature_analysis
                    }
                    
                    fold_subject_metrics.append({
                        'fold': fold_idx + 1,
                        'subject': test_subject,
                        'accuracy': subj_acc,
                        'sensitivity': subj_sens,
                        'specificity': subj_spec,
                        'auc': subj_auc,
                        'engel': engel_score,
                        'method': aggregation_method,
                        'time_bins': self.time_bins
                    })
                    
                    fold_predictions.append({
                        'subject': test_subject,
                        'y_true': y_true_final,
                        'y_pred': final_pred,
                        'y_pred_proba': avg_proba,
                        'metrics': {
                            'accuracy': subj_acc, 'sensitivity': subj_sens,
                            'specificity': subj_spec, 'auc': subj_auc
                        }
                    })
                    
                    run_metrics.extend(run_metrics_subject)
                
                # Calculate fold-level metrics (average across subjects in fold)
                if fold_subject_metrics:
                    fold_aucs = [m['auc'] for m in fold_subject_metrics]
                    fold_accs = [m['accuracy'] for m in fold_subject_metrics]
                    fold_sens = [m['sensitivity'] for m in fold_subject_metrics]
                    fold_specs = [m['specificity'] for m in fold_subject_metrics]
                    
                    fold_metrics.append({
                        'fold': fold_idx + 1,
                        'n_subjects': len(fold_subject_metrics),
                        'mean_auc': np.mean(fold_aucs),
                        'mean_accuracy': np.mean(fold_accs),
                        'mean_sensitivity': np.mean(fold_sens),
                        'mean_specificity': np.mean(fold_specs),
                        'std_auc': np.std(fold_aucs),
                        'method': aggregation_method,
                        'time_bins': self.time_bins
                    })
                    
                    subject_metrics.extend(fold_subject_metrics)
                    
                    print(f"Fold {fold_idx + 1} completed - Mean AUC: {np.mean(fold_aucs):.4f}")
                
            except Exception as e:
                print(f"Error processing fold {fold_idx + 1}: {str(e)}")
                continue
        
        print(f"\n10-fold CV completed! Successfully processed: {len(subject_metrics)} subjects across {len(fold_metrics)} folds")
        
        # Print hyperparameter analysis
        if hyperparameter_log and any(log['best_params'] for log in hyperparameter_log):
            print(f"\nHyperparameter Search Summary:")
            param_frequency = {}
            for log in hyperparameter_log:
                if log['best_params']:
                    for param, value in log['best_params'].items():
                        if param not in param_frequency:
                            param_frequency[param] = {}
                        if value not in param_frequency[param]:
                            param_frequency[param][value] = 0
                        param_frequency[param][value] += 1
            
            for param, values in param_frequency.items():
                most_common = max(values, key=values.get)
                print(f"   {param}: {most_common} (selected {values[most_common]}/{len(hyperparameter_log)} times)")
        
        # Store results
        self.results_method1 = results
        
        return results, subject_metrics, run_metrics, fold_metrics, hyperparameter_log
    
    def analyze_fp_resected(self, y_true, y_pred, resection_indices):
        """
        Analyze False Positives that are resected nodes
        """
        # Find False Positive nodes
        fp_mask = (y_pred == 1) & (y_true == 0)
        fp_indices = np.where(fp_mask)[0]
        
        if len(fp_indices) == 0:
            return {
                'total_fp': 0,
                'fp_resected_count': 0,
                'fp_resected_ratio': 0.0,
                'fp_indices': [],
                'fp_resected_indices': []
            }
        
        # Check FP nodes that are in resection_indices
        fp_resected_indices = []
        if resection_indices and len(resection_indices) > 0:
            valid_resection_indices = [idx for idx in resection_indices if idx < len(y_true)]
            fp_resected_indices = [idx for idx in fp_indices if idx in valid_resection_indices]
        
        fp_resected_count = len(fp_resected_indices)
        fp_resected_ratio = fp_resected_count / len(fp_indices) if len(fp_indices) > 0 else 0.0
        
        return {
            'total_fp': len(fp_indices),
            'fp_resected_count': fp_resected_count,
            'fp_resected_ratio': fp_resected_ratio,
            'fp_indices': fp_indices.tolist(),
            'fp_resected_indices': fp_resected_indices
        }

    def analyze_feature_importance(self, feature_importances):
        """
        Analyze feature importance with configurable dimensions
        Features: (time_bins × 10 eigenvalues × 4 metrics) = time_bins*40
        """
        expected_features = self.time_bins * 10 * 4
        
        if len(feature_importances) != expected_features:
            print(f"Warning: Expected {expected_features} features but got {len(feature_importances)}")
            return {}
        
        # Reshape features to (time_bins, 10, 4)
        # Corresponding to (time windows, eigenvalues, metrics)
        importance_3d = feature_importances.reshape(self.time_bins, 10, 4)
        
        # Aggregate by time windows (which time period is most important)
        time_importance = np.sum(importance_3d, axis=(1, 2))  # (time_bins,)
        
        # Aggregate by metric type (which graph metric is most important)
        metric_importance = np.sum(importance_3d, axis=(0, 1))  # (4,)
        
        # Aggregate by eigenvalues (which eigenvalues are most important)
        eigenvalue_importance = np.sum(importance_3d, axis=(0, 2))  # (10,)
        
        # Find most important feature combination
        max_idx = np.unravel_index(np.argmax(importance_3d), importance_3d.shape)
        max_time, max_eigen, max_metric = max_idx
        
        return {
            'time_importance': time_importance,
            'metric_importance': metric_importance,
            'eigenvalue_importance': eigenvalue_importance,
            'most_important_combination': {
                'time_window': max_time,
                'eigenvalue': max_eigen,
                'metric': max_metric,
                'importance_value': importance_3d[max_idx]
            },
            'top_5_features': self._get_top_features(feature_importances, top_k=5)
        }

    def _get_top_features(self, feature_importances, top_k=5):
        """Get top k most important features with configurable dimensions"""
        top_indices = np.argsort(feature_importances)[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            # Convert linear index to 3D index (time_bins, 10, 4)
            time_idx = idx // (10 * 4)
            remaining = idx % (10 * 4)
            eigen_idx = remaining // 4
            metric_idx = remaining % 4
            
            top_features.append({
                'rank': len(top_features) + 1,
                'linear_index': idx,
                'time_window': time_idx,
                'eigenvalue': eigen_idx,
                'metric': metric_idx,
                'importance': feature_importances[idx]
            })
        
        return top_features
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge cases
                if np.all(y_true == 0) and np.all(y_pred == 0):
                    tn, fp, fn, tp = len(y_true), 0, 0, 0
                elif np.all(y_true == 1) and np.all(y_pred == 1):
                    tn, fp, fn, tp = 0, 0, 0, len(y_true)
                else:
                    return 0, 0, 0, 0.5
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            try:
                auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
            except:
                auc = 0.5
            
            return accuracy, sensitivity, specificity, auc
            
        except Exception as e:
            return 0, 0, 0, 0.5
    
    def get_subject_engel_score(self, subject_id):
        """Get Engel score for a subject"""
        if self.engel_scores is None:
            return None
        
        possible_ids = [
            subject_id, subject_id.replace('subject_', ''),
            f"subject_{subject_id}", subject_id.replace('sub-', ''),
            f"sub-{subject_id}"
        ]
        
        for id_format in possible_ids:
            if id_format in self.engel_scores:
                return self.engel_scores[id_format]
        return None
    
    def get_subject_resection_indices(self, subject_id):
        """Get resection indices for a subject"""
        if self.resection_indices is None:
            return None
        
        possible_ids = [
            subject_id, subject_id.replace('subject_', ''),
            f"subject_{subject_id}", subject_id.replace('sub-', ''),
            f"sub-{subject_id}"
        ]
        
        for id_format in possible_ids:
            if id_format in self.resection_indices:
                return self.resection_indices[id_format]
        return []
    
    def save_results_to_excel(self, results, subject_metrics, run_metrics, fold_metrics, hyperparameter_log, save_path):
        """Save 10-fold CV results to Excel file"""
        try:
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                
                # Sheet 1: Subject Results
                if subject_metrics:
                    subject_df = pd.DataFrame(subject_metrics)
                    subject_df.to_excel(writer, sheet_name='Subject_Results', index=False)
                
                # Sheet 2: Run Results
                if run_metrics:
                    run_df = pd.DataFrame(run_metrics)
                    run_df.to_excel(writer, sheet_name='Run_Results', index=False)
                
                # Sheet 3: Fold Results
                if fold_metrics:
                    fold_df = pd.DataFrame(fold_metrics)
                    fold_df.to_excel(writer, sheet_name='Fold_Results', index=False)
                
                # Sheet 4: Overall Statistics
                if subject_metrics:
                    aucs = [m['auc'] for m in subject_metrics]
                    accs = [m['accuracy'] for m in subject_metrics]
                    sens = [m['sensitivity'] for m in subject_metrics]
                    specs = [m['specificity'] for m in subject_metrics]
                    
                    stats_data = [
                        {'Metric': 'AUC', 'Mean': np.mean(aucs), 'Std': np.std(aucs), 
                         'Min': np.min(aucs), 'Max': np.max(aucs), 'N_Subjects': len(aucs),
                         'Time_Bins': self.time_bins},
                        {'Metric': 'Accuracy', 'Mean': np.mean(accs), 'Std': np.std(accs),
                         'Min': np.min(accs), 'Max': np.max(accs), 'N_Subjects': len(accs),
                         'Time_Bins': self.time_bins},
                        {'Metric': 'Sensitivity', 'Mean': np.mean(sens), 'Std': np.std(sens),
                         'Min': np.min(sens), 'Max': np.max(sens), 'N_Subjects': len(sens),
                         'Time_Bins': self.time_bins},
                        {'Metric': 'Specificity', 'Mean': np.mean(specs), 'Std': np.std(specs),
                         'Min': np.min(specs), 'Max': np.max(specs), 'N_Subjects': len(specs),
                         'Time_Bins': self.time_bins}
                    ]
                    
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Overall_Statistics', index=False)
                
                # Sheet 5: Hyperparameter Analysis
                if hyperparameter_log:
                    hyperparam_data = []
                    
                    for log in hyperparameter_log:
                        if log.get('best_params'):
                            hyperparam_entry = {
                                'Fold': log['fold'],
                                'Test_Subjects': str(log['test_subjects']),
                                'Method': log['method'],
                                'Time_Bins': log['time_bins'],
                                'Training_Samples': log['training_samples'],
                                'SOZ_Samples': log['soz_samples']
                            }
                            # Add hyperparameter values
                            for param, value in log['best_params'].items():
                                hyperparam_entry[f'best_{param}'] = value
                            
                            hyperparam_data.append(hyperparam_entry)
                    
                    if hyperparam_data:
                        hyperparam_df = pd.DataFrame(hyperparam_data)
                        hyperparam_df.to_excel(writer, sheet_name='Hyperparameter_Analysis', index=False)
                
                # Sheet 6: FP-Resected Analysis
                if results:
                    fp_resected_data = []
                    
                    for result_key, result in results.items():
                        fp_analysis = result.get('fp_resected_analysis', {})
                        
                        fp_resected_data.append({
                            'Fold': result.get('fold', ''),
                            'Subject': result.get('subject', ''),
                            'Time_Bins': result.get('time_bins', self.time_bins),
                            'Engel_Score': result['engel_score'],
                            'AUC': result['subject_metrics']['auc'],
                            'Accuracy': result['subject_metrics']['accuracy'],
                            'Sensitivity': result['subject_metrics']['sensitivity'],
                            'Specificity': result['subject_metrics']['specificity'],
                            'Total_FP': fp_analysis.get('total_fp', 0),
                            'FP_Resected_Count': fp_analysis.get('fp_resected_count', 0),
                            'FP_Resected_Ratio': fp_analysis.get('fp_resected_ratio', 0.0),
                            'Has_Resection_Data': 1 if result['resection_indices'] else 0
                        })
                    
                    if fp_resected_data:
                        fp_resected_df = pd.DataFrame(fp_resected_data)
                        fp_resected_df.to_excel(writer, sheet_name='FP_Resected_Analysis', index=False)
                
                # Sheet 7: Feature Importance Summary
                if results:
                    all_time_importance = []
                    all_metric_importance = []
                    all_eigenvalue_importance = []
                    
                    feature_importance_details = []
                    
                    for result_key, result in results.items():
                        feature_analysis = result.get('feature_analysis', {})
                        
                        if feature_analysis:
                            all_time_importance.append(feature_analysis['time_importance'])
                            all_metric_importance.append(feature_analysis['metric_importance'])
                            all_eigenvalue_importance.append(feature_analysis['eigenvalue_importance'])
                            
                            # Record most important feature combination for each subject
                            most_important = feature_analysis['most_important_combination']
                            feature_importance_details.append({
                                'Fold': result.get('fold', ''),
                                'Subject': result.get('subject', ''),
                                'Time_Bins': result.get('time_bins', self.time_bins),
                                'AUC': result['subject_metrics']['auc'],
                                'Most_Important_Time_Window': most_important['time_window'],
                                'Most_Important_Eigenvalue': most_important['eigenvalue'], 
                                'Most_Important_Metric': most_important['metric'],
                                'Max_Importance_Value': most_important['importance_value']
                            })
                    
                    # Calculate average feature importance
                    if all_time_importance:
                        avg_time_importance = np.mean(all_time_importance, axis=0)
                        avg_metric_importance = np.mean(all_metric_importance, axis=0)
                        avg_eigenvalue_importance = np.mean(all_eigenvalue_importance, axis=0)
                        
                        # Save average feature importance
                        time_importance_data = [
                            {'Dimension': 'Time_Window', 'Index': i, 'Average_Importance': importance,
                             'Time_Bins': self.time_bins}
                            for i, importance in enumerate(avg_time_importance)
                        ]
                        
                        metric_names = ['Metric_0', 'Metric_1', 'Metric_2', 'Metric_3']
                        metric_importance_data = [
                            {'Dimension': 'Graph_Metric', 'Index': metric_names[i], 'Average_Importance': importance,
                             'Time_Bins': self.time_bins}
                            for i, importance in enumerate(avg_metric_importance)
                        ]
                        
                        eigenvalue_importance_data = [
                            {'Dimension': 'Eigenvalue', 'Index': i, 'Average_Importance': importance,
                             'Time_Bins': self.time_bins}
                            for i, importance in enumerate(avg_eigenvalue_importance)
                        ]
                        
                        # Combine all importance data
                        all_importance_data = time_importance_data + metric_importance_data + eigenvalue_importance_data
                        feature_summary_df = pd.DataFrame(all_importance_data)
                        feature_summary_df.to_excel(writer, sheet_name='Feature_Importance_Summary', index=False)
                    
                    # Save feature importance details for each subject
                    if feature_importance_details:
                        feature_details_df = pd.DataFrame(feature_importance_details)
                        feature_details_df.to_excel(writer, sheet_name='Feature_Importance_Details', index=False)
            
            print(f"10-fold CV results saved to: {save_path}")
            
        except Exception as e:
            print(f"Failed to save results to Excel: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_comprehensive_analysis(self, excel_path):
        """Run 10-fold cross-validation analysis on all Engel subjects"""
        print(f"\n10-fold Cross-Validation Analysis - All Engel Classes")
        print("="*60)
        print(f"Time bins: {self.time_bins}")
        print(f"Feature dimensions: {self.time_bins} × 10 × 4 = {self.time_bins * 10 * 4}")
        
        if self.use_hyperparameter_search:
            print(f"Hyperparameter search enabled:")
            print(f"   - Inner CV folds: {self.cv_folds}")
            print(f"   - Search iterations: {self.n_iter_search}")
            print(f"   - Scoring metric: {self.search_cv_scoring}")
        else:
            print("Using default hyperparameters (search disabled)")
        
        print(f"SMOTE: {'Enabled' if self.use_smote else 'Disabled'}")
        
        # Run 10-fold cross-validation (Method 1 only)
        print(f"\nRunning Method 1: {self.time_bins} Temporal Bins Aggregation with 10-Fold CV")
        results, subject_metrics, run_metrics, fold_metrics, hyperparameter_log = self.ten_fold_cross_validation('method1', None)
        
        # Print summary statistics
        if subject_metrics:
            aucs = [m['auc'] for m in subject_metrics]
            accs = [m['accuracy'] for m in subject_metrics]
            sens = [m['sensitivity'] for m in subject_metrics]
            specs = [m['specificity'] for m in subject_metrics]
            
            print(f"\n10-Fold CV Results Summary:")
            print(f"   - Time bins: {self.time_bins}")
            print(f"   - Total subjects: {len(subject_metrics)}")
            print(f"   - Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
            print(f"   - Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"   - Mean Sensitivity: {np.mean(sens):.4f} ± {np.std(sens):.4f}")
            print(f"   - Mean Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}")
            print(f"   - AUC Range: [{np.min(aucs):.4f}, {np.max(aucs):.4f}]")
        
        if fold_metrics:
            fold_aucs = [f['mean_auc'] for f in fold_metrics]
            print(f"\nFold-level Performance:")
            print(f"   - Folds completed: {len(fold_metrics)}")
            print(f"   - Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
            print(f"   - Best fold AUC: {np.max(fold_aucs):.4f}")
            print(f"   - Worst fold AUC: {np.min(fold_aucs):.4f}")
        
        # Save results to Excel
        if results:
            self.save_results_to_excel(
                results, subject_metrics, run_metrics, fold_metrics, hyperparameter_log,
                excel_path
            )
        
        print(f"\n10-fold CV analysis completed and saved to Excel!")
        
        return results, subject_metrics, fold_metrics

def main():
    """Main function with configurable sampling method and time bins"""
    print("\n10-Fold Cross-Validation SOZ Analysis System")
    print("Features: Configurable Sampling Methods, All Engel Classes, Subject-level CV, Configurable Time Bins")
    print("="*80)
    
    # Data path configuration
    data_path = r'\\10.20.37.22\dataset0\zhichao\codes\graph_koopman_epilepsy\result\improved_graph_allweight_4metrics_results' 
    # data_path = r'\\10.20.37.22\dataset0\zhichao\codes\graph_koopman_epilepsy\result\fc_direct_graph_metrics'  # 原始FC 
    soz_path = r'\\10.20.37.22\dataset0\zhichao\codes\graph_koopman_epilepsy\result\all_soz_index_80-160.pkl'
    participants_file = r'D:\project\graph_koopman\participants.xlsx'
    resection_path = r'\\10.20.37.22\dataset0\zhichao\codes\graph_koopman_epilepsy\result\adopt_subject_resected_index.pkl'
    
    ##### Configuration parameters - easy to modify for experiments
    TIME_BINS = 6  # Number of temporal bins (try 8, 9, 10, 12, 15, 20, etc.)
    SAMPLING_METHOD = 'undersample'  # Options: 'smote', 'undersample', 'none'
    
    # Update excel filename based on configuration
    base_output_dir = r'\\10.20.37.22\dataset0\zhichao\codes\graph_koopman_epilepsy\result'
    
    # 根据配置更新Excel文件名，使用正确的路径拼接
    if SAMPLING_METHOD == 'smote':
        filename = f'SOZ_10Fold_CV_AllEngel_SMOTE_{TIME_BINS}bins.xlsx'
    elif SAMPLING_METHOD == 'undersample':
        filename = f'SOZ_10Fold_CV_AllEngel_randomsample_{TIME_BINS}bins.xlsx'
    else:
        filename = f'SOZ_10Fold_CV_AllEngel_NoSampling_{TIME_BINS}bins.xlsx'
    
    # 使用os.path.join进行正确的路径构造
    excel_path = os.path.join(base_output_dir, filename)
    
    # Create framework instance with configurable time bins
    framework = EnhancedSOZFramework(
        data_path=data_path,
        soz_path=soz_path,
        participants_file=participants_file,
        resection_path=resection_path,
        time_bins=TIME_BINS  # Pass time_bins parameter
    )
    
    # Configure other settings
    framework.n_folds = 10  # Number of CV folds
    framework.sampling_method = SAMPLING_METHOD  # Set sampling method
    framework.use_smote = (SAMPLING_METHOD == 'smote')  # For backward compatibility
    framework.use_hyperparameter_search = True  # Enable hyperparameter search
    framework.n_iter_search = 20  # Number of hyperparameter combinations to try
    framework.cv_folds = 5  # Inner CV folds for hyperparameter search
    framework.search_cv_scoring = 'roc_auc'  # Scoring metric for hyperparameter search
    
    print(f"\nConfiguration:")
    print(f"  - Time bins: {TIME_BINS}")
    print(f"  - Feature dimensions: {TIME_BINS} × 10 × 4 = {TIME_BINS * 10 * 4}")
    print(f"  - Sampling method: {SAMPLING_METHOD}")
    print(f"  - Output file: {excel_path}")
    
    # Load data
    success = framework.load_data()
    if not success or not framework.subjects_data or not framework.soz_indices:
        print("Data loading failed, terminating")
        return
    
    # Run comprehensive analysis
    try:
        results, subject_metrics, fold_metrics = framework.run_comprehensive_analysis(excel_path)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {excel_path}")
        
    except Exception as e:
        print(f"\nAnalysis failed with error:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()