"""
Upcoding Labeling and Class Balancing Module
Handles labeling of upcoding cases and addresses class imbalance issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BalancingConfig:
    """Configuration for class balancing strategies"""
    method: str = 'smote'  # 'smote', 'adasyn', 'borderline_smote', 'combined', 'undersampling', 'class_weights'
    sampling_strategy: str = 'auto'  # 'auto', 'minority', or specific ratios
    random_state: int = 42
    k_neighbors: int = 5
    
    # SMOTE-specific parameters
    smote_kind: str = 'regular'  # 'regular', 'borderline1', 'borderline2', 'svm'
    
    # Combined method parameters
    combine_method: str = 'smoteenn'  # 'smoteenn', 'smotetomek'


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class (between 0 and 1)
            gamma: Focusing parameter for hard examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss
        
        Args:
            inputs: Predicted logits [N, C] where N is batch size, C is number of classes
            targets: Ground truth labels [N]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if inputs.device != targets.device:
                targets = targets.to(inputs.device)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class UpcodingLabelingBalancer:
    """
    Main class for handling upcoding labeling and class balancing
    """
    
    def __init__(self, config: BalancingConfig = None):
        """
        Initialize the labeling and balancing module
        
        Args:
            config: Configuration for balancing strategies
        """
        self.config = config or BalancingConfig()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.class_weights = None
        self.original_distribution = None
        self.balanced_distribution = None
        
        logger.info(f"Initialized with balancing method: {self.config.method}")
    
    def create_comprehensive_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive labels for upcoding detection
        Flags all claim lines when upcoding is detected at claim level
        
        Args:
            df: DataFrame with claims data
            
        Returns:
            DataFrame with comprehensive labeling
        """
        logger.info("Creating comprehensive upcoding labels...")
        
        # Create a copy to avoid modifying original data
        labeled_df = df.copy()
        
        
        # Step 1: Create final binary label (1 for upcoded, 0 otherwise)
        labeled_df['upcoding_label'] = labeled_df['is_upcoded'].fillna(False).astype(int)
        
        # Step 2: Create multi-class labels for upcoding types
        upcoding_types = labeled_df['upcoding_type'].fillna('None').unique()
        labeled_df['upcoding_type_encoded'] = self.label_encoder.fit_transform(
            labeled_df['upcoding_type'].fillna('None')
        )
        
        # Step 3: Add additional features for model training
        labeled_df = self._add_risk_features(labeled_df)

        logger.info(f"Labeling completed:")
        logger.info(f"  - Total samples: {len(labeled_df)}")
        logger.info(f"  - Upcoded samples: {labeled_df['upcoding_label'].sum()}")
        logger.info(f"  - Non-upcoded samples: {(labeled_df['upcoding_label'] == 0).sum()}")
        logger.info(f"  - Upcoding types found: {list(upcoding_types)}")
        
        # Store original distribution
        self.original_distribution = Counter(labeled_df['upcoding_label'])
        
        return labeled_df
    
    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional risk-based features for upcoding detection
        
        Args:
            df: DataFrame with labeled data
            
        Returns:
            DataFrame with additional features
        """
        df['total_charges'] = pd.to_numeric(df['total_charges'])
        # Calculate charge-based risk indicators
        df['charge_variance'] = df.groupby('claim_id')['total_charges'].transform('std').fillna(0)
        df['charge_mean'] = df.groupby('claim_id')['total_charges'].transform('mean').fillna(0)
       
        # Provider-based risk indicators
        provider_upcoding_rate = df.groupby('provider_id')['upcoding_label'].transform('mean').fillna(0)
        df['provider_upcoding_history'] = provider_upcoding_rate
         
        # Specialty-based risk indicators
        specialty_upcoding_rate = df.groupby('provider_specialty')['upcoding_label'].transform('mean').fillna(0)
        df['specialty_upcoding_history'] = specialty_upcoding_rate
        
        # Procedure complexity indicators
        df['high_value_procedure'] = (df['total_charges'] > df['total_charges'].quantile(0.9)).astype(int)
        df['multiple_procedures'] = df.groupby('claim_id')['line_number'].transform('count')
        
        return df
    
    def analyze_class_distribution(self, df: pd.DataFrame, label_col: str = 'upcoding_label') -> Dict[str, Any]:
        """
        Analyze class distribution and imbalance
        
        Args:
            df: DataFrame with labels
            label_col: Column name for labels
            
        Returns:
            Dictionary with distribution statistics
        """
        distribution = Counter(df[label_col])
        total_samples = len(df)
        
        analysis = {
            'distribution': dict(distribution),
            'total_samples': total_samples,
            'minority_class_ratio': min(distribution.values()) / total_samples,
            'majority_class_ratio': max(distribution.values()) / total_samples,
            'imbalance_ratio': max(distribution.values()) / min(distribution.values()),
            'is_imbalanced': min(distribution.values()) / total_samples < 0.1
        }
        
        logger.info(f"Class Distribution Analysis:")
        logger.info(f"  - Total samples: {analysis['total_samples']}")
        logger.info(f"  - Distribution: {analysis['distribution']}")
        logger.info(f"  - Imbalance ratio: {analysis['imbalance_ratio']:.2f}")
        logger.info(f"  - Minority class ratio: {analysis['minority_class_ratio']:.4f}")
        
        return analysis
    
    def apply_smote_balancing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique)
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Balanced X and y arrays
        """
        logger.info(f"Applying SMOTE balancing...")
        
        # Choose SMOTE variant
        if self.config.smote_kind == 'borderline1':
            sampler = BorderlineSMOTE(
                sampling_strategy=self.config.sampling_strategy,
                random_state=self.config.random_state,
                k_neighbors=self.config.k_neighbors,
                kind='borderline-1'
            )
        elif self.config.smote_kind == 'borderline2':
            sampler = BorderlineSMOTE(
                sampling_strategy=self.config.sampling_strategy,
                random_state=self.config.random_state,
                k_neighbors=self.config.k_neighbors,
                kind='borderline-2'
            )
        else:
            sampler = SMOTE(
                sampling_strategy=self.config.sampling_strategy,
                random_state=self.config.random_state,
                k_neighbors=self.config.k_neighbors
            )
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        logger.info(f"SMOTE completed:")
        logger.info(f"  - Original samples: {len(X)}")
        logger.info(f"  - Balanced samples: {len(X_balanced)}")
        logger.info(f"  - New distribution: {Counter(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def apply_adasyn_balancing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ADASYN (Adaptive Synthetic Sampling)
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Balanced X and y arrays
        """
        logger.info(f"Applying ADASYN balancing...")
        
        sampler = ADASYN(
            sampling_strategy=self.config.sampling_strategy,
            random_state=self.config.random_state,
            n_neighbors=self.config.k_neighbors
        )
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        logger.info(f"ADASYN completed:")
        logger.info(f"  - Original samples: {len(X)}")
        logger.info(f"  - Balanced samples: {len(X_balanced)}")
        logger.info(f"  - New distribution: {Counter(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def apply_combined_balancing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply combined over/under sampling techniques
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Balanced X and y arrays
        """
        logger.info(f"Applying combined balancing ({self.config.combine_method})...")
        
        if self.config.combine_method == 'smoteenn':
            sampler = SMOTEENN(
                sampling_strategy=self.config.sampling_strategy,
                random_state=self.config.random_state
            )
        else:  # smotetomek
            sampler = SMOTETomek(
                sampling_strategy=self.config.sampling_strategy,
                random_state=self.config.random_state
            )
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        logger.info(f"Combined balancing completed:")
        logger.info(f"  - Original samples: {len(X)}")
        logger.info(f"  - Balanced samples: {len(X_balanced)}")
        logger.info(f"  - New distribution: {Counter(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for handling imbalance
        
        Args:
            y: Label array
            
        Returns:
            Dictionary mapping class to weight
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        class_weights = {cls: weight for cls, weight in zip(classes, weights)}
        
        logger.info(f"Computed class weights: {class_weights}")
        
        self.class_weights = class_weights
        return class_weights
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray, 
                       method: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply specified balancing method to dataset
        
        Args:
            X: Feature array
            y: Label array
            method: Balancing method to use
            
        Returns:
            Balanced X and y arrays
        """
        method = method or self.config.method
        
        logger.info(f"Balancing dataset using method: {method}")
        
        if method == 'smote':
            return self.apply_smote_balancing(X, y)
        elif method == 'adasyn':
            return self.apply_adasyn_balancing(X, y)
        elif method == 'combined':
            return self.apply_combined_balancing(X, y)
        elif method == 'class_weights':
            # For class weights, we don't change the data
            self.compute_class_weights(y)
            return X, y
        else:
            logger.warning(f"Unknown balancing method: {method}. Returning original data.")
            return X, y
    
    def prepare_training_data(self, df: pd.DataFrame, 
                            text_column: str = 'input_text',
                            test_size: float = 0.2,
                            val_size: float = 0.1,
                            balance_method: str = None) -> Dict[str, Any]:
        """
        Prepare complete training dataset with labels and balancing
        
        Args:
            df: DataFrame with prepared data
            text_column: Column containing text data
            test_size: Proportion of test set
            val_size: Proportion of validation set
            balance_method: Method for handling class imbalance
            
        Returns:
            Dictionary containing train/val/test splits and metadata
        """
        logger.info("Preparing training data with labels and balancing...")
        
        # Create comprehensive labels
        labeled_df = self.create_comprehensive_labels(df)
        
        # Analyze class distribution
        class_analysis = self.analyze_class_distribution(labeled_df)

        # Prepare features and labels
        X_text = labeled_df[text_column].values
        y = labeled_df['upcoding_label'].values
        
        # Select numerical features for traditional ML models
        numerical_features = [
            'units', 'total_charges', 'allowed_amount', 'charge_ratio',
            'charge_variance', 'charge_mean', 'provider_upcoding_history',
            'specialty_upcoding_history', 'high_value_procedure', 'multiple_procedures'
        ]
        
        # Filter available numerical features
        available_numerical_features = [col for col in numerical_features if col in labeled_df.columns]
        
        if available_numerical_features:
            X_numerical = labeled_df[available_numerical_features].fillna(0).values
            X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        else:
            X_numerical_scaled = np.array([]).reshape(len(labeled_df), 0)
        
        # Split data
        # First split: train+val vs test
        X_text_temp, X_text_test, X_num_temp, X_num_test, y_temp, y_test = train_test_split(
            X_text, X_numerical_scaled, y, 
            test_size=test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
            X_text_temp, X_num_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        # Apply balancing to training set only
        if balance_method or self.config.method != 'class_weights':
            # For BERT models, we only balance numerical features
            if X_num_train.shape[1] > 0:
                X_num_train_balanced, y_train_balanced = self.balance_dataset(
                    X_num_train, y_train, balance_method
                )
                
                # For text data, we need to duplicate/generate corresponding text
                # This is a simplified approach - in practice, you might want more sophisticated text augmentation
                if len(y_train_balanced) != len(y_train):
                    # Create index mapping for text data
                    indices = np.arange(len(y_train))
                    balanced_indices = self._get_balanced_indices(indices, y_train, y_train_balanced)
                    X_text_train_balanced = X_text_train[balanced_indices]
                else:
                    X_text_train_balanced = X_text_train
                    X_num_train_balanced, y_train_balanced = X_num_train, y_train
            else:
                # If no numerical features, just use class weights
                X_text_train_balanced = X_text_train
                X_num_train_balanced = X_num_train
                y_train_balanced = y_train
                self.compute_class_weights(y_train)
        else:
            X_text_train_balanced = X_text_train
            X_num_train_balanced = X_num_train
            y_train_balanced = y_train
            self.compute_class_weights(y_train)
        
        # Store balanced distribution
        self.balanced_distribution = Counter(y_train_balanced)
        
        # Create proper CSV files for BioBERT model
        train_csv = pd.DataFrame({
            'text': X_text_train_balanced,
            'label': y_train_balanced
        })
        train_csv.to_csv(os.path.join('Data','train_data.csv'), index=False)
        
        val_csv = pd.DataFrame({
            'text': X_text_val,
            'label': y_val
        })
        val_csv.to_csv(os.path.join('Data','val_data.csv'), index=False)
        
        test_csv = pd.DataFrame({
            'text': X_text_test,
            'label': y_test
        })
        test_csv.to_csv(os.path.join('Data','test_data.csv'), index=False)
        
        # Prepare result dictionary
        result = {
            'train': {
                'text': X_text_train_balanced,
                'numerical': X_num_train_balanced,
                'labels': y_train_balanced
            },
            'val': {
                'text': X_text_val,
                'numerical': X_num_val,
                'labels': y_val
            },
            'test': {
                'text': X_text_test,
                'numerical': X_num_test,
                'labels': y_test
            },
            'metadata': {
                'class_weights': self.class_weights,
                'original_distribution': self.original_distribution,
                'balanced_distribution': self.balanced_distribution,
                'class_analysis': class_analysis,
                'numerical_features': available_numerical_features,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler
            }
        }
        
        logger.info("Training data preparation completed:")
        logger.info("CSV files generated for BioBERT model:")
        logger.info(f"  - train_data.csv: {len(y_train_balanced)} samples")
        logger.info(f"  - val_data.csv: {len(y_val)} samples") 
        logger.info(f"  - test_data.csv: {len(y_test)} samples")
        logger.info(f"  - Train samples: {len(y_train_balanced)} (balanced)")
        logger.info(f"  - Val samples: {len(y_val)}")
        logger.info(f"  - Test samples: {len(y_test)}")
        logger.info(f"  - Balanced distribution: {self.balanced_distribution}")
        
        return result
    
    def _get_balanced_indices(self, indices: np.ndarray, y_original: np.ndarray, 
                            y_balanced: np.ndarray) -> np.ndarray:
        """
        Get indices for text data corresponding to balanced numerical data
        This is a simplified implementation for demonstration
        """
        # For SMOTE, we need to handle synthetic samples
        # This is a basic implementation - in practice, you might use text augmentation
        
        if len(y_balanced) <= len(y_original):
            return indices[:len(y_balanced)]
        
        # If more samples needed, duplicate some indices
        additional_needed = len(y_balanced) - len(y_original)
        minority_class = 1 if Counter(y_original)[1] < Counter(y_original)[0] else 0
        minority_indices = indices[y_original == minority_class]
        
        # Randomly duplicate minority class indices
        np.random.seed(self.config.random_state)
        additional_indices = np.random.choice(minority_indices, additional_needed, replace=True)
        
        return np.concatenate([indices, additional_indices])
    
    def create_focal_loss(self, alpha: float = 0.25, gamma: float = 2.0) -> FocalLoss:
        """
        Create Focal Loss instance for training
        
        Args:
            alpha: Weighting factor for minority class
            gamma: Focusing parameter
            
        Returns:
            FocalLoss instance
        """
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    def visualize_distribution(self, save_path: str = None):
        """
        Visualize class distributions before and after balancing
        
        Args:
            save_path: Path to save the plot
        """
        if not self.original_distribution or not self.balanced_distribution:
            logger.warning("No distribution data available for visualization")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original distribution
        classes = list(self.original_distribution.keys())
        counts_orig = list(self.original_distribution.values())
        ax1.bar(classes, counts_orig, color=['lightcoral', 'lightblue'])
        ax1.set_title('Original Class Distribution')
        ax1.set_xlabel('Class (0: Normal, 1: Upcoded)')
        ax1.set_ylabel('Count')
        for i, count in enumerate(counts_orig):
            ax1.text(i, count + max(counts_orig) * 0.01, str(count), ha='center')
        
        # Balanced distribution
        classes_bal = list(self.balanced_distribution.keys())
        counts_bal = list(self.balanced_distribution.values())
        ax2.bar(classes_bal, counts_bal, color=['lightcoral', 'lightblue'])
        ax2.set_title('Balanced Class Distribution')
        ax2.set_xlabel('Class (0: Normal, 1: Upcoded)')
        ax2.set_ylabel('Count')
        for i, count in enumerate(counts_bal):
            ax2.text(i, count + max(counts_bal) * 0.01, str(count), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to: {save_path}")
        
        plt.show()


def main():
    """
    Example usage of the Upcoding Labeling and Balancing module
    """
    # Configuration for balancing
    config = BalancingConfig(
        method='smote',
        sampling_strategy='auto',
        random_state=42,
        k_neighbors=5
    )
    
    # Initialize the balancer
    balancer = UpcodingLabelingBalancer(config)
    
    try:
        # read output from your data preparation module)
        csv_path = os.path.join('Data','prepared_claims_data.csv')
        df = pd.read_csv(
            csv_path,
            dtype=str,  # Read all as strings initially for better control
            skip_blank_lines=True,
            na_values=['', 'NULL', 'null', 'N/A', 'n/a'],
            keep_default_na=False
        )
        
        logger.info(f"Successfully loaded CSV with {len(df)} records from {csv_path}")
        
        # Prepare training data with balancing
        training_data = balancer.prepare_training_data(
            df, 
            balance_method='smote',
            test_size=0.2,
            val_size=0.1
        )
        
        print("\n=== Training Data Preparation Results ===")
        print(f"Original distribution: {training_data['metadata']['original_distribution']}")
        print(f"Balanced distribution: {training_data['metadata']['balanced_distribution']}")
        print(f"Class weights: {training_data['metadata']['class_weights']}")
        print(f"Imbalance ratio: {training_data['metadata']['class_analysis']['imbalance_ratio']:.2f}")
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(training_data['train']['labels'])}")
        print(f"  Validation: {len(training_data['val']['labels'])}")
        print(f"  Test: {len(training_data['test']['labels'])}")
        
        print(f"\nCSV files generated:")
        print(f"  - train_data.csv: Ready for BioBERT training")
        print(f"  - val_data.csv: Ready for BioBERT validation")
        print(f"  - test_data.csv: Ready for BioBERT testing")
        
        # Create Focal Loss for training
        focal_loss = balancer.create_focal_loss(alpha=0.25, gamma=2.0)
        print(f"\nFocal Loss created with alpha=0.25, gamma=2.0")
        
        # Visualize distributions
        balancer.visualize_distribution()
        
        print("\nLabeling and balancing completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")


if __name__ == "__main__":
    main()