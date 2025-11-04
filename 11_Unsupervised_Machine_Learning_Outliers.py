"""
Healthcare Payment Integrity - Unsupervised Outlier Detection
Modular framework for detecting anomalies using Isolation Forest, LOF, and Autoencoders
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

TF_ENABLE_ONEDNN_OPTS=0

class DataLoader:
    """Module for loading and preprocessing data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def load_data(self, query: str = None) -> pd.DataFrame:
        """Load data from SQLite database"""
        if query is None:
            query = "SELECT * FROM claims_data WHERE allowed_amount > 1000"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df


class TimeSeriesFeatureEngine:
    """Module for creating time-series features for providers"""
    
    def __init__(self):
        self.provider_baselines = {}
        
    def create_provider_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create per-provider time series features"""
        df = df.copy()
        
        # Ensure date columns are datetime
        if 'service_date_header' in df.columns:
            df['service_date_header'] = pd.to_datetime(df['service_date_header'])
        if 'service_date_line' in df.columns:
            df['service_date_line'] = pd.to_datetime(df['service_date_line'])
        
        # Use service_date_line as primary, fallback to header
        df['service_date'] = df['service_date_line'].fillna(df['service_date_header'])
        
        # Create time windows
        df['year_month'] = df['service_date'].dt.to_period('M')
        df['year_week'] = df['service_date'].dt.to_period('W')
        df['year_quarter'] = df['service_date'].dt.to_period('Q')
        print("Done create_provider_time_series ...")
        
        return df
    
    def aggregate_provider_metrics(self, df: pd.DataFrame, 
                                   time_period: str = 'M') -> pd.DataFrame:
        """
        Aggregate provider metrics over time periods
        time_period: 'M' (monthly), 'W' (weekly), 'Q' (quarterly)
        """
        period_col = {
            'M': 'year_month',
            'W': 'year_week',
            'Q': 'year_quarter'
        }[time_period]
        
        # Aggregate by provider and time period
        agg_metrics = df.groupby(['provider_id', period_col]).agg({
            'claim_id': 'count',  # Total claims
            'total_charges_line': ['sum', 'mean', 'std', 'max'],
            'allowed_amount': ['sum', 'mean'],
            'units': ['sum', 'mean', 'max'],
            'expected_payment': ['sum', 'mean'],
            'is_upcoded_line': ['sum', 'mean']  # For validation
        }).reset_index()
        
        # Flatten column names
        agg_metrics.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in agg_metrics.columns.values]
        
        # Rename for clarity
        rename_dict = {
            'claim_id_count': 'total_claims',
            'total_charges_line_sum': 'total_charges',
            'total_charges_line_mean': 'avg_charge_per_claim',
            'total_charges_line_std': 'charge_volatility',
            'total_charges_line_max': 'max_charge',
            'allowed_amount_sum': 'total_allowed',
            'allowed_amount_mean': 'avg_allowed',
            'units_sum': 'total_units',
            'units_mean': 'avg_units',
            'units_max': 'max_units',
            'expected_payment_sum': 'total_payment',
            'expected_payment_mean': 'avg_payment',
            'is_upcoded_line_sum': 'upcoded_count',
            'is_upcoded_line_mean': 'upcoding_rate'
        }
        agg_metrics = agg_metrics.rename(columns=rename_dict)
        agg_metrics = agg_metrics[agg_metrics.total_claims > 5]
        print("Done aggregate_provider_metrics: ", )
        return agg_metrics
    
    def calculate_deviations(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate deviations from provider's historical baseline"""
        agg_df = agg_df.copy()
        
        # Calculate rolling statistics per provider
        for provider in agg_df['provider_id'].unique():
            provider_mask = agg_df['provider_id'] == provider
            provider_data = agg_df[provider_mask].sort_values(
                agg_df.columns[1]  # Sort by time period column
            )
            
            # Rolling mean and std (3-period window)
            for col in ['total_claims', 'total_charges', 'total_units', 'avg_charge_per_claim']:
                if col in provider_data.columns:
                    rolling_mean = provider_data[col].rolling(window=3, min_periods=1).mean()
                    rolling_std = provider_data[col].rolling(window=3, min_periods=1).std()
                    
                    # Z-score deviation
                    agg_df.loc[provider_mask, f'{col}_zscore'] = (
                        (provider_data[col] - rolling_mean) / (rolling_std + 1e-10)
                    )
                    
                    # Percentage change
                    agg_df.loc[provider_mask, f'{col}_pct_change'] = (
                        provider_data[col].pct_change() * 100
                    )
        
        # Fill NaN values
        agg_df = agg_df.fillna(0)
        
        return agg_df


class IsolationForestDetector:
    """Isolation Forest for detecting volume/charge spikes"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = RobustScaler()
        
    def fit_predict(self, X: pd.DataFrame, 
                   feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit Isolation Forest and predict anomalies
        Returns: (anomaly_labels, anomaly_scores)
        """
        if feature_cols is None:
            # Use numeric columns
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID and period columns
            feature_cols = [col for col in feature_cols 
                          if not any(x in col.lower() for x in ['id', 'period'])]
        
        X_features = X[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        # Predict: -1 for anomalies, 1 for normal
        predictions = self.model.fit_predict(X_scaled)
        anomaly_labels = (predictions == -1).astype(int)
        
        # Get anomaly scores (lower = more anomalous)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        return anomaly_labels, anomaly_scores


class LOFDetector:
    """Local Outlier Factor for density-based anomaly detection"""
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.scaler = RobustScaler()
        
    def fit_predict(self, X: pd.DataFrame, 
                   feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit LOF and predict anomalies
        Returns: (anomaly_labels, negative_outlier_factors)
        """
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols 
                          if not any(x in col.lower() for x in ['id', 'period'])]
        
        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit LOF
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1
        )
        
        # Predict
        predictions = self.model.fit_predict(X_scaled)
        anomaly_labels = (predictions == -1).astype(int)
        
        # Get negative outlier factors (more negative = more anomalous)
        negative_outlier_factors = self.model.negative_outlier_factor_
        
        return anomaly_labels, negative_outlier_factors


class AutoencoderDetector:
    """Autoencoder for reconstruction-based anomaly detection"""
    
    def __init__(self, encoding_dim: int = 16, contamination: float = 0.1, 
                 random_state: int = 42):
        self.encoding_dim = encoding_dim
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def build_autoencoder(self, input_dim: int) -> Model:
        """Build autoencoder architecture"""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def fit(self, X: pd.DataFrame, feature_cols: List[str] = None, 
            epochs: int = 50, batch_size: int = 256, validation_split: float = 0.2):
        """Train autoencoder on normal data"""
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols 
                          if not any(x in col.lower() for x in ['id', 'period', 'upcod'])]
        
        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Build model
        self.model = self.build_autoencoder(X_scaled.shape[1])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Calculate reconstruction errors on training data
        reconstructions = self.model.predict(X_scaled, verbose=0)
        train_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Set threshold based on contamination percentage
        self.threshold = np.percentile(train_errors, 100 * (1 - self.contamination))
        
        return history
    
    def predict(self, X: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on reconstruction error
        Returns: (anomaly_labels, reconstruction_errors)
        """
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols 
                          if not any(x in col.lower() for x in ['id', 'period', 'upcod'])]
        
        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X_features)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Anomaly labels based on threshold
        anomaly_labels = (reconstruction_errors > self.threshold).astype(int)
        
        return anomaly_labels, reconstruction_errors


class AnomalyAnalyzer:
    """Module for analyzing and combining anomaly detection results"""
    
    def __init__(self):
        self.results = {}
        
    def combine_predictions(self, predictions_dict: Dict[str, np.ndarray],
                          method: str = 'voting') -> np.ndarray:
        """
        Combine predictions from multiple models
        method: 'voting' (majority vote), 'union' (any), 'intersection' (all)
        """
        predictions = np.array(list(predictions_dict.values()))
        
        if method == 'voting':
            # Majority vote
            combined = (predictions.sum(axis=0) >= len(predictions) / 2).astype(int)
        elif method == 'union':
            # Any model flags as anomaly
            combined = (predictions.sum(axis=0) > 0).astype(int)
        elif method == 'intersection':
            # All models flag as anomaly
            combined = (predictions.sum(axis=0) == len(predictions)).astype(int)
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return combined
    
    def calculate_anomaly_scores(self, scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine and normalize anomaly scores from different models"""
        # Normalize each score to [0, 1]
        normalized_scores = []
        
        for name, scores in scores_dict.items():
            # Convert to anomaly scores (higher = more anomalous)
            if name in ['isolation_forest', 'lof']:
                # These are already anomaly scores (lower = more anomalous)
                scores = -scores
            
            # Min-max normalization
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            normalized_scores.append(scores_norm)
        
        # Average normalized scores
        combined_score = np.mean(normalized_scores, axis=0)
        
        return combined_score


class ResultsExporter:
    """Module for exporting and visualizing anomaly detection results"""
    
    def __init__(self, output_dir: str = 'outputs/anomaly_detection'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_anomalies(self, df: pd.DataFrame, 
                        predictions_dict: Dict[str, np.ndarray],
                        scores_dict: Dict[str, np.ndarray],
                        filename: str = 'anomaly_results.csv') -> str:
        """Export anomaly detection results to CSV"""
        filepath = os.path.join(self.output_dir, filename)
        
        export_df = df.copy()
        
        # Add predictions
        for model_name, predictions in predictions_dict.items():
            export_df[f'{model_name}_anomaly'] = predictions
        
        # Add scores
        for model_name, scores in scores_dict.items():
            export_df[f'{model_name}_score'] = scores
        
        # Add combined prediction
        analyzer = AnomalyAnalyzer()
        export_df['combined_anomaly'] = analyzer.combine_predictions(predictions_dict, method='voting')
        export_df['combined_score'] = analyzer.calculate_anomaly_scores(scores_dict)
        
        export_df.to_csv(filepath, index=False)
        print(f"\nAnomaly results exported to: {filepath}")
        return filepath
    
    def export_summary_stats(self, df: pd.DataFrame,
                           predictions_dict: Dict[str, np.ndarray],
                           filename: str = 'anomaly_summary.csv') -> str:
        """Export summary statistics of anomalies"""
        filepath = os.path.join(self.output_dir, filename)
        
        summary_list = []
        for model_name, predictions in predictions_dict.items():
            n_anomalies = predictions.sum()
            anomaly_rate = (n_anomalies / len(predictions)) * 100
            
            # Calculate statistics for anomalous records
            anomaly_mask = predictions == 1
            if anomaly_mask.sum() > 0:
                if 'total_charges' in df.columns:
                    avg_anomaly_charge = df.loc[anomaly_mask, 'total_charges'].mean()
                    avg_normal_charge = df.loc[~anomaly_mask, 'total_charges'].mean()
                else:
                    avg_anomaly_charge = np.nan
                    avg_normal_charge = np.nan
                
                summary_list.append({
                    'Model': model_name,
                    'Total_Records': len(predictions),
                    'Anomalies_Detected': n_anomalies,
                    'Anomaly_Rate_%': anomaly_rate,
                    'Avg_Anomaly_Charge': avg_anomaly_charge,
                    'Avg_Normal_Charge': avg_normal_charge,
                    'Charge_Ratio': avg_anomaly_charge / avg_normal_charge if avg_normal_charge > 0 else np.nan
                })
        
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(filepath, index=False)
        print(f"Summary statistics exported to: {filepath}")
        return filepath
    
    def plot_anomaly_distribution(self, df: pd.DataFrame, 
                                  predictions_dict: Dict[str, np.ndarray]):
        """Plot distribution of anomalies across models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Plot 1: Anomaly counts per model
        model_names = list(predictions_dict.keys())
        anomaly_counts = [pred.sum() for pred in predictions_dict.values()]
        
        axes[0].bar(model_names, anomaly_counts, color='coral', alpha=0.7)
        axes[0].set_title('Anomalies Detected by Each Model', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Anomalies')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(alpha=0.3, axis='y')
        
        # Plot 2: Venn diagram representation
        from matplotlib_venn import venn2, venn3
        if len(predictions_dict) == 2:
            sets = [set(np.where(pred == 1)[0]) for pred in predictions_dict.values()]
            venn2(sets, set_labels=model_names, ax=axes[1])
        elif len(predictions_dict) == 3:
            sets = [set(np.where(pred == 1)[0]) for pred in predictions_dict.values()]
            venn3(sets, set_labels=model_names, ax=axes[1])
        axes[1].set_title('Anomaly Overlap Between Models', fontsize=12, fontweight='bold')
        
        # Plot 3: Time series of anomalies (if time column exists)
        if 'year_month' in df.columns or 'year_week' in df.columns:
            time_col = 'year_month' if 'year_month' in df.columns else 'year_week'
            for model_name, predictions in predictions_dict.items():
                temp_df = df.copy()
                temp_df['anomaly'] = predictions
                anomaly_ts = temp_df.groupby(time_col)['anomaly'].sum()
                axes[2].plot(range(len(anomaly_ts)), anomaly_ts.values, 
                           marker='o', label=model_name, linewidth=2)
            
            axes[2].set_title('Anomalies Over Time', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Time Period')
            axes[2].set_ylabel('Number of Anomalies')
            axes[2].legend()
            axes[2].grid(alpha=0.3)
        
        # Plot 4: Distribution comparison
        if 'total_charges' in df.columns:
            analyzer = AnomalyAnalyzer()
            combined = analyzer.combine_predictions(predictions_dict, method='voting')
            
            normal_charges = df.loc[combined == 0, 'total_charges']
            anomaly_charges = df.loc[combined == 1, 'total_charges']
            
            axes[3].hist(normal_charges, bins=50, alpha=0.6, label='Normal', color='blue')
            axes[3].hist(anomaly_charges, bins=50, alpha=0.6, label='Anomaly', color='red')
            axes[3].set_title('Charge Distribution: Normal vs Anomaly', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Total Charges')
            axes[3].set_ylabel('Frequency')
            axes[3].legend()
            axes[3].set_yscale('log')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'anomaly_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Anomaly distribution plot saved to: {filepath}")
        plt.show()
    
    def plot_anomaly_scores(self, scores_dict: Dict[str, np.ndarray]):
        """Plot anomaly score distributions"""
        fig, axes = plt.subplots(1, len(scores_dict), figsize=(15, 4))
        if len(scores_dict) == 1:
            axes = [axes]
        
        for idx, (model_name, scores) in enumerate(scores_dict.items()):
            axes[idx].hist(scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{model_name}\nScore Distribution', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Anomaly Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'anomaly_scores.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Anomaly scores plot saved to: {filepath}")
        plt.show()


def main(db_path: str, time_period: str = 'M', contamination: float = 0.1):
    """
    Main execution pipeline for unsupervised anomaly detection
    
    Parameters:
    - db_path: Path to SQLite database
    - time_period: 'M' (monthly), 'W' (weekly), 'Q' (quarterly)
    - contamination: Expected proportion of outliers (0.05-0.15 typical)
    """
    
    print("="*100)
    print("HEALTHCARE PAYMENT INTEGRITY - UNSUPERVISED ANOMALY DETECTION")
    print("="*100)
    
    # 1. Load Data
    print("\n[1/6] Loading Data...")
    loader = DataLoader(db_path)
    df = loader.load_data()
    
    # 2. Create Time Series Features
    print("\n[2/6] Creating Time Series Features...")
    ts_engine = TimeSeriesFeatureEngine()
    df_ts = ts_engine.create_provider_time_series(df)
    
    # Aggregate by provider and time period
    agg_df = ts_engine.aggregate_provider_metrics(df_ts, time_period=time_period)
    print(f"Aggregated data: {agg_df.shape}")
    
    # Calculate deviations from baseline
    agg_df = ts_engine.calculate_deviations(agg_df)
    print(f"Features with deviations: {agg_df.shape[1]}")
    
    # 3. Isolation Forest Detection
    print("\n[3/6] Running Isolation Forest...")
    if_detector = IsolationForestDetector(contamination=contamination)
    if_labels, if_scores = if_detector.fit_predict(agg_df)
    print(f"Anomalies detected: {if_labels.sum()} ({if_labels.mean()*100:.2f}%)")
    
    # 4. LOF Detection
    print("\n[4/6] Running Local Outlier Factor...")
    lof_detector = LOFDetector(n_neighbors=20, contamination=contamination)
    lof_labels, lof_scores = lof_detector.fit_predict(agg_df)
    print(f"Anomalies detected: {lof_labels.sum()} ({lof_labels.mean()*100:.2f}%)")
    
    # 5. Autoencoder Detection
    print("\n[5/6] Training Autoencoder...")
    ae_detector = AutoencoderDetector(encoding_dim=16, contamination=contamination)
    ae_detector.fit(agg_df, epochs=50, batch_size=128)
    ae_labels, ae_errors = ae_detector.predict(agg_df)
    print(f"Anomalies detected: {ae_labels.sum()} ({ae_labels.mean()*100:.2f}%)")
    
    # 6. Export and Visualize Results
    print("\n[6/6] Exporting Results and Creating Visualizations...")
    print("="*100)
    
    predictions_dict = {
        'isolation_forest': if_labels,
        'lof': lof_labels,
        'autoencoder': ae_labels
    }
    
    scores_dict = {
        'isolation_forest': if_scores,
        'lof': lof_scores,
        'autoencoder': ae_errors
    }
    
    exporter = ResultsExporter()
    exporter.export_anomalies(agg_df, predictions_dict, scores_dict)
    exporter.export_summary_stats(agg_df, predictions_dict)
    
    # Generate visualizations
    exporter.plot_anomaly_distribution(agg_df, predictions_dict)
    exporter.plot_anomaly_scores(scores_dict)
    
    # Combined analysis
    analyzer = AnomalyAnalyzer()
    combined_labels = analyzer.combine_predictions(predictions_dict, method='voting')
    combined_scores = analyzer.calculate_anomaly_scores(scores_dict)
    
    print("\n" + "="*100)
    print("ANOMALY DETECTION SUMMARY")
    print("="*100)
    print(f"\nIsolation Forest: {if_labels.sum()} anomalies ({if_labels.mean()*100:.2f}%)")
    print(f"LOF: {lof_labels.sum()} anomalies ({lof_labels.mean()*100:.2f}%)")
    print(f"Autoencoder: {ae_labels.sum()} anomalies ({ae_labels.mean()*100:.2f}%)")
    print(f"Combined (Voting): {combined_labels.sum()} anomalies ({combined_labels.mean()*100:.2f}%)")
    
    # Agreement between models
    agreement = (if_labels == lof_labels).sum() / len(if_labels) * 100
    print(f"\nAgreement between IF and LOF: {agreement:.1f}%")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE - All outputs saved to './outputs/anomaly_detection' directory")
    print("="*100)
    
    return {
        'aggregated_data': agg_df,
        'predictions': predictions_dict,
        'scores': scores_dict,
        'combined_labels': combined_labels,
        'combined_scores': combined_scores,
        'models': {
            'isolation_forest': if_detector,
            'lof': lof_detector,
            'autoencoder': ae_detector
        }
    }


if __name__ == "__main__":
    db_path = os.path.join('Data', "claims_database.db")
    
    # Run analysis with monthly aggregation
    results = main(db_path, time_period='M', contamination=0.10)
    
    # Access results
    # results['aggregated_data'] - DataFrame with time-series features
    # results['predictions'] - Dictionary of anomaly predictions from each model
    # results['scores'] - Dictionary of anomaly scores from each model
    # results['combined_labels'] - Combined anomaly predictions (voting)
    # results['models'] - Trained model objects for future predictions