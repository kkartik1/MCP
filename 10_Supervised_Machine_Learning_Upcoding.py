"""
Healthcare Claims Upcoding Prediction System
A modular framework for detecting upcoding in healthcare claims using multiple ML techniques
Enhanced with CSV export and comprehensive visualization
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            precision_recall_curve, f1_score, accuracy_score,
                            precision_score, recall_score, roc_curve)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Feature Importance
from sklearn.inspection import permutation_importance
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader:
    """Module for loading and initial data exploration"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def load_data(self, query: str = None) -> pd.DataFrame:
        """Load data from SQLite database"""
        if query is None:
            query = "SELECT * FROM claims_data where allowed_amount > 1000"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate data summary statistics"""
        summary = {
            'shape': df.shape,
            'missing_values': df.isnull().sum(),
            'dtypes': df.dtypes,
            'target_distribution': df['is_upcoded_line'].value_counts(),
            'target_percentage': df['is_upcoded_line'].value_counts(normalize=True) * 100
        }
        return summary


class FeatureEngineering:
    """Module for feature engineering and preprocessing"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data"""
        df = df.copy()
        
        # Date features
        if 'service_date_header' in df.columns:
            df['service_date_header'] = pd.to_datetime(df['service_date_header'])
            df['service_year'] = df['service_date_header'].dt.year
            df['service_month'] = df['service_date_header'].dt.month
            df['service_day_of_week'] = df['service_date_header'].dt.dayofweek
            df['service_quarter'] = df['service_date_header'].dt.quarter
        
        if 'service_date_line' in df.columns:
            df['service_date_line'] = pd.to_datetime(df['service_date_line'])
        
        # Admission/Discharge features
        if 'admission_date' in df.columns and 'discharge_date' in df.columns:
            df['admission_date'] = pd.to_datetime(df['admission_date'])
            df['discharge_date'] = pd.to_datetime(df['discharge_date'])
            df['calculated_los'] = (df['discharge_date'] - df['admission_date']).dt.days
            df['los_match'] = (df['calculated_los'] == df['length_of_stay']).astype(int)
        
        # Financial features
        if 'total_charges_header' in df.columns and 'expected_payment' in df.columns:
            df['charge_to_payment_ratio'] = df['total_charges_header'] / (df['expected_payment'] + 1)
            df['payment_percentage'] = (df['expected_payment'] / (df['total_charges_header'] + 1)) * 100
        
        if 'total_charges_line' in df.columns and 'allowed_amount' in df.columns:
            df['line_charge_to_allowed_ratio'] = df['total_charges_line'] / (df['allowed_amount'] + 1)
            df['allowed_percentage'] = (df['allowed_amount'] / (df['total_charges_line'] + 1)) * 100
        
        if 'units' in df.columns and 'unit_cost' in df.columns:
            df['calculated_line_charges'] = df['units'] * df['unit_cost']
            df['charge_discrepancy'] = abs(df['total_charges_line'] - df['calculated_line_charges'])
        
        # Aggregate features by provider
        provider_stats = df.groupby('provider_id').agg({
            'total_charges_line': ['mean', 'sum', 'count'],
            'units': 'mean',
            'is_upcoded_line': 'mean'
        }).reset_index()
        provider_stats.columns = ['provider_id', 'provider_avg_charges', 'provider_total_charges',
                                  'provider_claim_count', 'provider_avg_units', 'provider_upcoding_rate']
        df = df.merge(provider_stats, on='provider_id', how='left')
        
        # Aggregate by specialty
        specialty_stats = df.groupby('provider_specialty').agg({
            'total_charges_line': 'mean',
            'is_upcoded_line': 'mean'
        }).reset_index()
        specialty_stats.columns = ['provider_specialty', 'specialty_avg_charges', 'specialty_upcoding_rate']
        df = df.merge(specialty_stats, on='provider_specialty', how='left')
        
        # HCPCS Code features
        if 'hcpcs_code' in df.columns:
            hcpcs_stats = df.groupby('hcpcs_code').agg({
                'total_charges_line': ['mean', 'count'],
                'is_upcoded_line': 'mean'
            }).reset_index()
            hcpcs_stats.columns = ['hcpcs_code', 'hcpcs_avg_charges', 'hcpcs_frequency', 'hcpcs_upcoding_rate']
            df = df.merge(hcpcs_stats, on='hcpcs_code', how='left')
        
        # Procedure description features
        if 'procedure_description' in df.columns:
            df['procedure_desc_length'] = df['procedure_description'].fillna('').str.len()
            df['procedure_desc_word_count'] = df['procedure_description'].fillna('').str.split().str.len()
            
            proc_stats = df.groupby('procedure_description').agg({
                'total_charges_line': ['mean', 'count'],
                'is_upcoded_line': 'mean'
            }).reset_index()
            proc_stats.columns = ['procedure_description', 'proc_avg_charges', 'proc_frequency', 'proc_upcoding_rate']
            df = df.merge(proc_stats, on='procedure_description', how='left')
        
        # Diagnosis complexity
        df['has_secondary_diagnosis'] = df['secondary_diagnosis'].notna().astype(int)
        
        # Unusual patterns
        df['high_units'] = (df['units'] > df['units'].quantile(0.95)).astype(int)
        df['high_charges'] = (df['total_charges_line'] > df['total_charges_line'].quantile(0.95)).astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'is_upcoded_line') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling"""
        df = df.copy()
        
        # Separate target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Drop ID columns and dates
        id_cols = ['claim_id', 'patient_id', 'member_id', 'provider_id', 'provider_npi']
        date_cols = ['service_date_header', 'service_date_line', 'admission_date', 'discharge_date']
        text_cols = ['upcoding_reason']
        
        leakage_cols = ['line_number', 'upcoding_type_header', 'is_upcoded_header', 'upcoding_reason', 'upcoding_type_line', 'claim_id']
        
        cols_to_drop = [col for col in id_cols + date_cols + text_cols + leakage_cols if col in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Convert boolean to int if present
        bool_cols = X.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X[col] = X[col].astype(int)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y


class ModelTrainer:
    """Module for training individual ML models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def get_models(self) -> Dict:
        """Initialize all ML models"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        return models
    
    def train_model(self, model, X_train, y_train, X_test, y_test, model_name: str) -> Dict:
        """Train and evaluate a single model"""
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        results = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        results['cv_f1_mean'] = cv_scores.mean()
        results['cv_f1_std'] = cv_scores.std()
        
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}" if results['roc_auc'] else "ROC AUC: N/A")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train all models and store results"""
        models = self.get_models()
        
        for name, model in models.items():
            self.results[name] = self.train_model(model, X_train, y_train, X_test, y_test, name)
            self.models[name] = self.results[name]['model']
        
        return self.results


class EnsembleModeler:
    """Module for ensemble methods"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ensemble_models = {}
        
    def create_voting_ensemble(self, base_models: Dict) -> VotingClassifier:
        """Create voting ensemble from base models"""
        estimators = [(name, model) for name, model in base_models.items()]
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        return voting_clf
    
    def create_stacking_ensemble(self) -> StackingClassifier:
        """Create stacking ensemble"""
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=self.random_state, eval_metric='logloss'))
        ]
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )
        return stacking_clf
    
    def train_ensemble(self, ensemble_model, X_train, y_train, X_test, y_test, name: str) -> Dict:
        """Train and evaluate ensemble model"""
        print(f"\nTraining {name}...")
        
        ensemble_model.fit(X_train, y_train)
        y_pred = ensemble_model.predict(X_test)
        y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
        
        results = {
            'model': ensemble_model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        
        return results


class ResultsExporter:
    """Module for exporting results to CSV and visualization"""
    
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_predictions_to_csv(self, results: Dict, filename: str = 'model_predictions.csv') -> str:
        """Export predictions from all models to a single CSV"""
        filepath = os.path.join(self.output_dir, filename)
        
        # Get test set from first model
        first_model_results = next(iter(results.values()))
        y_test = first_model_results['y_test']
        
        # Create dataframe with actual values
        export_df = pd.DataFrame({
            'actual': y_test.values
        })
        
        # Add predictions and probabilities from each model
        for model_name, result in results.items():
            export_df[f'{model_name}_pred'] = result['y_pred']
            if result['y_pred_proba'] is not None:
                export_df[f'{model_name}_proba'] = result['y_pred_proba']
        
        export_df.to_csv(filepath, index=False)
        print(f"\nPredictions exported to: {filepath}")
        return filepath
    
    def export_metrics_summary(self, results: Dict, filename: str = 'model_metrics.csv') -> str:
        """Export comprehensive metrics summary"""
        filepath = os.path.join(self.output_dir, filename)
        
        metrics_list = []
        for model_name, result in results.items():
            metrics_list.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1'],
                'ROC AUC': result.get('roc_auc', np.nan),
                'CV F1 Mean': result.get('cv_f1_mean', np.nan),
                'CV F1 Std': result.get('cv_f1_std', np.nan)
            })
        
        metrics_df = pd.DataFrame(metrics_list).sort_values('F1 Score', ascending=False)
        metrics_df.to_csv(filepath, index=False)
        print(f"Metrics exported to: {filepath}")
        return filepath
    
    def plot_confusion_matrices(self, results: Dict, top_n: int = 4):
        """Plot confusion matrices for top models"""
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1]['f1'], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(sorted_models):
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', 
                       ax=axes[idx], cmap='Blues', cbar=False)
            axes[idx].set_title(f'{name}\nF1: {result["f1"]:.4f}')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'confusion_matrices.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {filepath}")
        plt.show()
    
    def plot_roc_curves(self, results: Dict):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for model_name, result in results.items():
            if result.get('y_pred_proba') is not None:
                fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                auc_score = roc_auc_score(result['y_test'], result['y_pred_proba'])
                ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {filepath}")
        plt.show()
    
    def plot_metrics_comparison(self, results: Dict):
        """Create comprehensive metrics comparison visualization"""
        metrics_list = []
        for model_name, result in results.items():
            metrics_list.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1'],
                'ROC AUC': result.get('roc_auc', np.nan)
            })
        
        df_metrics = pd.DataFrame(metrics_list).set_index('Model')
        df_metrics = df_metrics.sort_values('F1 Score', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        df_metrics.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Metrics Comparison', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].legend(loc='best')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(alpha=0.3, axis='y')
        
        # Heatmap
        sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[1], cbar_kws={'label': 'Score'})
        axes[1].set_title('Metrics Heatmap', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {filepath}")
        plt.show()


class ModelEvaluator:
    """Module for comprehensive model evaluation"""
    
    def compare_models(self, results: Dict) -> pd.DataFrame:
        """Compare all models side by side"""
        comparison = []
        
        for name, result in results.items():
            comparison.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1'],
                'ROC AUC': result.get('roc_auc', None),
                'CV F1 Mean': result.get('cv_f1_mean', None),
                'CV F1 Std': result.get('cv_f1_std', None)
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('F1 Score', ascending=False)
        return comparison_df
    
    def print_summary(self, comparison_df: pd.DataFrame):
        """Print detailed summary statistics"""
        print("\n" + "="*100)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*100)
        print("\nDetailed Metrics:")
        print(comparison_df.to_string(index=False))
        
        print("\n" + "-"*100)
        print("SUMMARY STATISTICS")
        print("-"*100)
        
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        summary_stats = comparison_df[numeric_cols].describe()
        print("\n" + summary_stats.to_string())
        
        print("\n" + "-"*100)
        print("BEST PERFORMERS")
        print("-"*100)
        for col in numeric_cols:
            best_idx = comparison_df[col].idxmax()
            best_model = comparison_df.loc[best_idx, 'Model']
            best_score = comparison_df.loc[best_idx, col]
            print(f"Best {col}: {best_model} ({best_score:.4f})")


# Main execution pipeline
def main(db_path: str):
    """Main execution pipeline"""
    
    print("="*100)
    print("HEALTHCARE CLAIMS UPCODING PREDICTION SYSTEM")
    print("="*100)
    
    # 1. Load Data
    print("\n[1/8] Loading Data...")
    loader = DataLoader(db_path)
    df = loader.load_data()
    summary = loader.data_summary(df)
    print(f"\nTarget Distribution:\n{summary['target_distribution']}")
    print(f"\nTarget Percentage:\n{summary['target_percentage']}")
    
    # 2. Feature Engineering
    print("\n[2/8] Engineering Features...")
    fe = FeatureEngineering()
    df_engineered = fe.engineer_features(df)
    print(f"Features after engineering: {df_engineered.shape[1]}")
    
    # 3. Prepare Features
    print("\n[3/8] Preparing Features...")
    X, y = fe.prepare_features(df_engineered)
    print(f"Final feature set: {X.shape}")
    print(f"Features: {list(X.columns[:10])}... (showing first 10)")
    
    # 4. Train-Test Split
    print("\n[4/8] Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 5. Train Individual Models
    print("\n[5/8] Training Individual Models...")
    print("="*100)
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # 6. Train Ensemble Models
    print("\n[6/8] Training Ensemble Models...")
    print("="*100)
    ensemble = EnsembleModeler()
    
    top_models = dict(sorted(results.items(), 
                           key=lambda x: x[1]['f1'], reverse=True)[:3])
    top_base_models = {name: results[name]['model'] for name in top_models.keys()}
    
    voting_clf = ensemble.create_voting_ensemble(top_base_models)
    results['Voting Ensemble'] = ensemble.train_ensemble(
        voting_clf, X_train, y_train, X_test, y_test, 'Voting Ensemble'
    )
    
    stacking_clf = ensemble.create_stacking_ensemble()
    results['Stacking Ensemble'] = ensemble.train_ensemble(
        stacking_clf, X_train, y_train, X_test, y_test, 'Stacking Ensemble'
    )
    
    # 7. Export Results
    print("\n[7/8] Exporting Results...")
    print("="*100)
    exporter = ResultsExporter(output_dir='outputs')
    exporter.export_predictions_to_csv(results)
    exporter.export_metrics_summary(results)
    
    # 8. Evaluate and Visualize
    print("\n[8/8] Generating Visualizations and Summary...")
    print("="*100)
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(results)
    
    evaluator.print_summary(comparison_df)
    
    # Generate visualizations
    exporter.plot_confusion_matrices(results)
    exporter.plot_roc_curves(results)
    exporter.plot_metrics_comparison(results)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE - All outputs saved to './outputs' directory")
    print("="*100)
    
    return results, comparison_df, X.columns.tolist()


if __name__ == "__main__":
    db_path = os.path.join('Data', "claims_database.db")
    results, comparison, feature_names = main(db_path)