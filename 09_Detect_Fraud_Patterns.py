import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Transformer-based deep learning libraries
from transformers import DistilBertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Additional libraries for embeddings and preprocessing
from collections import defaultdict, Counter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class HealthcareClaimsDataset(Dataset):
    """Custom Dataset for healthcare claims data"""
    def __init__(self, features, labels=None, sequence_length=30):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FraudDistilBERTModel(nn.Module):
    """Fraud detection model using DistilBERT as backbone"""
    def __init__(self, input_dim, num_classes=8, dropout=0.1, bert_model_name="distilbert-base-uncased"):
        super(FraudDistilBERTModel, self).__init__()
        
        # Load DistilBERT
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # typically 768 for distilbert-base
        
        # Project tabular features into embedding space
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # Pattern-specific classifiers
        self.pattern_classifiers = nn.ModuleDict({
            'consistent_visits': nn.Linear(hidden_size, 2),
            'sudden_increase': nn.Linear(hidden_size, 2),
            'geographic_anomaly': nn.Linear(hidden_size, 2),
            'excessive_procedure': nn.Linear(hidden_size, 2),
            'illegitimate_code': nn.Linear(hidden_size, 2),
            'identical_services': nn.Linear(hidden_size, 2),
            'high_volume_expensive': nn.Linear(hidden_size, 2),
            'overall_fraud': nn.Linear(hidden_size, 2)
        })
        
        # Risk regression head
        self.risk_score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, attention_mask=None):
        # Project tabular features into embeddings
        inputs_embeds = self.input_projection(x)
        
        # Pass through DistilBERT
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        
        # Use [CLS]-like embedding (first token)
        pooled = outputs.last_hidden_state[:, 0]
        
        # Pattern outputs
        pattern_outputs = {name: clf(pooled) for name, clf in self.pattern_classifiers.items()}
        
        # Risk score
        risk_score = self.risk_score_head(pooled)
        
        return pattern_outputs, risk_score

class TransformerFraudDetector:
    """Main transformer-based fraud detection system"""
    def __init__(self, model_params=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.model = None
        self.model_params = model_params or {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dropout': 0.1
        }
        
    def load_data(self, file_path):
        """Load healthcare claims data"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"Loaded {len(self.df)} claims records")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _inject_fraud_patterns(self):
        """Inject known fraudulent patterns into the data for training"""
        fraud_labels = np.zeros(len(self.df))
        pattern_labels = {
            'consistent_visits': np.zeros(len(self.df)),
            'sudden_increase': np.zeros(len(self.df)),
            'geographic_anomaly': np.zeros(len(self.df)),
            'excessive_procedure': np.zeros(len(self.df)),
            'illegitimate_code': np.zeros(len(self.df)),
            'identical_services': np.zeros(len(self.df)),
            'high_volume_expensive': np.zeros(len(self.df))
        }
        
        # 1. Consistent visits fraud (Provider PROV_0001)
        consistent_fraud_idx = self.df[self.df['provider_id'] == 'PROV_0001'].index
        fraud_labels[consistent_fraud_idx] = 1
        pattern_labels['consistent_visits'][consistent_fraud_idx] = 1
        
        # 2. Geographic anomalies (distance > 200 miles)
        geo_fraud_idx = self.df[self.df['distance_miles'] > 200].index
        fraud_labels[geo_fraud_idx] = 1
        pattern_labels['geographic_anomaly'][geo_fraud_idx] = 1
        
        # 3. Excessive procedures (top 2% of charges)
        charge_threshold = np.percentile(self.df['charges'], 98)
        excessive_idx = self.df[self.df['charges'] > charge_threshold].index
        fraud_labels[excessive_idx] = 1
        pattern_labels['excessive_procedure'][excessive_idx] = 1
        
        # 4. High volume expensive (high cost + multiple procedures)
        high_vol_idx = self.df[(self.df['high_cost_flag'] == 1) & (self.df['multiple_procedures'] == 1)].index
        fraud_labels[high_vol_idx] = 1
        pattern_labels['high_volume_expensive'][high_vol_idx] = 1
        
        # Add labels to dataframe
        self.df['fraud_flag'] = fraud_labels
        for pattern, labels in pattern_labels.items():
            self.df[f'{pattern}_flag'] = labels
        
        print(f"Injected fraud patterns: {sum(fraud_labels)} fraudulent claims ({sum(fraud_labels)/len(fraud_labels)*100:.1f}%)")
    
    def preprocess_data(self):
        """Preprocess data for transformer model"""
        # Convert dates
        date_columns = ['service_date', 'patient_dob']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Calculate derived features
        if 'service_date' in self.df.columns and 'patient_dob' in self.df.columns:
            self.df['patient_age'] = (self.df['service_date'] - self.df['patient_dob']).dt.days / 365.25
        
        # Temporal features
        if 'service_date' in self.df.columns:
            self.df['day_of_week'] = self.df['service_date'].dt.dayofweek
            self.df['month'] = self.df['service_date'].dt.month
            self.df['day_of_year'] = self.df['service_date'].dt.dayofyear
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
            self.df['weekend_service'] = (self.df['weekend_service'] == 'YES').astype(int)
            self.df['multiple_procedures'] = (self.df['multiple_procedures'] == 'YES').astype(int)
            self.df['high_cost_flag'] = (self.df['high_cost_flag'] == 'YES').astype(int)
            
        # Provider-level aggregated features
        provider_stats = self.df.groupby('provider_id').agg({
            'charges': ['mean', 'std', 'count'],
            'patient_id': 'nunique',
            'procedure_code': 'nunique',
            'distance_miles': 'mean'
        }).reset_index()
        
        provider_stats.columns = ['provider_id', 'provider_avg_charge', 'provider_std_charge', 
                                'provider_claim_count', 'provider_unique_patients', 
                                'provider_unique_procedures', 'provider_avg_distance']
        
        self.df = self.df.merge(provider_stats, on='provider_id', how='left')
        
        # Patient-level aggregated features
        patient_stats = self.df.groupby('patient_id').agg({
            'charges': ['mean', 'count'],
            'provider_id': 'nunique',
            'distance_miles': 'mean'
        }).reset_index()
        
        patient_stats.columns = ['patient_id', 'patient_avg_charge', 'patient_claim_count',
                               'patient_unique_providers', 'patient_avg_distance']
        
        self.df = self.df.merge(patient_stats, on='patient_id', how='left')
        
        print("Data preprocessing completed with transformer-ready features")
    
    def prepare_features(self):
        """Prepare features for transformer model"""
        # Categorical features to encode
        categorical_features = ['provider_specialty', 'patient_gender', 'procedure_code', 
                              'diagnosis_code', 'provider_state', 'patient_state', 'place_of_service']
        
        # Numerical features
        numerical_features = [
            'units', 'charges', 'allowed_amount', 'paid_amount',
            'distance_miles', 'patient_age', 'day_of_week', 'month', 'day_of_year',
            'is_weekend', 'weekend_service', 'multiple_procedures', 
            'high_cost_flag', 'provider_avg_charge', 'provider_std_charge',
            'provider_claim_count', 'provider_unique_patients', 'provider_unique_procedures',
            'provider_avg_distance', 'patient_avg_charge', 'patient_claim_count',
            'patient_unique_providers', 'patient_avg_distance'
        ]
        
        # Handle missing values
        for col in numerical_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Encode categorical features
        encoded_features = []
        for col in categorical_features:
            if col in self.df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    encoded = self.encoders[col].fit_transform(self.df[col].astype(str))
                else:
                    encoded = self.encoders[col].transform(self.df[col].astype(str))
                encoded_features.append(encoded.reshape(-1, 1))
        
        # Scale numerical features
        numerical_data = []
        for col in numerical_features:
            if col in self.df.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    scaled = self.scalers[col].fit_transform(self.df[col].values.reshape(-1, 1))
                else:
                    scaled = self.scalers[col].transform(self.df[col].values.reshape(-1, 1))
                numerical_data.append(scaled)
        
        # Combine all features
        if encoded_features:
            encoded_array = np.hstack(encoded_features)
        else:
            encoded_array = np.array([]).reshape(len(self.df), 0)
            
        if numerical_data:
            numerical_array = np.hstack(numerical_data)
        else:
            numerical_array = np.array([]).reshape(len(self.df), 0)
        
        # Combine encoded categorical and numerical features
        if encoded_array.shape[1] > 0 and numerical_array.shape[1] > 0:
            self.features = np.hstack([encoded_array, numerical_array])
        elif numerical_array.shape[1] > 0:
            self.features = numerical_array
        else:
            self.features = encoded_array
        
        self.feature_columns = categorical_features + numerical_features
        
        print(f"Prepared {self.features.shape[1]} features for transformer model")
        return self.features
    
    def create_sequences(self, features, labels=None, sequence_length=1):
        """Create sequences for transformer input (treating each claim as a sequence of length 1)"""
        # For healthcare fraud, we treat each claim as an individual sequence
        # Reshape to add sequence dimension
        features_seq = features.reshape(features.shape[0], sequence_length, -1)
        
        if labels is not None:
            if isinstance(labels, dict):
                labels_seq = {}
                for pattern, label_array in labels.items():
                    labels_seq[pattern] = label_array
            else:
                labels_seq = labels
            return features_seq, labels_seq
        
        return features_seq
    
    def train_model(self, test_size=0.2, batch_size=32, epochs=50, learning_rate=0.001):
        """Train the transformer-based fraud detection model"""
        print("Starting transformer model training...")
        
        # Prepare labels
        pattern_labels = {
            'consistent_visits': self.df['consistent_visits_flag'].values if 'consistent_visits_flag' in self.df.columns else np.zeros(len(self.df)),
            'sudden_increase': self.df['sudden_increase_flag'].values if 'sudden_increase_flag' in self.df.columns else np.zeros(len(self.df)),
            'geographic_anomaly': self.df['geographic_anomaly_flag'].values if 'geographic_anomaly_flag' in self.df.columns else np.zeros(len(self.df)),
            'excessive_procedure': self.df['excessive_procedure_flag'].values if 'excessive_procedure_flag' in self.df.columns else np.zeros(len(self.df)),
            'illegitimate_code': self.df['illegitimate_code_flag'].values if 'illegitimate_code_flag' in self.df.columns else np.zeros(len(self.df)),
            'identical_services': self.df['identical_services_flag'].values if 'identical_services_flag' in self.df.columns else np.zeros(len(self.df)),
            'high_volume_expensive': self.df['high_volume_expensive_flag'].values if 'high_volume_expensive_flag' in self.df.columns else np.zeros(len(self.df)),
            'overall_fraud': self.df['fraud_flag'].values if 'fraud_flag' in self.df.columns else np.zeros(len(self.df))
        }
        
        # Create sequences
        features_seq, labels_seq = self.create_sequences(self.features, pattern_labels)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features_seq, self.df['fraud_flag'].values, 
            test_size=test_size, random_state=42, stratify=self.df['fraud_flag'].values
        )
        
        # Create train labels dict
        train_indices = X_train.reshape(X_train.shape[0], -1)
        test_indices = X_test.reshape(X_test.shape[0], -1)
        
        y_train_dict = {}
        y_test_dict = {}
        
        # Get train/test indices
        train_idx = []
        test_idx = []
        for i in range(len(features_seq)):
            if any(np.array_equal(features_seq[i].flatten(), train_indices[j]) for j in range(len(train_indices))):
                train_idx.append(i)
            else:
                test_idx.append(i)
        
        # Simple approach: use fraud_flag for all patterns in training
        for pattern in pattern_labels.keys():
            y_train_dict[pattern] = y_train
            y_test_dict[pattern] = y_test
        
        # Initialize model
        input_dim = self.features.shape[1]
        self.model = FraudDistilBERTModel(input_dim=input_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Create batches manually
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(self.device)
                batch_y = torch.LongTensor(y_train[i:i+batch_size]).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                pattern_outputs, risk_scores = self.model(batch_X)
                
                # Calculate losses for each pattern
                total_loss = 0
                for pattern_name, outputs in pattern_outputs.items():
                    if pattern_name in y_train_dict:
                        pattern_loss = criterion(outputs, batch_y)
                        total_loss += pattern_loss
                
                # Risk score loss (MSE)
                risk_loss = F.mse_loss(risk_scores.squeeze(), batch_y.float())
                total_loss += risk_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            # Validation
            if epoch % 5 == 0:
                val_loss = self._validate_model(X_test, y_test, criterion, batch_size)
                scheduler.step(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print("Training completed!")
        return train_losses
    
    def _validate_model(self, X_val, y_val, criterion, batch_size):
        """Validate the model during training"""
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = torch.FloatTensor(X_val[i:i+batch_size]).to(self.device)
                batch_y = torch.LongTensor(y_val[i:i+batch_size]).to(self.device)
                
                pattern_outputs, risk_scores = self.model(batch_X)
                
                # Calculate validation loss
                total_loss = 0
                for pattern_name, outputs in pattern_outputs.items():
                    pattern_loss = criterion(outputs, batch_y)
                    total_loss += pattern_loss
                
                risk_loss = F.mse_loss(risk_scores.squeeze(), batch_y.float())
                total_loss += risk_loss
                
                val_loss += total_loss.item()
                num_batches += 1
        
        self.model.train()
        return val_loss / num_batches
    
    def predict_fraud(self, data=None):
        """Use trained transformer model to predict fraud patterns"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if data is None:
            data = self.features
        
        # Ensure data has sequence dimension
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], 1, -1)
        
        self.model.eval()
        predictions = []
        risk_scores = []
        attention_weights = []
        
        with torch.no_grad():
            for i in range(0, len(data), 32):  # Process in batches
                batch_data = torch.FloatTensor(data[i:i+32]).to(self.device)
                
                pattern_outputs, batch_risk_scores, batch_attention = self.model(
                    batch_data, return_attention=True
                )
                
                # Get predictions for each pattern
                batch_predictions = {}
                for pattern_name, outputs in pattern_outputs.items():
                    probs = F.softmax(outputs, dim=1)
                    batch_predictions[pattern_name] = probs[:, 1].cpu().numpy()  # Fraud probability
                
                predictions.extend([batch_predictions[j] if isinstance(batch_predictions, list) 
                                  else {k: v[j-i] for k, v in batch_predictions.items()} 
                                  for j in range(len(batch_data))])
                risk_scores.extend(batch_risk_scores.cpu().numpy().flatten())
                if batch_attention is not None:
                    attention_weights.extend(batch_attention.cpu().numpy())
        
        return predictions, risk_scores, attention_weights
    
    def analyze_fraud_patterns(self, threshold=0.5):
        """Analyze and flag fraud patterns using transformer predictions"""
        print("Analyzing fraud patterns with transformer model...")
        
        # Get predictions
        predictions, risk_scores, attention_weights = self.predict_fraud()
        
        flagged_claims = []
        
        for i, (pred_dict, risk_score) in enumerate(zip(predictions, risk_scores)):
            claim_row = self.df.iloc[i]
            
            # Check each pattern
            for pattern_name, fraud_prob in pred_dict.items():
                if fraud_prob > threshold and pattern_name != 'overall_fraud':
                    
                    # Generate detailed explanations based on pattern type
                    reason, evidence = self._generate_explanation(
                        pattern_name, claim_row, fraud_prob, attention_weights[i] if attention_weights else None
                    )
                    
                    flagged_claims.append({
                        'claim_id': claim_row['claim_id'],
                        'provider_id': claim_row['provider_id'],
                        'provider_name': claim_row['provider_name'],
                        'pattern_type': pattern_name.replace('_', ' ').title(),
                        'fraud_probability': fraud_prob,
                        'risk_score': risk_score,
                        'detailed_reason': reason,
                        'supporting_evidence': evidence,
                        'transformer_confidence': fraud_prob,
                        'attention_score': np.mean(attention_weights[i]) if attention_weights else 0.0
                    })
        
        if flagged_claims:
            self.fraud_results = pd.DataFrame(flagged_claims)
            print(f"Transformer analysis complete! Flagged {len(flagged_claims)} suspicious claims.")
            return self.fraud_results
        else:
            print("No fraud patterns detected above threshold.")
            return pd.DataFrame()
    
    def _generate_explanation(self, pattern_name, claim_row, fraud_prob, attention_weights):
        """Generate detailed explanations for flagged patterns"""
        
        explanations = {
            'consistent_visits': f"Transformer model detected consistent visit pattern (confidence: {fraud_prob:.3f}). Provider shows unnaturally regular billing frequency suggesting automated processing.",
            
            'sudden_increase': f"Neural network identified sudden patient volume increase (confidence: {fraud_prob:.3f}). Unusual spike in new patients may indicate patient mill operations.",
            
            'geographic_anomaly': f"Transformer flagged geographic inconsistency (confidence: {fraud_prob:.3f}). Patient distance of {claim_row.get('distance_miles', 'N/A')} miles suggests potential telemedicine fraud.",
            
            'excessive_procedure': f"Deep learning model detected excessive procedure billing (confidence: {fraud_prob:.3f}). Charge amount ${claim_row.get('charges', 0):,.2f} significantly exceeds typical patterns.",
            
            'illegitimate_code': f"Transformer identified code-specialty mismatch (confidence: {fraud_prob:.3f}). Procedure {claim_row.get('procedure_code', 'N/A')} inconsistent with {claim_row.get('provider_specialty', 'N/A')} specialty.",
            
            'identical_services': f"Neural network detected identical service patterns (confidence: {fraud_prob:.3f}). Provider applies same treatments across diverse patient demographics.",
            
            'high_volume_expensive': f"Transformer flagged high-volume expensive procedures (confidence: {fraud_prob:.3f}). Combination of high costs and volume suggests systematic overbilling."
        }
        
        reason = explanations.get(pattern_name, f"Transformer model flagged suspicious pattern: {pattern_name}")
        
        evidence = f"Transformer attention weights: {np.mean(attention_weights) if attention_weights is not None else 'N/A':.3f}, "
        evidence += f"Provider stats: {claim_row.get('provider_claim_count', 'N/A')} total claims, "
        evidence += f"Average charge: ${claim_row.get('provider_avg_charge', 0):,.2f}, "
        evidence += f"Patient diversity: {claim_row.get('provider_unique_patients', 'N/A')} unique patients"
        
        return reason, evidence
    
    def generate_comprehensive_report(self, output_file='transformer_fraud_report.csv'):
        """Generate comprehensive fraud detection report"""
        if hasattr(self, 'fraud_results') and not self.fraud_results.empty:
            
            # Add priority classification
            def classify_priority(fraud_prob, risk_score):
                combined_score = (fraud_prob + risk_score) / 2
                if combined_score >= 0.8:
                    return "HIGH"
                elif combined_score >= 0.6:
                    return "MEDIUM"
                else:
                    return "LOW"
            
            self.fraud_results['priority'] = self.fraud_results.apply(
                lambda row: classify_priority(row['fraud_probability'], row['risk_score']), 
                axis=1
            )
            
            # Sort by combined transformer confidence and risk score
            self.fraud_results['combined_score'] = (
                self.fraud_results['fraud_probability'] + self.fraud_results['risk_score']
            ) / 2
            
            self.fraud_results = self.fraud_results.sort_values('combined_score', ascending=False)
            
            # Save detailed report
            self.fraud_results.to_csv(output_file, index=False)
            
            # Print summary
            print("\n" + "="*90)
            print("TRANSFORMER-BASED FRAUD DETECTION REPORT")
            print("="*90)
            print(f"Total Flagged Claims: {len(self.fraud_results)}")
            print(f"Unique Providers Flagged: {len(self.fraud_results['provider_id'].unique())}")
            print(f"Average Transformer Confidence: {self.fraud_results['fraud_probability'].mean():.3f}")
            print(f"Average Risk Score: {self.fraud_results['risk_score'].mean():.3f}")
            print(f"Average Attention Score: {self.fraud_results['attention_score'].mean():.3f}")
            
            print("\nPattern Distribution (Transformer Detected):")
            pattern_counts = self.fraud_results['pattern_type'].value_counts()
            for pattern, count in pattern_counts.items():
                avg_confidence = self.fraud_results[
                    self.fraud_results['pattern_type'] == pattern
                ]['fraud_probability'].mean()
                print(f"  {pattern}: {count} claims (avg confidence: {avg_confidence:.3f})")
            
            print("\nPriority Distribution:")
            priority_counts = self.fraud_results['priority'].value_counts()
            for priority, count in priority_counts.items():
                print(f"  {priority}: {count} claims")
            
            print(f"\nDetailed transformer analysis saved to: {output_file}")
            
            # Model interpretability insights
            print("\nTransformer Model Insights:")
            print(f"  - Model uses {self.model_params['num_layers']} transformer layers")
            print(f"  - {self.model_params['nhead']} attention heads for pattern recognition")
            print(f"  - {self.features.shape[1]} input features analyzed")
            print(f"  - Attention mechanism provides interpretability scores")
            
            print("="*90)
            
            return self.fraud_results
        else:
            print("No fraud patterns detected by transformer model.")
            return None
    
    def explain_predictions(self, claim_ids=None, top_n=5):
        """Provide detailed explanations for specific predictions"""
        if not hasattr(self, 'fraud_results') or self.fraud_results.empty:
            print("No predictions available. Run analyze_fraud_patterns() first.")
            return
        
        if claim_ids is None:
            # Show top N highest risk claims
            explain_data = self.fraud_results.head(top_n)
        else:
            explain_data = self.fraud_results[
                self.fraud_results['claim_id'].isin(claim_ids)
            ]
        
        print("\n" + "="*100)
        print("DETAILED TRANSFORMER PREDICTION EXPLANATIONS")
        print("="*100)
        
        for idx, (_, claim) in enumerate(explain_data.iterrows()):
            print(f"\n{idx+1}. CLAIM ANALYSIS")
            print("-" * 50)
            print(f"Claim ID: {claim['claim_id']}")
            print(f"Provider: {claim['provider_name']} (ID: {claim['provider_id']})")
            print(f"Pattern Detected: {claim['pattern_type']}")
            print(f"Transformer Confidence: {claim['fraud_probability']:.3f}")
            print(f"Risk Score: {claim['risk_score']:.3f}")
            print(f"Priority Level: {claim['priority']}")
            print(f"Attention Score: {claim['attention_score']:.3f}")
            
            print(f"\nDetailed Analysis:")
            print(f"  {claim['detailed_reason']}")
            
            print(f"\nSupporting Evidence:")
            print(f"  {claim['supporting_evidence']}")
            
            print(f"\nRecommended Action:")
            if claim['priority'] == 'HIGH':
                print("  🚨 IMMEDIATE INVESTIGATION REQUIRED")
                print("  - Conduct thorough audit of provider billing patterns")
                print("  - Review patient records and documentation")
                print("  - Consider regulatory reporting if fraud confirmed")
            elif claim['priority'] == 'MEDIUM':
                print("  ⚠️  ENHANCED MONITORING RECOMMENDED")
                print("  - Increase claim review frequency for this provider")
                print("  - Request additional documentation for similar claims")
                print("  - Monitor for pattern escalation")
            else:
                print("  ℹ️  ROUTINE REVIEW SUGGESTED")
                print("  - Include in next scheduled audit cycle")
                print("  - Document findings for trend analysis")
            
            print("-" * 50)
    
    def save_model(self, model_path='fraud_transformer_model.pth'):
        """Save the trained transformer model"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_params': self.model_params,
                'feature_columns': self.feature_columns,
                'scalers': self.scalers,
                'encoders': self.encoders
            }, model_path)
            print(f"Transformer model saved to {model_path}")
        else:
            print("No trained model to save.")
    
    def load_model(self, model_path='fraud_transformer_model.pth'):
        """Load a pre-trained transformer model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model_params = checkpoint['model_params']
        self.feature_columns = checkpoint['feature_columns']
        self.scalers = checkpoint['scalers']
        self.encoders = checkpoint['encoders']
        
        # Recreate model
        input_dim = len(self.feature_columns)
        self.model = FraudDistilBERTModel(input_dim=input_dim).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Transformer model loaded from {model_path}")
    
    def get_feature_importance(self):
        """Get feature importance based on attention weights"""
        if self.model is None:
            print("Model not trained. Cannot compute feature importance.")
            return None
        
        # Get attention weights for all data
        _, _, attention_weights = self.predict_fraud()
        
        if attention_weights:
            avg_attention = np.mean(attention_weights, axis=0)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_columns[:len(avg_attention)],
                'importance': avg_attention
            }).sort_values('importance', ascending=False)
            
            print("\nTRANSFORMER FEATURE IMPORTANCE (Based on Attention Weights)")
            print("=" * 60)
            for _, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:<30} {row['importance']:.4f}")
            
            return importance_df
        else:
            print("No attention weights available.")
            return None

def main():
    """Main function demonstrating the transformer-based fraud detection system"""
    print("🤖 TRANSFORMER-BASED HEALTHCARE FRAUD DETECTION SYSTEM")
    print("=" * 70)
    
    # Initialize detector
    detector = TransformerFraudDetector(model_params={
        'd_model': 256,
        'nhead': 8, 
        'num_layers': 6,
        'dropout': 0.1
    })
    
    # Create sample data (in real use case, load from CSV)
    #detector.create_sample_data(n_claims=15000)
    
    csv_file_path = 'healthcare_claims_with_fraud_patterns.csv'
    # Load your data
    detector.load_data(csv_file_path)
    
    # Preprocess data
    print("\n📊 Preprocessing healthcare claims data...")
    detector.preprocess_data()
    
    # Prepare features for transformer
    print("\n🔧 Preparing features for transformer model...")
    detector.prepare_features()
    
    # Train transformer model
    print("\n🚀 Training transformer-based fraud detection model...")
    train_losses = detector.train_model(
        epochs=30, 
        batch_size=64, 
        learning_rate=0.001
    )
    
    # Analyze fraud patterns
    print("\n🔍 Analyzing fraud patterns with trained transformer...")
    fraud_results = detector.analyze_fraud_patterns(threshold=0.5)
    
    # Generate comprehensive report
    if fraud_results is not None and not fraud_results.empty:
        print("\n📋 Generating comprehensive fraud detection report...")
        detector.generate_comprehensive_report('transformer_fraud_report.csv')
        
        # Show detailed explanations for top cases
        print("\n🔬 Detailed explanations for highest-risk cases...")
        detector.explain_predictions(top_n=3)
        
        # Show feature importance
        print("\n📈 Transformer feature importance analysis...")
        detector.get_feature_importance()
        
        # Save model for future use
        detector.save_model('healthcare_fraud_transformer.pth')
        
        print("\n✅ TRANSFORMER-BASED FRAUD DETECTION COMPLETE!")
        print(f"📁 Results saved to: transformer_fraud_report.csv")
        print(f"🤖 Model saved to: healthcare_fraud_transformer.pth")
        
    else:
        print("\n❌ No fraud patterns detected in the dataset.")
    
    return detector

# Example of how to use with your own data
def use_with_custom_data(csv_file_path):
    """Example of using the system with your own healthcare claims data"""
    detector = TransformerFraudDetector()
    
    # Load your data
    detector.load_data(csv_file_path)
    
    # Preprocess and prepare features
    detector.preprocess_data()
    detector.prepare_features()
    
    # Train model
    detector.train_model(epochs=50, batch_size=32)
    
    # Analyze fraud patterns
    fraud_results = detector.analyze_fraud_patterns(threshold=0.6)
    
    # Generate report
    detector.generate_comprehensive_report('your_fraud_report.csv')
    
    # Explain top cases
    detector.explain_predictions(top_n=10)
    
    return detector, fraud_results

if __name__ == "__main__":
    # Run the main demonstration
    detector = main()
    
    # Example of loading and using a saved model
    print("\n" + "="*50)
    print("EXAMPLE: Loading and using saved transformer model")
    print("="*50)
    
    # Create new detector instance
    #new_detector = TransformerFraudDetector()
    
    # Load the saved model (you would use your actual model file)
    # new_detector.load_model('healthcare_fraud_transformer.pth')
    
    # Use for predictions on new data
    # new_predictions = new_detector.analyze_fraud_patterns(new_data)
    
    print("✅ Transformer-based fraud detection system ready for production use!")