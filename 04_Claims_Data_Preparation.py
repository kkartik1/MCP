"""
Claims Data Preparation Module
Prepares data from SQLite database for ML model training by creating combined text sequences
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from typing import List, Dict, Optional, Tuple
import re, os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CodeMappings:
    """
    Data preparation class for loading key fields description
    """
    def get_mapping(table_name, key_column, value_column):
        """
        Reads data from a SQLite table and returns it as a dictionary.

        Args:
            db_path (str): The path to the SQLite database file.
            table_name (str): The name of the table to read from.
            key_column (str): The name of the column to use for dictionary keys.
            value_column (str): The name of the column to use for dictionary values.

        Returns:
            dict: A dictionary mapping key_column values to value_column values.
        """
        try:
            # Construct the SQL query
            query = f"SELECT {key_column}, {value_column} FROM {table_name};"
            db_path = os.path.join('Data',"claims_database.db")
            engine = create_engine(f'sqlite:///{db_path}', echo=False)
            logger.info(f"Data preparation initialized for database: {db_path}")
            df = pd.read_sql_query(query, engine)
            df.columns = ['code', 'description']
            logger.info(f"Loaded {len(df)} records from database")
            # Create the dictionary using a dictionary comprehension and zip
            mappings = {code: description for code, description in zip(df['code'], df['description'])}
            return mappings
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise
        
    rev_mapping = get_mapping('revenue_codes', 'revenue_code', 'short_description')
    mod_mapping = get_mapping('modifier_codes', 'modifier', 'short_description')
    icd_mapping = get_mapping('icd_codes', 'icd_code', 'description')
    drg_mapping = get_mapping('drg_codes', 'ms_drg_code', 'ms_drg_description')
    pos_mapping = get_mapping('pos_codes', 'pos_code', 'description')
        

class ClaimsDataPreparation:
    """
    Data preparation class for processing claims data for ML model training
    """
    
    def __init__(self, db_path: str = os.path.join('Data',"claims_database.db")):
        """
        Initialize the data preparation module
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.code_mappings = CodeMappings()
        
        logger.info(f"Data preparation initialized for database: {db_path}")
    
    def load_claims_data(self) -> pd.DataFrame:
        """
        Load claims data from SQLite database
        
        Returns:
            pd.DataFrame: Claims data
        """
        try:
            query = """
            SELECT 
                claim_id,
                line_number,
                claim_type,
                provider_id,
                provider_specialty,
                place_of_service,
                primary_diagnosis,
                secondary_diagnosis,
                drg_code,
                revenue_code,
                hcpcs_code,
                procedure_description,
                units,
                total_charges_line,
                allowed_amount,
                is_upcoded_header,
                is_upcoded_line,
                upcoding_type_header,
                upcoding_type_line,
                upcoding_reason
            FROM claims_data
            """
            
            df = pd.read_sql_query(query, self.engine)
            logger.info(f"Loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise
    
    def get_code_description(self, code: str, code_type: str) -> str:
        """
        Get description for medical codes
        
        Args:
            code (str): Medical code
            code_type (str): Type of code ('cpt', 'icd', 'modifier', 'pos')
            
        Returns:
            str: Code description or the original code if not found
        """
        if pd.isna(code) or not code:
            return "Unknown"
        
        code = code.replace('.', '')
        code = str(code).strip().upper()
        
        mappings = {
            'rev': self.code_mappings.rev_mapping,
            'mod': self.code_mappings.mod_mapping,
            'icd': self.code_mappings.icd_mapping,
            'drg': self.code_mappings.drg_mapping,
            'pos': self.code_mappings.pos_mapping
        }
        
        code_dict = mappings.get(code_type.lower(), {})
        return code_dict.get(code, f"Code {code}")
    
    def extract_modifiers(self, hcpcs_code: str) -> Tuple[str, List[str]]:
        """
        Extract CPT code and modifiers from HCPCS code
        
        Args:
            hcpcs_code (str): HCPCS code potentially with modifiers
            
        Returns:
            Tuple[str, List[str]]: CPT code and list of modifiers
        """
        if pd.isna(hcpcs_code) or not hcpcs_code:
            return "Unknown", []
        
        hcpcs_code = str(hcpcs_code).strip()
        
        # Split by common delimiters
        parts = re.split(r'[-,\s]+', hcpcs_code)
        cpt_code = parts[0] if parts else "Unknown"
        modifiers = [m.strip() for m in parts[1:] if m.strip()]
        
        return cpt_code, modifiers
    
    def create_text_sequence(self, row: pd.Series) -> str:
        """
        Create a combined text sequence for BERT model input
        
        Args:
            row (pd.Series): Single row of claims data
            
        Returns:
            str: Formatted text sequence
        """
        # Extract CPT code and modifiers
        cpt_code, modifiers = self.extract_modifiers(row.get('hcpcs_code', ''))
        
        # Get descriptions
        cpt_description = row.get('procedure_description')
        primary_icd_desc = self.get_code_description(row.get('primary_diagnosis', ''), 'icd')
        secondary_icd_desc = self.get_code_description(row.get('secondary_diagnosis', ''), 'icd')
        pos_desc = self.get_code_description(row.get('place_of_service', ''), 'pos')
        rev_desc = self.get_code_description(row.get('revenue_code', ''), 'rev')
        drg_desc = self.get_code_description(row.get('drg_code', ''), 'drg')
        
        # Build modifier descriptions
        modifier_descriptions = []
        for modifier in modifiers:
            modifier_desc = self.get_code_description(modifier, 'mod')
            modifier_descriptions.append(modifier_desc)
        
        # Create text sequence components
        components = []
        
        # Add procedure information
        if cpt_description and cpt_description != "Unknown":
            components.append(f"procedure description: {cpt_description}")
            
        if cpt_code and cpt_code != "Unknown":
            components.append(f"cpt code: {cpt_code}")
        
        # Add modifier information
        if modifier_descriptions:
            modifier_text = ", ".join(modifier_descriptions)
            components.append(f"modifiers: {modifier_text}")
        
        # Add diagnosis information
        if primary_icd_desc and primary_icd_desc != "Unknown":
            components.append(f"primary diagnosis: {primary_icd_desc}")
            
        if secondary_icd_desc and secondary_icd_desc != "Unknown":
            components.append(f"secondary diagnosis: {secondary_icd_desc}")
        
        # Add place of service
        if pos_desc and pos_desc != "Unknown":
            components.append(f"place of service: {pos_desc}")
        
        # Add revenue code
        if rev_desc and rev_desc != "Unknown":
            components.append(f"revenue_code: {rev_desc}")
            
        # Add drg code
        if drg_desc and rev_desc != "Unknown":
            components.append(f"drg_code: {drg_desc}")
        
        # Add provider specialty
        provider_specialty = row.get('provider_specialty', '')
        if provider_specialty and not pd.isna(provider_specialty):
            components.append(f"provider type: {provider_specialty}")
        
        # Add units if relevant
        units = row.get('units', '')
        if units and not pd.isna(units) and units > 1:
            components.append(f"units: {units}")
        
        # Add charge information for economic anomaly detection
        total_charges = row.get('total_charges_line', '')
        allowed_amount = row.get('allowed_amount', '')
        if total_charges and allowed_amount and not pd.isna(total_charges) and not pd.isna(allowed_amount):
            charge_ratio = float(total_charges) / float(allowed_amount) if float(allowed_amount) > 0 else 0
            components.append(f"charge ratio: {charge_ratio:.2f}")
        
        # Combine all components
        text_content = ". ".join(components)
        
        # Format for BERT with special tokens
        formatted_sequence = f"[CLS] {text_content}. [SEP]"
        
        return formatted_sequence
    
    def prepare_training_data(self, include_features: bool = True) -> pd.DataFrame:
        """
        Prepare complete training dataset with text sequences and labels
        
        Args:
            include_features (bool): Whether to include additional numerical features
            
        Returns:
            pd.DataFrame: Prepared training data
        """
        try:
            # Load data
            df = self.load_claims_data()
            
            if df.empty:
                logger.warning("No data found in database")
                return pd.DataFrame()
            
            # Create text sequences
            logger.info("Creating text sequences...")
            df['input_text'] = df.apply(self.create_text_sequence, axis=1)
            
            # Create labels (combining header and line upcoding indicators)
            df['is_upcoded'] = (
                df['is_upcoded_header'].fillna(False) | 
                df['is_upcoded_line'].fillna(False)
            ).astype(int)
            
            # Create upcoding type (prioritize header, then line)
            df['upcoding_type'] = df['upcoding_type_header'].fillna(df['upcoding_type_line'])
            
            # Prepare base columns
            prepared_df = df[[
                'claim_id', 
                'line_number', 
                'provider_id',
                'input_text', 
                'is_upcoded', 
                'upcoding_type',
                'upcoding_reason'
            ]].copy()
            
            # Add additional features if requested
            if include_features:
                # Numerical features
                prepared_df['units'] = df['units'].fillna(1)
                prepared_df['total_charges'] = df['total_charges_line'].fillna(0)
                prepared_df['allowed_amount'] = df['allowed_amount'].fillna(0)
                
                # Calculate charge ratio
                prepared_df['charge_ratio'] = np.where(
                    prepared_df['allowed_amount'] > 0,
                    prepared_df['total_charges'] / prepared_df['allowed_amount'],
                    0
                )
                
                # Categorical features (encoded)
                prepared_df['claim_type'] = df['claim_type']
                prepared_df['provider_specialty'] = df['provider_specialty']
                prepared_df['place_of_service'] = df['place_of_service']
                
                # Extract CPT code for additional analysis
                cpt_codes = df['hcpcs_code'].apply(lambda x: self.extract_modifiers(x)[0])
                prepared_df['cpt_code'] = cpt_codes
                
                # Count modifiers
                modifier_counts = df['hcpcs_code'].apply(lambda x: len(self.extract_modifiers(x)[1]))
                prepared_df['modifier_count'] = modifier_counts
            
            # Remove rows with empty text sequences
            prepared_df = prepared_df[prepared_df['input_text'].str.len() > 20]
            
            logger.info(f"Prepared {len(prepared_df)} training samples")
            logger.info(f"Upcoded samples: {prepared_df['is_upcoded'].sum()}")
            logger.info(f"Non-upcoded samples: {(1 - prepared_df['is_upcoded']).sum()}")
            
            return prepared_df
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def get_text_statistics(self, prepared_df: pd.DataFrame) -> Dict:
        """
        Get statistics about the prepared text data
        
        Args:
            prepared_df (pd.DataFrame): Prepared training data
            
        Returns:
            Dict: Text statistics
        """
        if prepared_df.empty or 'input_text' not in prepared_df.columns:
            return {}
        
        text_lengths = prepared_df['input_text'].str.len()
        word_counts = prepared_df['input_text'].str.split().str.len()
        
        stats = {
            'total_samples': len(prepared_df),
            'avg_text_length': text_lengths.mean(),
            'max_text_length': text_lengths.max(),
            'min_text_length': text_lengths.min(),
            'avg_word_count': word_counts.mean(),
            'max_word_count': word_counts.max(),
            'min_word_count': word_counts.min(),
            'upcoded_ratio': prepared_df['is_upcoded'].mean()
        }
        
        return stats
    
    def save_prepared_data(self, prepared_df: pd.DataFrame, output_path: str):
        """
        Save prepared data to file
        
        Args:
            prepared_df (pd.DataFrame): Prepared training data
            output_path (str): Output file path
        """
        try:
            if output_path.endswith('.csv'):
                prepared_df.to_csv(output_path, index=False)
            elif output_path.endswith('.parquet'):
                prepared_df.to_parquet(output_path, index=False)
            else:
                prepared_df.to_csv(f"{output_path}.csv", index=False)
            
            logger.info(f"Prepared data saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save prepared data: {e}")
            raise


def main():
    """
    Example usage of the Claims Data Preparation module
    """
    # Initialize data preparation
    data_prep = ClaimsDataPreparation()
    
    try:
        # Prepare training data
        prepared_df = data_prep.prepare_training_data(include_features=True)
        
        if not prepared_df.empty:
            # Display statistics
            stats = data_prep.get_text_statistics(prepared_df)
            print("\n=== Data Preparation Statistics ===")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
            
            # Display sample text sequences
            print("\n=== Sample Text Sequences ===")
            for i, row in prepared_df.head(3).iterrows():
                print(f"\nSample {i+1} (Upcoded: {bool(row['is_upcoded'])}):")
                print(f"Text: {row['input_text']}")
                if row['upcoding_type']:
                    print(f"Upcoding Type: {row['upcoding_type']}")
            
            # Save prepared data
            data_prep.save_prepared_data(prepared_df, os.path.join('Data',"prepared_claims_data.csv"))
            
            print(f"\nData preparation completed successfully!")
            print(f"Total samples: {len(prepared_df)}")
            print(f"Ready for BERT model training.")
        
        else:
            print("No data available for preparation.")
    
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")


if __name__ == "__main__":
    main()
