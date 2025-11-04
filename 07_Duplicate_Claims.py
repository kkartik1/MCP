import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                patient_id,
                provider_npi,
                service_date_line,
                hcpcs_code
            FROM claims_data
            """
            
            df = pd.read_sql_query(query, self.engine)
            logger.info(f"Loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise


# --- MODIFY COLUMN NAMES BELOW AS PER YOUR CSV STRUCTURE ---
# Expecting df to have columns 'claim_number' and 'line_number' in addition to previous fields

try:
    # Load data
    # Initialize data preparation
    data_prep = ClaimsDataPreparation()
    df = data_prep.load_claims_data()
    
    if df.empty:
        logger.warning("No data found in database")
    else:
        df.to_csv("claims_data.csv", index=False)
           
except Exception as e:
    logger.error(f"Failed to prepare data: {e}")
    raise

print(df.columns)

train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # You can add stratification if needed
        )
        
# Reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
        
logger.info(f"Split data: {len(train_df)} training samples, {len(test_df)} test samples")

# Create a text representation of each claim (combining the four key fields)
train_df['claim_text'] = (
    train_df['patient_id'].astype(str) + ' ' +
    train_df['provider_npi'].astype(str) + ' ' +
    train_df['service_date_line'].astype(str) + ' ' +
    train_df['hcpcs_code'].astype(str)
)

# Load transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode claims to get embeddings
embeddings = model.encode(train_df['claim_text'].tolist(), show_progress_bar=True)

file_path="embeddings1.npy"
np.save(file_path, embeddings1)
print(f"embeddings1 stored at {file_path}")

def check_duplicate(embeddings1, embeddings2):
    # Compute cosine similarity between embeddings1 and embeddings2
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    return similarity_matrix
# Set a threshold for duplicates
SIM_THRESHOLD = 0.75
    
# Find pairs of claims above the similarity threshold
duplicate_rows = []
n = len(train_df)
m = len(test_df)

for j in range(m):
    claim_text = (
        test_df.loc[j,'patient_id'] + ' ' +
        test_df.loc[j,'provider_npi'].astype(str)  + ' ' +
        test_df.loc[j,'service_date_line'] + ' ' +
        test_df.loc[j,'procedure_description']
    )

    embeddings2 = model.encode(claim_text, show_progress_bar=True)
    similarity_matrix = check_duplicate(embeddings1, embeddings2)
    for i in range(n):
        if similarity_matrix[i, 0] >= SIM_THRESHOLD:
            duplicate_rows.append({
                "source_index": i,
                "source_claim_number": train_df.loc[i, 'claim_number'],
                "source_line_number": train_df.loc[i, 'line_number'],
                "target_index": j,
                "target_claim_number": test_df.loc[j, 'claim_number'],
                "target_line_number": test_df.loc[j, 'line_number'],
                "similarity": similarity_matrix[i, j],
                })

# Create the DataFrame
duplicates_df = pd.DataFrame(duplicate_rows)

if not duplicates_df.empty:
    print(duplicates_df)
else:
    print("No duplicates found at the chosen threshold.")

# Optionally, save to CSV:
duplicates_df.to_csv("duplicates_pairs.csv", index=False)