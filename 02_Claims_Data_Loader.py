"""
Claims Data Ingestion Module
Combines claims header and line data into a single SQLite table using SQLAlchemy
"""

import pandas as pd
from sqlalchemy import create_engine, Column, String, Integer, Float, Date, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import DECIMAL
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()

class ClaimsData(Base):
    """
    Combined claims table with composite primary key (claim_id, line_number)
    Merges header and line data into a single denormalized table
    """
    __tablename__ = 'claims_data'
    
    # Composite Primary Key
    claim_id = Column(String(50), primary_key=True, nullable=False)
    line_number = Column(Integer, primary_key=True, nullable=False)
    
    # Header data fields
    claim_type = Column(String(50))
    patient_id = Column(String(50))
    member_id = Column(String(50))
    provider_id = Column(String(50))
    provider_npi = Column(String(20))
    provider_specialty = Column(String(100))
    place_of_service = Column(String(50))
    service_date_header = Column(Date)
    primary_diagnosis = Column(String(20))
    secondary_diagnosis = Column(String(20))
    total_charges_header = Column(DECIMAL(10, 2))
    expected_payment = Column(DECIMAL(10, 2))
    is_upcoded_header = Column(Boolean)
    upcoding_type_header = Column(String(50))
    upcoding_reason = Column(String(200))
    admission_date = Column(Date)
    discharge_date = Column(Date)
    length_of_stay = Column(Integer)
    drg_code = Column(String(20))
    
    # Line data fields
    revenue_code = Column(String(20))
    hcpcs_code = Column(String(20))
    procedure_description = Column(String(200))
    service_date_line = Column(Date)
    units = Column(Integer)
    unit_cost = Column(DECIMAL(10, 2))
    total_charges_line = Column(DECIMAL(10, 2))
    allowed_amount = Column(DECIMAL(10, 2))
    is_upcoded_line = Column(Boolean)
    upcoding_type_line = Column(String(50))

    def __repr__(self):
        return f"<ClaimsData(claim_id='{self.claim_id}', line_number={self.line_number})>"


class ClaimsDataIngestion:
    """
    Data ingestion class for processing and storing claims data
    """
    
    def __init__(self, db_path: str = "claims_data.db"):
        """
        Initialize the ingestion module
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info(f"Database initialized at: {db_path}")
    
    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """
        Parse date string to datetime object
        
        Args:
            date_str: Date string or datetime object
            
        Returns:
            datetime object or None if parsing fails
        """
        if pd.isna(date_str) or date_str is None:
            return None
            
        if isinstance(date_str, datetime):
            return date_str.date()
            
        try:
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(str(date_str), fmt).date()
                except ValueError:
                    continue
            return None
        except Exception as e:
            logger.warning(f"Failed to parse date: {date_str}, Error: {e}")
            return None
    
    def _clean_numeric_value(self, value: Any) -> Optional[float]:
        """
        Clean and convert numeric values
        
        Args:
            value: Numeric value to clean
            
        Returns:
            Float value or None if conversion fails
        """
        if pd.isna(value) or value is None:
            return None
            
        try:
            # Remove currency symbols and commas
            if isinstance(value, str):
                value = value.replace('$', '').replace(',', '').strip()
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _clean_boolean_value(self, value: Any) -> Optional[bool]:
        """
        Clean and convert boolean values
        
        Args:
            value: Boolean value to clean
            
        Returns:
            Boolean value or None if conversion fails
        """
        if pd.isna(value) or value is None:
            return None
            
        if isinstance(value, bool):
            return value
            
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ['true', '1', 'yes', 'y']:
                return True
            elif value in ['false', '0', 'no', 'n']:
                return False
                
        try:
            return bool(int(value))
        except (ValueError, TypeError):
            return None
    
    def load_claims_header(self, file_path: str, file_format: str = 'csv') -> pd.DataFrame:
        """
        Load claims header data from file
        
        Args:
            file_path (str): Path to header data file
            file_format (str): File format ('csv', 'excel', 'json')
            
        Returns:
            pandas DataFrame with header data
        """
        try:
            if file_format.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_format.lower() in ['excel', 'xlsx']:
                df = pd.read_excel(file_path)
            elif file_format.lower() == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Loaded {len(df)} header records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load header data: {e}")
            raise
    
    def load_claims_line(self, file_path: str, file_format: str = 'csv') -> pd.DataFrame:
        """
        Load claims line data from file
        
        Args:
            file_path (str): Path to line data file
            file_format (str): File format ('csv', 'excel', 'json')
            
        Returns:
            pandas DataFrame with line data
        """
        try:
            if file_format.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_format.lower() in ['excel', 'xlsx']:
                df = pd.read_excel(file_path)
            elif file_format.lower() == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Loaded {len(df)} line records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load line data: {e}")
            raise
    
    def merge_and_ingest_data(self, header_df: pd.DataFrame, line_df: pd.DataFrame, 
                             batch_size: int = 1000) -> int:
        """
        Merge header and line data, then ingest into database
        
        Args:
            header_df (pd.DataFrame): Claims header data
            line_df (pd.DataFrame): Claims line data
            batch_size (int): Number of records to process in each batch
            
        Returns:
            int: Number of records successfully ingested
        """
        try:
            # Merge header and line data on claim_id
            merged_df = pd.merge(
                header_df, 
                line_df, 
                on='claim_id', 
                how='inner',
                suffixes=('_header', '_line')
            )
            
            logger.info(f"Merged data: {len(merged_df)} records")
            
            if merged_df.empty:
                logger.warning("No matching records found between header and line data")
                return 0
            
            # Process data in batches
            total_inserted = 0
            
            for start_idx in range(0, len(merged_df), batch_size):
                end_idx = min(start_idx + batch_size, len(merged_df))
                batch_df = merged_df.iloc[start_idx:end_idx]
                
                batch_records = []
                
                for _, row in batch_df.iterrows():
                    # Create ClaimsData object with cleaned data
                    claims_record = ClaimsData(
                        claim_id=str(row.get('claim_id', '')),
                        line_number=int(row.get('line_number', 0)),
                        
                        # Header fields
                        claim_type=row.get('claim_type'),
                        patient_id=row.get('patient_id'),
                        member_id=row.get('member_id'),
                        provider_id=row.get('provider_id'),
                        provider_npi=row.get('provider_npi'),
                        provider_specialty=row.get('provider_specialty'),
                        place_of_service=row.get('place_of_service'),
                        service_date_header=self._parse_date(row.get('service_date_header', row.get('service_date'))),
                        primary_diagnosis=row.get('primary_diagnosis'),
                        secondary_diagnosis=row.get('secondary_diagnosis'),
                        total_charges_header=self._clean_numeric_value(row.get('total_charges_header', row.get('total_charges'))),
                        expected_payment=self._clean_numeric_value(row.get('expected_payment')),
                        is_upcoded_header=self._clean_boolean_value(row.get('is_upcoded_header', row.get('is_upcoded'))),
                        upcoding_type_header=row.get('upcoding_type_header', row.get('upcoding_type')),
                        upcoding_reason=row.get('upcoding_reason'),
                        admission_date=self._parse_date(row.get('admission_date')),
                        discharge_date=self._parse_date(row.get('discharge_date')),
                        length_of_stay=row.get('length_of_stay'),
                        drg_code=row.get('drg_code'),
                        
                        # Line fields
                        revenue_code=row.get('revenue_code'),
                        hcpcs_code=row.get('hcpcs_code'),
                        procedure_description=row.get('procedure_description'),
                        service_date_line=self._parse_date(row.get('service_date_line', row.get('service_date'))),
                        units=row.get('units'),
                        unit_cost=self._clean_numeric_value(row.get('unit_cost')),
                        total_charges_line=self._clean_numeric_value(row.get('total_charges_line', row.get('total_charges'))),
                        allowed_amount=self._clean_numeric_value(row.get('allowed_amount')),
                        is_upcoded_line=self._clean_boolean_value(row.get('is_upcoded_line', row.get('is_upcoded'))),
                        upcoding_type_line=row.get('upcoding_type_line', row.get('upcoding_type'))
                    )
                    
                    batch_records.append(claims_record)
                
                # Insert batch
                try:
                    self.session.add_all(batch_records)
                    self.session.commit()
                    total_inserted += len(batch_records)
                    logger.info(f"Inserted batch {start_idx//batch_size + 1}: {len(batch_records)} records")
                    
                except Exception as e:
                    self.session.rollback()
                    logger.error(f"Failed to insert batch {start_idx//batch_size + 1}: {e}")
                    
                    # Try inserting records individually to identify problematic records
                    for record in batch_records:
                        try:
                            self.session.add(record)
                            self.session.commit()
                            total_inserted += 1
                        except Exception as individual_error:
                            self.session.rollback()
                            logger.error(f"Failed to insert record {record.claim_id}-{record.line_number}: {individual_error}")
            
            logger.info(f"Successfully ingested {total_inserted} records")
            return total_inserted
            
        except Exception as e:
            logger.error(f"Failed to merge and ingest data: {e}")
            self.session.rollback()
            raise
    
    def get_record_count(self) -> int:
        """
        Get total number of records in the database
        
        Returns:
            int: Total record count
        """
        try:
            count = self.session.query(ClaimsData).count()
            return count
        except Exception as e:
            logger.error(f"Failed to get record count: {e}")
            return 0
    
    def get_sample_records(self, limit: int = 10) -> list:
        """
        Get sample records from the database
        
        Args:
            limit (int): Number of sample records to retrieve
            
        Returns:
            list: Sample records
        """
        try:
            records = self.session.query(ClaimsData).limit(limit).all()
            return records
        except Exception as e:
            logger.error(f"Failed to get sample records: {e}")
            return []
    
    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()
            logger.info("Database session closed")


def main():
    """
    Example usage of the Claims Data Ingestion module
    """
    # Initialize ingestion module
    database_path = os.path.join('Data', 'claims_database.db')
    ingestion = ClaimsDataIngestion(database_path)
    
    try:
        # Example: Load data from CSV files
        # header_df = ingestion.load_claims_header("claims_header.csv", "csv")
        # line_df = ingestion.load_claims_line("claims_line.csv", "csv")
        csv_path = os.path.join('Data','healthcare_claims.csv')
        header_data = pd.read_csv(
            csv_path,
            dtype=str,  # Read all as strings initially for better control
            skip_blank_lines=True,
            na_values=['', 'NULL', 'null', 'N/A', 'n/a'],
            keep_default_na=False
        )
        
        logger.info(f"Successfully loaded CSV with {len(header_data)} records from {csv_path}")
        
        csv_path = os.path.join('Data','claim_line_items.csv')
        line_data = pd.read_csv(
           csv_path,
           dtype=str,  # Read all as strings initially for better control
           skip_blank_lines=True,
           na_values=['', 'NULL', 'null', 'N/A', 'n/a'],
           keep_default_na=False
        )
       
        logger.info(f"Successfully loaded CSV with {len(line_data)} records from {csv_path}")
        
        header_df = pd.DataFrame(header_data)
        line_df = pd.DataFrame(line_data)
        
        # Ingest data
        records_inserted = ingestion.merge_and_ingest_data(header_df, line_df)
        
        print(f"\nIngestion completed successfully!")
        print(f"Records inserted: {records_inserted}")
        print(f"Total records in database: {ingestion.get_record_count()}")
        
        # Display sample records
        print("\nSample records:")
        sample_records = ingestion.get_sample_records(3)
        for record in sample_records:
            print(f"Claim ID: {record.claim_id}, Line: {record.line_number}, "
                  f"Patient: {record.patient_id}, Procedure: {record.procedure_description}")
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
    
    finally:
        ingestion.close()


if __name__ == "__main__":
    main()