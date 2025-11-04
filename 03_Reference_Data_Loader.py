"""
Reference Data Loader for Claims Database
Loads medical code reference tables into SQLite database
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Date, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
import os
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()

# Reference table models
class DRGCode(Base):
    """MS-DRG (Medicare Severity Diagnosis Related Groups) codes table"""
    __tablename__ = 'drg_codes'
    
    ms_drg_code = Column(String(10), primary_key=True, nullable=False)
    mdc_code = Column(String(10))
    medical_surgical = Column(String(20))
    ms_drg_description = Column(Text)
    deprecated = Column(Boolean, default=False)
    deprecated_date = Column(Date)

    def __repr__(self):
        return f"<DRGCode(ms_drg_code='{self.ms_drg_code}', description='{self.ms_drg_description[:50]}...')>"


class ICDCode(Base):
    """ICD (International Classification of Diseases) codes table"""
    __tablename__ = 'icd_codes'
    
    icd_code = Column(String(20), primary_key=True, nullable=False)
    description = Column(Text)

    def __repr__(self):
        return f"<ICDCode(icd_code='{self.icd_code}', description='{self.description[:50]}...')>"


class ModifierCode(Base):
    """CPT/HCPCS modifier codes table"""
    __tablename__ = 'modifier_codes'
    
    modifier = Column(String(10), primary_key=True, nullable=False)
    short_description = Column(Text)

    def __repr__(self):
        return f"<ModifierCode(modifier='{self.modifier}', description='{self.short_description}')>"


class PlaceOfServiceCode(Base):
    """Place of Service (POS) codes table"""
    __tablename__ = 'pos_codes'
    
    pos_code = Column(String(10), primary_key=True, nullable=False)
    description = Column(Text)

    def __repr__(self):
        return f"<PlaceOfServiceCode(pos_code='{self.pos_code}', description='{self.description}')>"


class RevenueCode(Base):
    """Revenue codes table for hospital billing"""
    __tablename__ = 'revenue_codes'
    
    revenue_code = Column(String(10), primary_key=True, nullable=False)
    short_description = Column(Text)
    long_description = Column(Text)

    def __repr__(self):
        return f"<RevenueCode(revenue_code='{self.revenue_code}', short_desc='{self.short_description}')>"


class ReferenceDataLoader:
    """
    Loader class for medical reference data into SQLite database
    """
    
    def __init__(self, db_path: str = os.path.join('Data',"claims_database.db")):
        """
        Initialize the reference data loader
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        # Create all reference tables
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info(f"Reference data loader initialized for database: {db_path}")
    
    def _clean_string_value(self, value) -> Optional[str]:
        """Clean and validate string values"""
        if pd.isna(value) or value is None:
            return None
        
        # Convert to string and strip whitespace
        cleaned = str(value).strip()
        
        # Return None if empty after cleaning
        return cleaned if cleaned else None
    
    def _format_pos_code(self, value) -> Optional[str]:
        """Format POS code as two-digit alphanumeric"""
        if pd.isna(value) or value is None:
            return None
        
        # Convert to string and strip whitespace
        pos_code = str(value).strip()
        
        if not pos_code:
            return None
        
        # If single digit numeric, prepend zero
        if pos_code.isdigit() and len(pos_code) == 1:
            return f"0{pos_code}"
        
        # Return as-is for other formats (already two digits or alphanumeric)
        return pos_code
    
    def _format_revenue_code(self, value) -> Optional[str]:
        """Format revenue code as four-digit code"""
        if pd.isna(value) or value is None:
            return None
        
        # Convert to string and strip whitespace
        revenue_code = str(value).strip()
        
        if not revenue_code:
            return None
        
        # If numeric, pad with leading zeros to make it 4 digits
        if revenue_code.isdigit():
            return revenue_code.zfill(4)
        
        # Return as-is for non-numeric codes
        return revenue_code
    
    def _format_drg_code(self, value) -> Optional[str]:
        """Format DRG code as three-digit code"""
        if pd.isna(value) or value is None:
            return None
        
        # Convert to string and strip whitespace
        drg_code = str(value).strip()
        
        if not drg_code:
            return None
        
        # If numeric, pad with leading zeros to make it 3 digits
        if drg_code.isdigit():
            return drg_code.zfill(3)
        
        # Return as-is for non-numeric codes
        return drg_code
    
    def _clean_boolean_value(self, value) -> Optional[bool]:
        """Clean and convert boolean values"""
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
    
    def _parse_date(self, date_str) -> Optional[datetime]:
        """Parse date string to datetime object"""
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
    
    def load_drg_codes(self, file_path: str = os.path.join('Reference_Files',"drg.csv")) -> int:
        """
        Load DRG codes from CSV file
        
        Args:
            file_path (str): Path to DRG CSV file
            
        Returns:
            int: Number of records loaded
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"DRG file not found: {file_path}")
                return 0
            
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Read {len(df)} DRG records from {file_path}")
            
            # Expected columns: ms_drg_code, mdc_code, medical_surgical, ms_drg_description, deprecated, deprecated_date
            records_loaded = 0
            
            for _, row in df.iterrows():
                try:
                    drg_record = DRGCode(
                        ms_drg_code=self._format_drg_code(row.get('ms_drg_code')),
                        mdc_code=self._clean_string_value(row.get('mdc_code')),
                        medical_surgical=self._clean_string_value(row.get('medical_surgical')),
                        ms_drg_description=self._clean_string_value(row.get('ms_drg_description')),
                        deprecated=self._clean_boolean_value(row.get('deprecated', False)),
                        deprecated_date=self._parse_date(row.get('deprecated_date'))
                    )
                    
                    # Skip records without primary key
                    if not drg_record.ms_drg_code:
                        continue
                    
                    # Use merge to handle duplicates
                    existing = self.session.query(DRGCode).filter_by(ms_drg_code=drg_record.ms_drg_code).first()
                    if existing:
                        # Update existing record
                        existing.mdc_code = drg_record.mdc_code
                        existing.medical_surgical = drg_record.medical_surgical
                        existing.ms_drg_description = drg_record.ms_drg_description
                        existing.deprecated = drg_record.deprecated
                        existing.deprecated_date = drg_record.deprecated_date
                    else:
                        self.session.add(drg_record)
                    
                    records_loaded += 1
                    
                except Exception as e:
                    logger.error(f"Error processing DRG record: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"Successfully loaded {records_loaded} DRG records")
            return records_loaded
            
        except Exception as e:
            logger.error(f"Failed to load DRG codes: {e}")
            self.session.rollback()
            return 0
    
    def load_icd_codes(self, file_path: str = os.path.join('Reference_Files',"icd.csv")) -> int:
        """
        Load ICD codes from CSV file
        
        Args:
            file_path (str): Path to ICD CSV file
            
        Returns:
            int: Number of records loaded
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"ICD file not found: {file_path}")
                return 0
            
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Read {len(df)} ICD records from {file_path}")
            
            # Expected columns: ICD, Description
            records_loaded = 0
            
            for _, row in df.iterrows():
                try:
                    icd_record = ICDCode(
                        icd_code=self._clean_string_value(row.get('ICD')),
                        description=self._clean_string_value(row.get('Description'))
                    )
                    
                    # Skip records without primary key
                    if not icd_record.icd_code:
                        continue
                    
                    # Use merge to handle duplicates
                    existing = self.session.query(ICDCode).filter_by(icd_code=icd_record.icd_code).first()
                    if existing:
                        existing.description = icd_record.description
                    else:
                        self.session.add(icd_record)
                    
                    records_loaded += 1
                    
                except Exception as e:
                    logger.error(f"Error processing ICD record: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"Successfully loaded {records_loaded} ICD records")
            return records_loaded
            
        except Exception as e:
            logger.error(f"Failed to load ICD codes: {e}")
            self.session.rollback()
            return 0
    
    def load_modifier_codes(self, file_path: str = os.path.join('Reference_Files',"modifier_table.csv")) -> int:
        """
        Load modifier codes from CSV file
        
        Args:
            file_path (str): Path to modifier CSV file
            
        Returns:
            int: Number of records loaded
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Modifier file not found: {file_path}")
                return 0
            
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Read {len(df)} modifier records from {file_path}")
            
            # Expected columns: modifier, short_description
            records_loaded = 0
            
            for _, row in df.iterrows():
                try:
                    modifier_record = ModifierCode(
                        modifier=self._clean_string_value(row.get('modifier')),
                        short_description=self._clean_string_value(row.get('short_description'))
                    )
                    
                    # Skip records without primary key
                    if not modifier_record.modifier:
                        continue
                    
                    # Use merge to handle duplicates
                    existing = self.session.query(ModifierCode).filter_by(modifier=modifier_record.modifier).first()
                    if existing:
                        existing.short_description = modifier_record.short_description
                    else:
                        self.session.add(modifier_record)
                    
                    records_loaded += 1
                    
                except Exception as e:
                    logger.error(f"Error processing modifier record: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"Successfully loaded {records_loaded} modifier records")
            return records_loaded
            
        except Exception as e:
            logger.error(f"Failed to load modifier codes: {e}")
            self.session.rollback()
            return 0
    
    def load_pos_codes(self, file_path: str = os.path.join('Reference_Files',"pos.csv")) -> int:
        """
        Load Place of Service codes from CSV file
        
        Args:
            file_path (str): Path to POS CSV file
            
        Returns:
            int: Number of records loaded
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"POS file not found: {file_path}")
                return 0
            
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Read {len(df)} POS records from {file_path}")
            
            # Expected columns: pos, Description
            records_loaded = 0
            
            for _, row in df.iterrows():
                try:
                    pos_record = PlaceOfServiceCode(
                        pos_code=self._format_pos_code(row.get('pos')),
                        description=self._clean_string_value(row.get('Description'))
                    )
                    
                    # Skip records without primary key
                    if not pos_record.pos_code:
                        continue
                    
                    # Use merge to handle duplicates
                    existing = self.session.query(PlaceOfServiceCode).filter_by(pos_code=pos_record.pos_code).first()
                    if existing:
                        existing.description = pos_record.description
                    else:
                        self.session.add(pos_record)
                    
                    records_loaded += 1
                    
                except Exception as e:
                    logger.error(f"Error processing POS record: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"Successfully loaded {records_loaded} POS records")
            return records_loaded
            
        except Exception as e:
            logger.error(f"Failed to load POS codes: {e}")
            self.session.rollback()
            return 0
    
    def load_revenue_codes(self, file_path: str = os.path.join('Reference_Files',"revenue_code.csv")) -> int:
        """
        Load Revenue codes from CSV file
        
        Args:
            file_path (str): Path to Revenue code CSV file
            
        Returns:
            int: Number of records loaded
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Revenue code file not found: {file_path}")
                return 0
            
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Read {len(df)} revenue code records from {file_path}")
            
            # Expected columns: Revenue Code, Short Description, Long Description
            records_loaded = 0
            
            for _, row in df.iterrows():
                try:
                    revenue_record = RevenueCode(
                        revenue_code=self._format_revenue_code(row.get('Revenue Code')),
                        short_description=self._clean_string_value(row.get('Short Description')),
                        long_description=self._clean_string_value(row.get('Long Description'))
                    )
                    
                    # Skip records without primary key
                    if not revenue_record.revenue_code:
                        continue
                    
                    # Use merge to handle duplicates
                    existing = self.session.query(RevenueCode).filter_by(revenue_code=revenue_record.revenue_code).first()
                    if existing:
                        existing.short_description = revenue_record.short_description
                        existing.long_description = revenue_record.long_description
                    else:
                        self.session.add(revenue_record)
                    
                    records_loaded += 1
                    
                except Exception as e:
                    logger.error(f"Error processing revenue code record: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"Successfully loaded {records_loaded} revenue code records")
            return records_loaded
            
        except Exception as e:
            logger.error(f"Failed to load revenue codes: {e}")
            self.session.rollback()
            return 0
    
    def load_all_reference_data(self, data_directory: str = ".") -> Dict[str, int]:
        """
        Load all reference data files from a directory
        
        Args:
            data_directory (str): Directory containing CSV files
            
        Returns:
            Dict[str, int]: Summary of records loaded for each table
        """
        results = {}
        
        # Define file mappings
        file_mappings = {
            'drg_codes': os.path.join('Reference_Files','drg.csv'),
            'icd_codes': os.path.join('Reference_Files','icd.csv'),
            'modifier_codes': os.path.join('Reference_Files','modifier_table.csv'),
            'pos_codes': os.path.join('Reference_Files','pos.csv'),
            'revenue_codes': os.path.join('Reference_Files','revenue_code.csv')
        }
        
        # Load each reference table
        for table_name, filename in file_mappings.items():
            file_path = os.path.join(data_directory, filename)
            
            if table_name == 'drg_codes':
                results[table_name] = self.load_drg_codes(file_path)
            elif table_name == 'icd_codes':
                results[table_name] = self.load_icd_codes(file_path)
            elif table_name == 'modifier_codes':
                results[table_name] = self.load_modifier_codes(file_path)
            elif table_name == 'pos_codes':
                results[table_name] = self.load_pos_codes(file_path)
            elif table_name == 'revenue_codes':
                results[table_name] = self.load_revenue_codes(file_path)
        
        return results
    
    def get_table_counts(self) -> Dict[str, int]:
        """
        Get record counts for all reference tables
        
        Returns:
            Dict[str, int]: Record counts for each table
        """
        try:
            counts = {}
            counts['drg_codes'] = self.session.query(DRGCode).count()
            counts['icd_codes'] = self.session.query(ICDCode).count()
            counts['modifier_codes'] = self.session.query(ModifierCode).count()
            counts['pos_codes'] = self.session.query(PlaceOfServiceCode).count()
            counts['revenue_codes'] = self.session.query(RevenueCode).count()
            
            return counts
            
        except Exception as e:
            logger.error(f"Failed to get table counts: {e}")
            return {}
    
    def get_sample_records(self, table_name: str, limit: int = 5) -> List:
        """
        Get sample records from a reference table
        
        Args:
            table_name (str): Name of the table
            limit (int): Number of sample records
            
        Returns:
            List: Sample records
        """
        try:
            table_mapping = {
                'drg_codes': DRGCode,
                'icd_codes': ICDCode,
                'modifier_codes': ModifierCode,
                'pos_codes': PlaceOfServiceCode,
                'revenue_codes': RevenueCode
            }
            
            model_class = table_mapping.get(table_name)
            if not model_class:
                return []
            
            records = self.session.query(model_class).limit(limit).all()
            return records
            
        except Exception as e:
            logger.error(f"Failed to get sample records for {table_name}: {e}")
            return []
    
    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()
            logger.info("Database session closed")


def main():
    """
    Main function to load all reference data
    """
    # Initialize the loader
    loader = ReferenceDataLoader(os.path.join('Data',"claims_database.db"))
    
    try:
        print("Starting reference data loading...")
        
        # Load all reference data
        results = loader.load_all_reference_data(".")
        
        # Display results
        print("\n=== Reference Data Loading Results ===")
        total_loaded = 0
        for table_name, count in results.items():
            print(f"{table_name}: {count} records")
            total_loaded += count
        
        print(f"\nTotal records loaded: {total_loaded}")
        
        # Get final table counts
        print("\n=== Current Table Counts ===")
        counts = loader.get_table_counts()
        for table_name, count in counts.items():
            print(f"{table_name}: {count} records")
        
        # Display sample records
        print("\n=== Sample Records ===")
        for table_name in ['icd_codes', 'modifier_codes', 'pos_codes']:
            print(f"\n{table_name.upper()}:")
            samples = loader.get_sample_records(table_name, 3)
            for sample in samples:
                print(f"  {sample}")
        
        print("\nReference data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Reference data loading failed: {e}")
        
    finally:
        loader.close()


if __name__ == "__main__":
    main()