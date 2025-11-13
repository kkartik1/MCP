import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

class HealthcareClaimsAnalyzer:
    """
    Comprehensive descriptive analysis for healthcare claims data
    focusing on upcoding detection patterns
    """
    
    def __init__(self, db_path, output_dir='analysis_output'):
        """Initialize with database connection and output directory"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.df = None
        self.output_dir = output_dir
        
        # Create output directories
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(f"{self.output_dir}/visualizations").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/csv_outputs").mkdir(exist_ok=True)
        
        print(f"Output directory created: {self.output_dir}")
        
    def load_data(self, table_name='claims'):
        """Load data from SQLite database"""
        query = f"SELECT * FROM {table_name}"
        self.df = pd.read_sql_query(query, self.conn)
        
        # Convert date columns
        date_cols = ['service_date_header', 'admission_date', 'discharge_date', 'service_date_line']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Convert boolean columns (handle both TRUE/FALSE and 1/0)
        bool_cols = ['is_upcoded_header', 'is_upcoded_line']
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(bool)
        
        print(f"Data loaded successfully: {len(self.df)} records")
        return self.df
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        print("\n" + "="*80)
        print("1. BASIC DATA STATISTICS")
        print("="*80)
        
        print(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        print("\n--- Missing Values Analysis ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Data types
        print("\n--- Data Types ---")
        print(self.df.dtypes.value_counts())
        
        return missing_df
    
    def claims_overview(self):
        """Analyze claims-level characteristics"""
        print("\n" + "="*80)
        print("2. CLAIMS OVERVIEW")
        print("="*80)
        
        # Unique counts
        print("\n--- Unique Entity Counts ---")
        entities = {
            'Unique Claims': self.df['claim_id'].nunique(),
            'Unique Patients': self.df['patient_id'].nunique(),
            'Unique Members': self.df['member_id'].nunique(),
            'Unique Providers': self.df['provider_id'].nunique(),
            'Unique Provider NPIs': self.df['provider_npi'].nunique()
        }
        for k, v in entities.items():
            print(f"{k}: {v:,}")
        
        # Claims per patient
        claims_per_patient = self.df.groupby('patient_id')['claim_id'].nunique()
        print(f"\n--- Claims Distribution per Patient ---")
        print(f"Mean: {claims_per_patient.mean():.2f}")
        print(f"Median: {claims_per_patient.median():.2f}")
        print(f"Max: {claims_per_patient.max()}")
        print(f"Patients with >10 claims: {(claims_per_patient > 10).sum():,}")
        
        # Line items per claim
        lines_per_claim = self.df.groupby('claim_id')['line_number'].count()
        print(f"\n--- Line Items per Claim ---")
        print(f"Mean: {lines_per_claim.mean():.2f}")
        print(f"Median: {lines_per_claim.median():.2f}")
        print(f"Max: {lines_per_claim.max()}")
        
        return entities, claims_per_patient, lines_per_claim
    
    def financial_analysis(self):
        """Analyze financial metrics"""
        print("\n" + "="*80)
        print("3. FINANCIAL ANALYSIS")
        print("="*80)
        
        # Header-level charges
        print("\n--- Total Charges (Header Level) ---")
        print(self.df['total_charges_header'].describe())
        print(f"Total Charges Sum: ${self.df['total_charges_header'].sum():,.2f}")
        
        # Line-level charges
        print("\n--- Total Charges (Line Level) ---")
        print(self.df['total_charges_line'].describe())
        print(f"Total Charges Sum: ${self.df['total_charges_line'].sum():,.2f}")
        
        # Expected payment vs allowed amount
        print("\n--- Payment Analysis ---")
        print(f"Total Expected Payment: ${self.df['expected_payment'].sum():,.2f}")
        print(f"Total Allowed Amount: ${self.df['allowed_amount'].sum():,.2f}")
        
        # Cost per unit
        self.df['cost_per_unit'] = self.df['total_charges_line'] / self.df['units'].replace(0, np.nan)
        print(f"\n--- Cost per Unit ---")
        print(self.df['cost_per_unit'].describe())
        
        # Charge-to-payment ratio
        self.df['charge_to_payment_ratio'] = (
            self.df['total_charges_header'] / self.df['expected_payment'].replace(0, np.nan)
        )
        print(f"\n--- Charge-to-Payment Ratio ---")
        print(self.df['charge_to_payment_ratio'].describe())
        
        return self.df[['total_charges_header', 'expected_payment', 'allowed_amount']].describe()
    
    def upcoding_analysis(self):
        """Detailed upcoding analysis - PRIMARY FOCUS"""
        print("\n" + "="*80)
        print("4. UPCODING ANALYSIS (CRITICAL)")
        print("="*80)
        
        # Header-level upcoding
        upcoded_header = self.df['is_upcoded_header'].sum()
        upcoded_header_pct = (upcoded_header / len(self.df) * 100)
        
        print(f"\n--- Header-Level Upcoding ---")
        print(f"Upcoded Claims: {upcoded_header:,} ({upcoded_header_pct:.2f}%)")
        print(f"Clean Claims: {(~self.df['is_upcoded_header']).sum():,}")
        
        # Line-level upcoding
        upcoded_line = self.df['is_upcoded_line'].sum()
        upcoded_line_pct = (upcoded_line / len(self.df) * 100)
        
        print(f"\n--- Line-Level Upcoding ---")
        print(f"Upcoded Lines: {upcoded_line:,} ({upcoded_line_pct:.2f}%)")
        print(f"Clean Lines: {(~self.df['is_upcoded_line']).sum():,}")
        
        # Upcoding types (header)
        print(f"\n--- Upcoding Types (Header) ---")
        print(self.df['upcoding_type_header'].value_counts(dropna=False))
        
        # Upcoding types (line)
        print(f"\n--- Upcoding Types (Line) ---")
        print(self.df['upcoding_type_line'].value_counts(dropna=False))
        
        # Upcoding reasons
        print(f"\n--- Top 10 Upcoding Reasons ---")
        print(self.df['upcoding_reason'].value_counts().head(10))
        
        # Financial impact of upcoding - use boolean indexing properly
        upcoded_mask = self.df['is_upcoded_header'] == True
        clean_mask = self.df['is_upcoded_header'] == False
        
        upcoded_charges = self.df.loc[upcoded_mask, 'total_charges_header'].sum()
        clean_charges = self.df.loc[clean_mask, 'total_charges_header'].sum()
        
        print(f"\n--- Financial Impact ---")
        print(f"Charges from Upcoded Claims: ${upcoded_charges:,.2f}")
        print(f"Charges from Clean Claims: ${clean_charges:,.2f}")
        
        if upcoded_header > 0:
            print(f"Avg Charge (Upcoded): ${self.df.loc[upcoded_mask, 'total_charges_header'].mean():,.2f}")
        if clean_mask.sum() > 0:
            print(f"Avg Charge (Clean): ${self.df.loc[clean_mask, 'total_charges_header'].mean():,.2f}")
        
        return {
            'upcoded_header_pct': upcoded_header_pct,
            'upcoded_line_pct': upcoded_line_pct,
            'financial_impact': upcoded_charges
        }
    
    def provider_analysis(self):
        """Analyze provider-related patterns"""
        print("\n" + "="*80)
        print("5. PROVIDER ANALYSIS")
        print("="*80)
        
        # Provider specialty distribution
        print("\n--- Provider Specialty Distribution ---")
        specialty_dist = self.df['provider_specialty'].value_counts().head(15)
        print(specialty_dist)
        
        # Upcoding by specialty
        print("\n--- Upcoding Rate by Specialty (Top 10) ---")
        specialty_upcode = self.df.groupby('provider_specialty').agg({
            'is_upcoded_header': ['sum', 'count', 'mean']
        })
        specialty_upcode.columns = ['Upcoded', 'Total', 'Rate']
        specialty_upcode['Rate'] = (specialty_upcode['Rate'] * 100).round(2)
        specialty_upcode = specialty_upcode.sort_values('Rate', ascending=False)
        print(specialty_upcode.head(10))
        
        # Provider volume analysis
        provider_volume = self.df.groupby('provider_id').agg({
            'claim_id': 'count',
            'is_upcoded_header': 'sum',
            'total_charges_header': 'sum'
        })
        provider_volume.columns = ['Claims', 'Upcoded', 'Total_Charges']
        provider_volume['Upcode_Rate'] = (provider_volume['Upcoded'] / provider_volume['Claims'] * 100).round(2)
        
        print(f"\n--- High-Volume Providers (Top 10) ---")
        print(provider_volume.nlargest(10, 'Claims'))
        
        print(f"\n--- High Upcoding Rate Providers (min 10 claims) ---")
        high_upcoders = provider_volume[provider_volume['Claims'] >= 10].nlargest(10, 'Upcode_Rate')
        print(high_upcoders)
        
        return specialty_upcode, provider_volume
    
    def service_analysis(self):
        """Analyze service-related characteristics"""
        print("\n" + "="*80)
        print("6. SERVICE & CLINICAL ANALYSIS")
        print("="*80)
        
        # Place of service
        print("\n--- Place of Service Distribution ---")
        print(self.df['place_of_service'].value_counts())
        
        # Claim type
        print("\n--- Claim Type Distribution ---")
        print(self.df['claim_type'].value_counts())
        
        # DRG codes
        print("\n--- Top 10 DRG Codes ---")
        print(self.df['drg_code'].value_counts().head(10))
        
        # Revenue codes
        print("\n--- Top 10 Revenue Codes ---")
        print(self.df['revenue_code'].value_counts().head(10))
        
        # HCPCS codes
        print("\n--- Top 10 HCPCS Codes ---")
        print(self.df['hcpcs_code'].value_counts().head(10))
        
        # Length of stay analysis
        if 'length_of_stay' in self.df.columns:
            print("\n--- Length of Stay Statistics ---")
            print(self.df['length_of_stay'].describe())
            
            # LOS by upcoding status
            los_upcode = self.df.groupby('is_upcoded_header')['length_of_stay'].mean()
            print(f"\nAvg LOS (Upcoded): {los_upcode.get(True, 0):.2f}")
            print(f"Avg LOS (Clean): {los_upcode.get(False, 0):.2f}")
        
        return self.df['place_of_service'].value_counts()
    
    def diagnosis_analysis(self):
        """Analyze diagnosis patterns"""
        print("\n" + "="*80)
        print("7. DIAGNOSIS ANALYSIS")
        print("="*80)
        
        # Primary diagnosis
        print("\n--- Top 15 Primary Diagnoses ---")
        print(self.df['primary_diagnosis'].value_counts().head(15))
        
        # Secondary diagnosis
        print("\n--- Top 15 Secondary Diagnoses ---")
        print(self.df['secondary_diagnosis'].value_counts().head(15))
        
        # Diagnosis complexity (presence of secondary diagnosis)
        has_secondary = self.df['secondary_diagnosis'].notna().sum()
        print(f"\n--- Diagnosis Complexity ---")
        print(f"Claims with Secondary Diagnosis: {has_secondary:,} ({has_secondary/len(self.df)*100:.2f}%)")
        
        # Upcoding by diagnosis presence
        secondary_mask = self.df['secondary_diagnosis'].notna()
        no_secondary_mask = self.df['secondary_diagnosis'].isna()
        
        upcode_with_secondary = self.df.loc[secondary_mask, 'is_upcoded_header'].mean() * 100
        upcode_without_secondary = self.df.loc[no_secondary_mask, 'is_upcoded_header'].mean() * 100
        
        print(f"\nUpcode Rate (with secondary dx): {upcode_with_secondary:.2f}%")
        print(f"Upcode Rate (without secondary dx): {upcode_without_secondary:.2f}%")
        
        return self.df['primary_diagnosis'].value_counts()
    
    def temporal_analysis(self):
        """Analyze temporal patterns"""
        print("\n" + "="*80)
        print("8. TEMPORAL ANALYSIS")
        print("="*80)
        
        # Service date range
        if 'service_date_header' in self.df.columns:
            print("\n--- Service Date Range ---")
            print(f"Earliest: {self.df['service_date_header'].min()}")
            print(f"Latest: {self.df['service_date_header'].max()}")
            
            # Monthly trends
            self.df['year_month'] = self.df['service_date_header'].dt.to_period('M')
            monthly_claims = self.df.groupby('year_month').size()
            print(f"\n--- Monthly Claim Volume ---")
            print(monthly_claims)
            
            # Monthly upcoding rate
            monthly_upcode = self.df.groupby('year_month')['is_upcoded_header'].mean() * 100
            print(f"\n--- Monthly Upcoding Rate ---")
            print(monthly_upcode)
        
        return monthly_claims if 'service_date_header' in self.df.columns else None
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        print("\n" + "="*80)
        print("9. CORRELATION ANALYSIS")
        print("="*80)
        
        # Select numerical columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Correlation with upcoding
        if 'is_upcoded_header' in self.df.columns:
            upcode_corr = self.df[num_cols].corrwith(self.df['is_upcoded_header'].astype(int))
            upcode_corr = upcode_corr.sort_values(ascending=False)
            
            print("\n--- Correlation with Header-Level Upcoding ---")
            print(upcode_corr)
        
        if 'is_upcoded_line' in self.df.columns:
            upcode_line_corr = self.df[num_cols].corrwith(self.df['is_upcoded_line'].astype(int))
            upcode_line_corr = upcode_line_corr.sort_values(ascending=False)
            
            print("\n--- Correlation with Line-Level Upcoding ---")
            print(upcode_line_corr)
        
        return upcode_corr if 'is_upcoded_header' in self.df.columns else None
    
    def outlier_detection(self):
        """Identify outliers in key metrics"""
        print("\n" + "="*80)
        print("10. OUTLIER DETECTION")
        print("="*80)
        
        # Charge outliers
        q1 = self.df['total_charges_header'].quantile(0.25)
        q3 = self.df['total_charges_header'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (self.df['total_charges_header'] < lower_bound) | (self.df['total_charges_header'] > upper_bound)
        charge_outliers = self.df[outlier_mask]
        non_outliers = self.df[~outlier_mask]
        
        print(f"\n--- Charge Outliers ---")
        print(f"Number of outliers: {len(charge_outliers):,}")
        print(f"Percentage: {len(charge_outliers)/len(self.df)*100:.2f}%")
        print(f"Outlier upcoding rate: {charge_outliers['is_upcoded_header'].mean()*100:.2f}%")
        print(f"Non-outlier upcoding rate: {non_outliers['is_upcoded_header'].mean()*100:.2f}%")
        
        # LOS outliers
        if 'length_of_stay' in self.df.columns:
            los_threshold = self.df['length_of_stay'].quantile(0.95)
            los_outlier_mask = self.df['length_of_stay'] > los_threshold
            los_outliers = self.df[los_outlier_mask]
            
            print(f"\n--- Length of Stay Outliers (>95th percentile) ---")
            print(f"Number: {len(los_outliers):,}")
            if len(los_outliers) > 0:
                print(f"Avg LOS: {los_outliers['length_of_stay'].mean():.2f} days")
                print(f"Upcode rate: {los_outliers['is_upcoded_header'].mean()*100:.2f}%")
        
        return charge_outliers
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        
        total_claims = self.df['claim_id'].nunique()
        total_lines = len(self.df)
        upcoded_claims = self.df.groupby('claim_id')['is_upcoded_header'].first().sum()
        upcode_rate = (upcoded_claims / total_claims * 100)
        
        total_charges = self.df['total_charges_header'].sum()
        upcoded_charges = self.df.loc[self.df['is_upcoded_header'] == True, 'total_charges_header'].sum()
        
        print(f"""
Dataset Overview:
- Total Claims: {total_claims:,}
- Total Line Items: {total_lines:,}
- Date Range: {self.df['service_date_header'].min()} to {self.df['service_date_header'].max()}

Upcoding Metrics:
- Upcoded Claims: {upcoded_claims:,} ({upcode_rate:.2f}%)
- Financial Impact: ${upcoded_charges:,.2f} ({upcoded_charges/total_charges*100:.2f}% of total charges)

Key Findings:
- Unique Patients: {self.df['patient_id'].nunique():,}
- Unique Providers: {self.df['provider_id'].nunique():,}
- Average Claim Value: ${self.df['total_charges_header'].mean():,.2f}
- Most Common Specialty: {self.df['provider_specialty'].mode()[0] if len(self.df['provider_specialty'].mode()) > 0 else 'N/A'}
        """)
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        viz_dir = f"{self.output_dir}/visualizations"
        
        # 1. Upcoding Overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Upcoding Analysis Overview', fontsize=16, fontweight='bold')
        
        # Header vs Line Upcoding
        upcode_counts = pd.DataFrame({
            'Level': ['Header', 'Line'],
            'Upcoded': [self.df['is_upcoded_header'].sum(), self.df['is_upcoded_line'].sum()],
            'Clean': [(~self.df['is_upcoded_header']).sum(), (~self.df['is_upcoded_line']).sum()]
        })
        upcode_counts.set_index('Level')[['Upcoded', 'Clean']].plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#2ecc71'])
        axes[0, 0].set_title('Upcoded vs Clean Claims')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend(loc='best')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # Upcoding Types Distribution
        upcode_types = self.df['upcoding_type_header'].value_counts().head(10)
        upcode_types.plot(kind='barh', ax=axes[0, 1], color='#3498db')
        axes[0, 1].set_title('Top 10 Upcoding Types')
        axes[0, 1].set_xlabel('Count')
        
        # Financial Impact
        financial_data = pd.DataFrame({
            'Category': ['Upcoded Claims', 'Clean Claims'],
            'Total Charges': [
                self.df.loc[self.df['is_upcoded_header'] == True, 'total_charges_header'].sum(),
                self.df.loc[self.df['is_upcoded_header'] == False, 'total_charges_header'].sum()
            ]
        })
        axes[1, 0].pie(financial_data['Total Charges'], labels=financial_data['Category'], 
                       autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], startangle=90)
        axes[1, 0].set_title('Financial Impact Distribution')
        
        # Upcoding Rate by Month
        if 'year_month' in self.df.columns:
            monthly_upcode = self.df.groupby('year_month')['is_upcoded_header'].mean() * 100
            monthly_upcode.plot(kind='line', ax=axes[1, 1], marker='o', color='#e74c3c', linewidth=2)
            axes[1, 1].set_title('Monthly Upcoding Rate Trend')
            axes[1, 1].set_ylabel('Upcoding Rate (%)')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Temporal data not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/01_upcoding_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 01_upcoding_overview.png")
        
        # 2. Provider Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Provider Analysis', fontsize=16, fontweight='bold')
        
        # Top Specialties by Volume
        top_specialties = self.df['provider_specialty'].value_counts().head(15)
        top_specialties.plot(kind='barh', ax=axes[0, 0], color='#9b59b6')
        axes[0, 0].set_title('Top 15 Provider Specialties by Volume')
        axes[0, 0].set_xlabel('Number of Claims')
        
        # Upcoding Rate by Specialty
        specialty_upcode = self.df.groupby('provider_specialty').agg({
            'is_upcoded_header': ['sum', 'count', 'mean']
        })
        specialty_upcode.columns = ['Upcoded', 'Total', 'Rate']
        specialty_upcode['Rate'] = specialty_upcode['Rate'] * 100
        specialty_upcode = specialty_upcode[specialty_upcode['Total'] >= 100].nlargest(15, 'Rate')
        
        specialty_upcode['Rate'].plot(kind='barh', ax=axes[0, 1], color='#e67e22')
        axes[0, 1].set_title('Top 15 Specialties by Upcoding Rate (min 100 claims)')
        axes[0, 1].set_xlabel('Upcoding Rate (%)')
        
        # Provider Volume Distribution
        provider_volume = self.df.groupby('provider_id').size()
        axes[1, 0].hist(provider_volume, bins=50, color='#1abc9c', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Provider Claim Volume Distribution')
        axes[1, 0].set_xlabel('Number of Claims per Provider')
        axes[1, 0].set_ylabel('Number of Providers')
        axes[1, 0].set_yscale('log')
        
        # Place of Service Distribution
        pos_dist = self.df['place_of_service'].value_counts().head(10)
        pos_dist.plot(kind='bar', ax=axes[1, 1], color='#34495e')
        axes[1, 1].set_title('Top 10 Places of Service')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/02_provider_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 02_provider_analysis.png")
        
        # 3. Financial Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Financial Analysis', fontsize=16, fontweight='bold')
        
        # Charge Distribution
        charges = self.df['total_charges_header'].dropna()
        axes[0, 0].hist(charges[charges < charges.quantile(0.95)], bins=50, 
                        color='#16a085', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Claim Charges Distribution (up to 95th percentile)')
        axes[0, 0].set_xlabel('Total Charges ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot: Charges by Upcoding Status
        upcode_charges_data = [
            self.df.loc[self.df['is_upcoded_header'] == False, 'total_charges_header'].dropna(),
            self.df.loc[self.df['is_upcoded_header'] == True, 'total_charges_header'].dropna()
        ]
        bp = axes[0, 1].boxplot(upcode_charges_data, labels=['Clean', 'Upcoded'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
        axes[0, 1].set_title('Charges Distribution by Upcoding Status')
        axes[0, 1].set_ylabel('Total Charges ($)')
        axes[0, 1].set_yscale('log')
        
        # Payment vs Charges
        sample_df = self.df.sample(min(10000, len(self.df)))
        axes[1, 0].scatter(sample_df['total_charges_header'], sample_df['expected_payment'], 
                          alpha=0.3, s=10, color='#8e44ad')
        axes[1, 0].set_title('Expected Payment vs Total Charges (sample)')
        axes[1, 0].set_xlabel('Total Charges ($)')
        axes[1, 0].set_ylabel('Expected Payment ($)')
        axes[1, 0].plot([0, sample_df['total_charges_header'].max()], 
                        [0, sample_df['total_charges_header'].max()], 
                        'r--', alpha=0.5, label='1:1 line')
        axes[1, 0].legend()
        
        # Charge-to-Payment Ratio
        ratio = self.df['charge_to_payment_ratio'].dropna()
        ratio_filtered = ratio[(ratio > 0) & (ratio < ratio.quantile(0.95))]
        axes[1, 1].hist(ratio_filtered, bins=50, color='#c0392b', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Charge-to-Payment Ratio Distribution')
        axes[1, 1].set_xlabel('Ratio')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/03_financial_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 03_financial_analysis.png")
        
        # 4. Clinical Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clinical & Service Analysis', fontsize=16, fontweight='bold')
        
        # Top Primary Diagnoses
        top_dx = self.df['primary_diagnosis'].value_counts().head(15)
        top_dx.plot(kind='barh', ax=axes[0, 0], color='#27ae60')
        axes[0, 0].set_title('Top 15 Primary Diagnoses')
        axes[0, 0].set_xlabel('Count')
        
        # DRG Code Distribution
        top_drg = self.df['drg_code'].value_counts().head(15)
        top_drg.plot(kind='barh', ax=axes[0, 1], color='#2980b9')
        axes[0, 1].set_title('Top 15 DRG Codes')
        axes[0, 1].set_xlabel('Count')
        
        # Length of Stay Distribution
        if 'length_of_stay' in self.df.columns:
            los = self.df['length_of_stay'].dropna()
            los_filtered = los[los < los.quantile(0.95)]
            axes[1, 0].hist(los_filtered, bins=30, color='#f39c12', edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('Length of Stay Distribution (up to 95th percentile)')
            axes[1, 0].set_xlabel('Days')
            axes[1, 0].set_ylabel('Frequency')
        else:
            axes[1, 0].text(0.5, 0.5, 'Length of Stay data not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
        
        # Claim Type Distribution
        claim_types = self.df['claim_type'].value_counts()
        axes[1, 1].pie(claim_types.values, labels=claim_types.index, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Claim Type Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/04_clinical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 04_clinical_analysis.png")
        
        # 5. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Select numerical columns for correlation
        corr_cols = ['total_charges_header', 'expected_payment', 'allowed_amount', 
                     'total_charges_line', 'units', 'unit_cost', 'length_of_stay']
        corr_cols = [col for col in corr_cols if col in self.df.columns]
        
        if len(corr_cols) > 1:
            corr_matrix = self.df[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/05_correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: 05_correlation_heatmap.png")
        
        print("\n✓ All visualizations generated successfully!")
    
    def export_summary_data(self):
        """Export analysis results to CSV files"""
        print("\n" + "="*80)
        print("EXPORTING SUMMARY DATA TO CSV")
        print("="*80)
        
        csv_dir = f"{self.output_dir}/csv_outputs"
        
        # 1. Overall Summary Statistics
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Claims',
                'Total Line Items',
                'Unique Patients',
                'Unique Providers',
                'Upcoded Claims (Header)',
                'Upcoding Rate (Header) %',
                'Upcoded Lines',
                'Upcoding Rate (Line) %',
                'Total Charges',
                'Upcoded Charges',
                'Average Claim Charge',
                'Median Claim Charge',
                'Average Expected Payment'
            ],
            'Value': [
                self.df['claim_id'].nunique(),
                len(self.df),
                self.df['patient_id'].nunique(),
                self.df['provider_id'].nunique(),
                self.df.groupby('claim_id')['is_upcoded_header'].first().sum(),
                (self.df.groupby('claim_id')['is_upcoded_header'].first().mean() * 100),
                self.df['is_upcoded_line'].sum(),
                (self.df['is_upcoded_line'].mean() * 100),
                self.df['total_charges_header'].sum(),
                self.df.loc[self.df['is_upcoded_header'] == True, 'total_charges_header'].sum(),
                self.df['total_charges_header'].mean(),
                self.df['total_charges_header'].median(),
                self.df['expected_payment'].mean()
            ]
        })
        summary_stats.to_csv(f"{csv_dir}/01_overall_summary.csv", index=False)
        print(f"✓ Saved: 01_overall_summary.csv")
        
        # 2. Upcoding Type Analysis
        upcode_type_summary = pd.DataFrame({
            'Header_Types': self.df['upcoding_type_header'].value_counts(),
            'Line_Types': self.df['upcoding_type_line'].value_counts()
        }).fillna(0)
        upcode_type_summary.to_csv(f"{csv_dir}/02_upcoding_types.csv")
        print(f"✓ Saved: 02_upcoding_types.csv")
        
        # 3. Top Upcoding Reasons
        upcode_reasons = self.df['upcoding_reason'].value_counts().head(50)
        upcode_reasons.to_csv(f"{csv_dir}/03_upcoding_reasons.csv", header=['Count'])
        print(f"✓ Saved: 03_upcoding_reasons.csv")
        
        # 4. Provider Specialty Analysis
        specialty_analysis = self.df.groupby('provider_specialty').agg({
            'claim_id': 'count',
            'is_upcoded_header': ['sum', 'mean'],
            'total_charges_header': ['sum', 'mean'],
            'expected_payment': 'mean'
        }).round(2)
        specialty_analysis.columns = ['Total_Claims', 'Upcoded_Count', 'Upcode_Rate', 
                                      'Total_Charges', 'Avg_Charge', 'Avg_Payment']
        specialty_analysis['Upcode_Rate'] = (specialty_analysis['Upcode_Rate'] * 100).round(2)
        specialty_analysis = specialty_analysis.sort_values('Total_Claims', ascending=False)
        specialty_analysis.to_csv(f"{csv_dir}/04_specialty_analysis.csv")
        print(f"✓ Saved: 04_specialty_analysis.csv")
        
        # 5. Provider-Level Analysis (Top 1000)
        provider_analysis = self.df.groupby('provider_id').agg({
            'claim_id': 'count',
            'is_upcoded_header': ['sum', 'mean'],
            'total_charges_header': ['sum', 'mean'],
            'provider_specialty': 'first',
            'provider_npi': 'first'
        }).round(2)
        provider_analysis.columns = ['Total_Claims', 'Upcoded_Count', 'Upcode_Rate', 
                                     'Total_Charges', 'Avg_Charge', 'Specialty', 'NPI']
        provider_analysis['Upcode_Rate'] = (provider_analysis['Upcode_Rate'] * 100).round(2)
        provider_analysis = provider_analysis.sort_values('Total_Claims', ascending=False).head(1000)
        provider_analysis.to_csv(f"{csv_dir}/05_top_providers.csv")
        print(f"✓ Saved: 05_top_providers.csv")
        
        # 6. Diagnosis Code Analysis
        dx_analysis = pd.DataFrame({
            'Primary_Diagnosis': self.df['primary_diagnosis'].value_counts().head(100),
            'Secondary_Diagnosis': self.df['secondary_diagnosis'].value_counts().head(100)
        }).fillna(0)
        dx_analysis.to_csv(f"{csv_dir}/06_diagnosis_codes.csv")
        print(f"✓ Saved: 06_diagnosis_codes.csv")
        
        # 7. Procedure Codes (DRG, HCPCS, Revenue)
        procedure_summary = pd.DataFrame({
            'DRG_Code': self.df['drg_code'].value_counts().head(50),
            'Revenue_Code': self.df['revenue_code'].value_counts().head(50),
            'HCPCS_Code': self.df['hcpcs_code'].value_counts().head(50)
        }).fillna(0)
        procedure_summary.to_csv(f"{csv_dir}/07_procedure_codes.csv")
        print(f"✓ Saved: 07_procedure_codes.csv")
        
        # 8. Place of Service Analysis
        pos_analysis = self.df.groupby('place_of_service').agg({
            'claim_id': 'count',
            'is_upcoded_header': ['sum', 'mean'],
            'total_charges_header': 'mean'
        }).round(2)
        pos_analysis.columns = ['Total_Claims', 'Upcoded_Count', 'Upcode_Rate', 'Avg_Charge']
        pos_analysis['Upcode_Rate'] = (pos_analysis['Upcode_Rate'] * 100).round(2)
        pos_analysis = pos_analysis.sort_values('Total_Claims', ascending=False)
        pos_analysis.to_csv(f"{csv_dir}/08_place_of_service.csv")
        print(f"✓ Saved: 08_place_of_service.csv")
        
        # 9. Monthly Trends (if available)
        if 'year_month' in self.df.columns:
            monthly_trends = self.df.groupby('year_month').agg({
                'claim_id': 'count',
                'is_upcoded_header': ['sum', 'mean'],
                'total_charges_header': ['sum', 'mean']
            }).round(2)
            monthly_trends.columns = ['Total_Claims', 'Upcoded_Count', 'Upcode_Rate', 
                                     'Total_Charges', 'Avg_Charge']
            monthly_trends['Upcode_Rate'] = (monthly_trends['Upcode_Rate'] * 100).round(2)
            monthly_trends.to_csv(f"{csv_dir}/09_monthly_trends.csv")
            print(f"✓ Saved: 09_monthly_trends.csv")
        
        # 10. Financial Metrics Summary
        financial_summary = pd.DataFrame({
            'Metric': ['Total Charges', 'Mean Charges', 'Median Charges', 'Std Charges',
                      'Total Expected Payment', 'Mean Expected Payment',
                      'Total Allowed Amount', 'Mean Allowed Amount'],
            'Value': [
                self.df['total_charges_header'].sum(),
                self.df['total_charges_header'].mean(),
                self.df['total_charges_header'].median(),
                self.df['total_charges_header'].std(),
                self.df['expected_payment'].sum(),
                self.df['expected_payment'].mean(),
                self.df['allowed_amount'].sum(),
                self.df['allowed_amount'].mean()
            ]
        }).round(2)
        financial_summary.to_csv(f"{csv_dir}/10_financial_summary.csv", index=False)
        print(f"✓ Saved: 10_financial_summary.csv")
        
        # 11. High-Risk Claims (High charges + Upcoded)
        high_risk = self.df[
            (self.df['is_upcoded_header'] == True) & 
            (self.df['total_charges_header'] > self.df['total_charges_header'].quantile(0.90))
        ][['claim_id', 'patient_id', 'provider_id', 'provider_specialty', 
           'total_charges_header', 'expected_payment', 'upcoding_type_header', 
           'upcoding_reason', 'primary_diagnosis']].drop_duplicates('claim_id').head(1000)
        high_risk.to_csv(f"{csv_dir}/11_high_risk_claims.csv", index=False)
        print(f"✓ Saved: 11_high_risk_claims.csv")
        
        # 12. Correlation Matrix
        corr_cols = ['total_charges_header', 'expected_payment', 'allowed_amount', 
                     'total_charges_line', 'units', 'length_of_stay']
        corr_cols = [col for col in corr_cols if col in self.df.columns]
        if len(corr_cols) > 1:
            corr_matrix = self.df[corr_cols].corr().round(3)
            corr_matrix.to_csv(f"{csv_dir}/12_correlation_matrix.csv")
            print(f"✓ Saved: 12_correlation_matrix.csv")
        
        print("\n✓ All CSV outputs exported successfully!")
        print(f"\nOutput location: {os.path.abspath(self.output_dir)}")
    
    def run_full_analysis(self):
        """Execute complete analysis pipeline"""
        print("\n" + "="*80)
        print("HEALTHCARE CLAIMS DESCRIPTIVE ANALYSIS")
        print("Focus: Upcoding Detection")
        print("="*80)
        
        self.basic_statistics()
        self.claims_overview()
        self.financial_analysis()
        self.upcoding_analysis()
        self.provider_analysis()
        self.service_analysis()
        self.diagnosis_analysis()
        self.temporal_analysis()
        self.correlation_analysis()
        self.outlier_detection()
        self.generate_summary_report()
        
        # Generate visualizations and exports
        self.create_visualizations()
        self.export_summary_data()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {os.path.abspath(self.output_dir)}")
        print(f"  - Visualizations: {os.path.abspath(self.output_dir)}/visualizations/")
        print(f"  - CSV Outputs: {os.path.abspath(self.output_dir)}/csv_outputs/")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Usage Example
if __name__ == "__main__":
    # Initialize analyzer with output directory
    analyzer = HealthcareClaimsAnalyzer(
        db_path='Data/claims_database.db',
        output_dir='claims_analysis_results'
    )
    
    # Load data (replace 'claims' with your actual table name)
    analyzer.load_data(table_name='claims_data')
    
    # Run full analysis with visualizations and CSV exports
    analyzer.run_full_analysis()
    
    # Close connection
    analyzer.close()
    
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print("""
    To use this script:
    
    1. Replace 'path_to_your_database.db' with your SQLite database path
    2. Replace 'claims' with your actual table name if different
    3. Run: python healthcare_claims_analysis.py
    
    Outputs will be generated in 'claims_analysis_results/' directory:
    
    VISUALIZATIONS (PNG files):
    - 01_upcoding_overview.png - Upcoding trends and distributions
    - 02_provider_analysis.png - Provider patterns and specialties
    - 03_financial_analysis.png - Financial metrics and distributions
    - 04_clinical_analysis.png - Clinical codes and services
    - 05_correlation_heatmap.png - Feature correlations
    
    CSV OUTPUTS:
    - 01_overall_summary.csv - Key metrics summary
    - 02_upcoding_types.csv - Upcoding type breakdown
    - 03_upcoding_reasons.csv - Top upcoding reasons
    - 04_specialty_analysis.csv - Analysis by provider specialty
    - 05_top_providers.csv - Top 1000 providers with metrics
    - 06_diagnosis_codes.csv - Diagnosis code frequencies
    - 07_procedure_codes.csv - DRG, HCPCS, Revenue codes
    - 08_place_of_service.csv - Service location analysis
    - 09_monthly_trends.csv - Temporal trends
    - 10_financial_summary.csv - Financial statistics
    - 11_high_risk_claims.csv - High-risk upcoded claims
    - 12_correlation_matrix.csv - Feature correlation matrix
    """)