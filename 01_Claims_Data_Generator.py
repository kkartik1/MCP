import random
import pandas as pd
from datetime import datetime, timedelta
import uuid
import numpy as np
import os

class HealthcareClaimsGenerator:
    def __init__(self):
        # Common diagnosis codes
        self.diagnosis_codes = {
            'mild': ['Z00.00', 'K59.00', 'M79.3', 'H52.4', 'J06.9'],
            'moderate': ['E11.9', 'I10', 'M17.9', 'J44.1', 'N39.0'],
            'severe': ['I21.9', 'J44.0', 'N18.6', 'C78.00', 'G93.1']
        }
        
        # Procedure codes with base costs
        self.procedure_codes = {
            'routine': [('99213', 180), ('99214', 280), ('36415', 25), ('80053', 45)],
            'complex': [('99223', 450), ('99233', 380), ('43235', 1200), ('47562', 8500)],
            'surgical': [('29881', 3500), ('64721', 2800), ('66984', 2200), ('27447', 15000)]
        }
        
        # DRG codes with expected LOS and costs
        self.drg_codes = {
            'medical': [
                (194, 'Simple Pneumonia', 3, 8500),
                (292, 'Heart Failure', 4, 12000),
                (683, 'Renal Failure', 5, 15000)
            ],
            'surgical': [
                (469, 'Major Joint Replacement', 2, 25000),
                (247, 'Percutaneous Cardiovascular Proc', 1, 18000),
                (329, 'Major Small & Large Bowel Proc', 6, 35000)
            ]
        }
        
        # Provider specialties
        self.specialties = [
            'Internal Medicine', 'Family Practice', 'Cardiology', 
            'Orthopedic Surgery', 'General Surgery', 'Emergency Medicine',
            'Radiology', 'Anesthesiology', 'Pathology'
        ]
        
        # Place of service codes
        self.place_of_service = {
            'office': 11,
            'inpatient': 21,
            'outpatient': 22,
            'emergency': 23,
            'ambulatory_surgical': 24
        }

    def generate_patient_data(self):
        """Generate synthetic patient data"""
        return {
            'patient_id': str(uuid.uuid4())[:8],
            'member_id': f"MBR{random.randint(100000, 999999)}",
            'date_of_birth': datetime(1950, 1, 1) + timedelta(days=random.randint(0, 25550)),
            'gender': random.choice(['M', 'F']),
            'zip_code': f"{random.randint(10000, 99999)}"
        }

    def generate_provider_data(self):
        """Generate synthetic provider data"""
        return {
            'provider_id': f"PRV{random.randint(100000, 999999)}",
            'npi': f"{random.randint(1000000000, 9999999999)}",
            'specialty': random.choice(self.specialties),
            'provider_type': random.choice(['Individual', 'Facility']),
            'tax_id': f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}"
        }

    def introduce_drg_upcoding(self, base_drg, base_cost, base_los):
        """Introduce DRG upcoding scenarios"""
        upcoding_scenarios = [
            # Severity level manipulation
            {'factor': 1.3, 'reason': 'Severity level inflated'},
            {'factor': 1.5, 'reason': 'Complication added inappropriately'},
            {'factor': 1.2, 'reason': 'Principal diagnosis manipulation'}
        ]
        
        if random.random() < 0.15:  # 15% chance of DRG upcoding
            scenario = random.choice(upcoding_scenarios)
            return {
                'drg_code': base_drg + random.randint(1, 3),  # Higher DRG typically means higher severity
                'upcoded_cost': int(base_cost * scenario['factor']),
                'upcoded_los': base_los + random.randint(1, 2),
                'is_upcoded': True,
                'upcoding_type': 'DRG',
                'upcoding_reason': scenario['reason']
            }
        return {
            'drg_code': base_drg,
            'upcoded_cost': base_cost,
            'upcoded_los': base_los,
            'is_upcoded': False,
            'upcoding_type': None,
            'upcoding_reason': None
        }

    def introduce_em_upcoding(self, base_code, base_cost):
        """Introduce E&M upcoding scenarios"""
        em_hierarchy = {
            '99211': (75, '99212'),
            '99212': (120, '99213'),
            '99213': (180, '99214'),
            '99214': (280, '99215'),
            '99215': (380, None)
        }
        
        if base_code in em_hierarchy and random.random() < 0.20:  # 20% chance of E&M upcoding
            higher_code = em_hierarchy[base_code][1]
            if higher_code:
                return {
                    'procedure_code': higher_code,
                    'upcoded_cost': em_hierarchy[higher_code][0],
                    'is_upcoded': True,
                    'upcoding_type': 'E&M',
                    'upcoding_reason': 'E&M level inappropriately increased'
                }
        
        return {
            'procedure_code': base_code,
            'upcoded_cost': base_cost,
            'is_upcoded': False,
            'upcoding_type': None,
            'upcoding_reason': None
        }

    def introduce_surgical_upcoding(self, base_code, base_cost):
        """Introduce surgical upcoding scenarios"""
        surgical_upcoding_scenarios = [
            {'factor': 1.4, 'reason': 'Bilateral procedure coded when unilateral performed'},
            {'factor': 1.6, 'reason': 'Complex procedure coded instead of simple'},
            {'factor': 1.3, 'reason': 'Additional modifier inappropriately applied'}
        ]
        
        if random.random() < 0.12:  # 12% chance of surgical upcoding
            scenario = random.choice(surgical_upcoding_scenarios)
            return {
                'procedure_code': base_code,
                'upcoded_cost': int(base_cost * scenario['factor']),
                'is_upcoded': True,
                'upcoding_type': 'Surgical',
                'upcoding_reason': scenario['reason']
            }
        
        return {
            'procedure_code': base_code,
            'upcoded_cost': base_cost,
            'is_upcoded': False,
            'upcoding_type': None,
            'upcoding_reason': None
        }

    def generate_inpatient_claim(self):
        """Generate inpatient facility claim"""
        patient = self.generate_patient_data()
        provider = self.generate_provider_data()
        
        # Select DRG
        drg_category = random.choice(['medical', 'surgical'])
        base_drg_code, drg_description, base_los, base_cost = random.choice(self.drg_codes[drg_category])
        
        # Apply DRG upcoding
        drg_info = self.introduce_drg_upcoding(base_drg_code, base_cost, base_los)
        
        admission_date = datetime.now() - timedelta(days=random.randint(1, 365))
        discharge_date = admission_date + timedelta(days=drg_info['upcoded_los'])
        
        # Generate line items
        line_items = []
        
        # Room and board
        daily_rate = random.randint(1200, 2500)
        line_items.append({
            'line_number': 1,
            'revenue_code': '0101',
            'hcpcs_code': None,
            'procedure_description': 'Room and Board - Medical/Surgical',
            'service_date': admission_date.date(),
            'units': drg_info['upcoded_los'],
            'unit_cost': daily_rate,
            'total_charges': daily_rate * drg_info['upcoded_los'],
            'allowed_amount': daily_rate * drg_info['upcoded_los'] * 0.85,
            'is_upcoded': drg_info['is_upcoded'],
            'upcoding_type': drg_info['upcoding_type']
        })
        
        # Pharmacy
        line_items.append({
            'line_number': 2,
            'revenue_code': '0250',
            'hcpcs_code': None,
            'procedure_description': 'Pharmacy',
            'service_date': admission_date.date(),
            'units': 1,
            'unit_cost': random.randint(500, 3000),
            'total_charges': random.randint(500, 3000),
            'allowed_amount': random.randint(425, 2550),
            'is_upcoded': False,
            'upcoding_type': None
        })
        
        # Laboratory
        lab_cost = random.randint(200, 1500)
        line_items.append({
            'line_number': 3,
            'revenue_code': '0300',
            'hcpcs_code': '80053',
            'procedure_description': 'Comprehensive Metabolic Panel',
            'service_date': admission_date.date(),
            'units': 1,
            'unit_cost': lab_cost,
            'total_charges': lab_cost,
            'allowed_amount': lab_cost * 0.80,
            'is_upcoded': False,
            'upcoding_type': None
        })
        
        total_charges = sum([item['total_charges'] for item in line_items])
        
        return {
            'claim_id': str(uuid.uuid4()),
            'claim_type': 'Inpatient Facility',
            'patient_id': patient['patient_id'],
            'member_id': patient['member_id'],
            'provider_id': provider['provider_id'],
            'provider_npi': provider['npi'],
            'provider_specialty': provider['specialty'],
            'place_of_service': self.place_of_service['inpatient'],
            'admission_date': admission_date.date(),
            'discharge_date': discharge_date.date(),
            'length_of_stay': drg_info['upcoded_los'],
            'drg_code': drg_info['drg_code'],
            'primary_diagnosis': random.choice(self.diagnosis_codes['moderate']),
            'secondary_diagnosis': random.choice(self.diagnosis_codes['mild']),
            'total_charges': total_charges,
            'expected_payment': total_charges * 0.85,
            'is_upcoded': drg_info['is_upcoded'],
            'upcoding_type': drg_info['upcoding_type'],
            'upcoding_reason': drg_info['upcoding_reason'],
            'line_items': line_items
        }

    def generate_outpatient_professional_claim(self):
        """Generate outpatient professional claim"""
        patient = self.generate_patient_data()
        provider = self.generate_provider_data()
        
        service_date = datetime.now() - timedelta(days=random.randint(1, 90))
        
        # Select base procedure
        procedure_type = random.choice(['routine', 'complex'])
        base_code, base_cost = random.choice(self.procedure_codes[procedure_type])
        
        # Apply E&M upcoding
        proc_info = self.introduce_em_upcoding(base_code, base_cost)
        
        line_items = []
        
        # Main procedure
        line_items.append({
            'line_number': 1,
            'revenue_code': None,
            'hcpcs_code': proc_info['procedure_code'],
            'procedure_description': f'E&M Service Level {proc_info["procedure_code"][-1]}',
            'service_date': service_date.date(),
            'units': 1,
            'unit_cost': proc_info['upcoded_cost'],
            'total_charges': proc_info['upcoded_cost'],
            'allowed_amount': proc_info['upcoded_cost'] * 0.90,
            'is_upcoded': proc_info['is_upcoded'],
            'upcoding_type': proc_info['upcoding_type']
        })
        
        # Additional services (sometimes)
        if random.random() < 0.3:
            additional_cost = random.randint(50, 200)
            line_items.append({
                'line_number': 2,
                'revenue_code': None,
                'hcpcs_code': '36415',
                'procedure_description': 'Venipuncture',
                'service_date': service_date.date(),
                'units': 1,
                'unit_cost': additional_cost,
                'total_charges': additional_cost,
                'allowed_amount': additional_cost * 0.85,
                'is_upcoded': False,
                'upcoding_type': None
            })
        
        total_charges = sum([item['total_charges'] for item in line_items])
        
        return {
            'claim_id': str(uuid.uuid4()),
            'claim_type': 'Outpatient Professional',
            'patient_id': patient['patient_id'],
            'member_id': patient['member_id'],
            'provider_id': provider['provider_id'],
            'provider_npi': provider['npi'],
            'provider_specialty': provider['specialty'],
            'place_of_service': self.place_of_service['office'],
            'service_date': service_date.date(),
            'primary_diagnosis': random.choice(self.diagnosis_codes['mild']),
            'secondary_diagnosis': None,
            'total_charges': total_charges,
            'expected_payment': total_charges * 0.90,
            'is_upcoded': proc_info['is_upcoded'],
            'upcoding_type': proc_info['upcoding_type'],
            'upcoding_reason': proc_info['upcoding_reason'],
            'line_items': line_items
        }

    def generate_outpatient_surgical_claim(self):
        """Generate outpatient surgical claim"""
        patient = self.generate_patient_data()
        provider = self.generate_provider_data()
        
        service_date = datetime.now() - timedelta(days=random.randint(1, 180))
        
        # Select surgical procedure
        base_code, base_cost = random.choice(self.procedure_codes['surgical'])
        
        # Apply surgical upcoding
        proc_info = self.introduce_surgical_upcoding(base_code, base_cost)
        
        line_items = []
        
        # Main surgical procedure
        line_items.append({
            'line_number': 1,
            'revenue_code': '0360',
            'hcpcs_code': proc_info['procedure_code'],
            'procedure_description': 'Surgical Procedure',
            'service_date': service_date.date(),
            'units': 1,
            'unit_cost': proc_info['upcoded_cost'],
            'total_charges': proc_info['upcoded_cost'],
            'allowed_amount': proc_info['upcoded_cost'] * 0.80,
            'is_upcoded': proc_info['is_upcoded'],
            'upcoding_type': proc_info['upcoding_type']
        })
        
        # Anesthesia
        anesthesia_cost = random.randint(800, 2000)
        line_items.append({
            'line_number': 2,
            'revenue_code': '0370',
            'hcpcs_code': '00142',
            'procedure_description': 'Anesthesia Service',
            'service_date': service_date.date(),
            'units': random.randint(3, 8),
            'unit_cost': anesthesia_cost // random.randint(3, 8),
            'total_charges': anesthesia_cost,
            'allowed_amount': anesthesia_cost * 0.85,
            'is_upcoded': False,
            'upcoding_type': None
        })
        
        # Recovery room
        recovery_cost = random.randint(300, 800)
        line_items.append({
            'line_number': 3,
            'revenue_code': '0710',
            'hcpcs_code': None,
            'procedure_description': 'Recovery Room',
            'service_date': service_date.date(),
            'units': 1,
            'unit_cost': recovery_cost,
            'total_charges': recovery_cost,
            'allowed_amount': recovery_cost * 0.90,
            'is_upcoded': False,
            'upcoding_type': None
        })
        
        total_charges = sum([item['total_charges'] for item in line_items])
        
        return {
            'claim_id': str(uuid.uuid4()),
            'claim_type': 'Outpatient Surgical',
            'patient_id': patient['patient_id'],
            'member_id': patient['member_id'],
            'provider_id': provider['provider_id'],
            'provider_npi': provider['npi'],
            'provider_specialty': provider['specialty'],
            'place_of_service': self.place_of_service['ambulatory_surgical'],
            'service_date': service_date.date(),
            'primary_diagnosis': random.choice(self.diagnosis_codes['moderate']),
            'secondary_diagnosis': random.choice(self.diagnosis_codes['mild']),
            'total_charges': total_charges,
            'expected_payment': total_charges * 0.82,
            'is_upcoded': proc_info['is_upcoded'],
            'upcoding_type': proc_info['upcoding_type'],
            'upcoding_reason': proc_info['upcoding_reason'],
            'line_items': line_items
        }

    def generate_claims_dataset(self, num_claims=1000):
        """Generate a dataset of healthcare claims"""
        claims = []
        line_items_all = []
        
        claim_types = [
            ('inpatient', 0.2),
            ('outpatient_professional', 0.5),
            ('outpatient_surgical', 0.3)
        ]
        
        for i in range(num_claims):
            # Select claim type based on distribution
            rand = random.random()
            cumulative = 0
            for claim_type, prob in claim_types:
                cumulative += prob
                if rand <= cumulative:
                    if claim_type == 'inpatient':
                        claim = self.generate_inpatient_claim()
                    elif claim_type == 'outpatient_professional':
                        claim = self.generate_outpatient_professional_claim()
                    else:
                        claim = self.generate_outpatient_surgical_claim()
                    break
            
            # Extract line items for separate table
            line_items = claim.pop('line_items')
            for item in line_items:
                item['claim_id'] = claim['claim_id']
                line_items_all.append(item)
            
            claims.append(claim)
        
        # Convert to DataFrames
        claims_df = pd.DataFrame(claims)
        line_items_df = pd.DataFrame(line_items_all)
        
        return claims_df, line_items_df

    def generate_summary_report(self, claims_df):
        """Generate summary report of the generated claims"""
        total_claims = len(claims_df)
        upcoded_claims = len(claims_df[claims_df['is_upcoded'] == True])
        
        print("=== Healthcare Claims Generation Summary ===")
        print(f"Total Claims Generated: {total_claims:,}")
        print(f"Upcoded Claims: {upcoded_claims:,} ({upcoded_claims/total_claims*100:.1f}%)")
        print()
        
        print("Claims by Type:")
        type_counts = claims_df['claim_type'].value_counts()
        for claim_type, count in type_counts.items():
            print(f"  {claim_type}: {count:,} ({count/total_claims*100:.1f}%)")
        print()
        
        print("Upcoding by Type:")
        upcoding_counts = claims_df[claims_df['is_upcoded'] == True]['upcoding_type'].value_counts()
        for upcode_type, count in upcoding_counts.items():
            print(f"  {upcode_type}: {count:,} ({count/upcoded_claims*100:.1f}% of upcoded claims)")
        print()
        
        print("Financial Impact:")
        total_charges = claims_df['total_charges'].sum()
        upcoded_charges = claims_df[claims_df['is_upcoded'] == True]['total_charges'].sum()
        print(f"  Total Charges: ${total_charges:,.2f}")
        print(f"  Upcoded Charges: ${upcoded_charges:,.2f}")
        print(f"  Potential Overpayment: ${upcoded_charges * 0.15:,.2f} (estimated)")


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = HealthcareClaimsGenerator()
    
    # Generate claims dataset
    print("Generating healthcare claims dataset...")
    claims_df, line_items_df = generator.generate_claims_dataset(num_claims=1000000)
    
    # Generate summary report
    generator.generate_summary_report(claims_df)
    
    # Save to CSV files
    file_path = os.path.join('Data', 'healthcare_claims.csv')
    claims_df.to_csv(file_path, index=False)
    file_path = os.path.join('Data', 'claim_line_items.csv')
    line_items_df.to_csv(file_path, index=False)
    
    print(f"\nDataset saved to:")
    print(f"  - healthcare_claims.csv ({len(claims_df)} claims)")
    print(f"  - claim_line_items.csv ({len(line_items_df)} line items)")
    
    # Display sample data
    print("\n=== Sample Claims Data ===")
    print(claims_df.head(3).to_string())
    
    print("\n=== Sample Line Items Data ===")
    print(line_items_df.head(5).to_string())
    
    # Show upcoding examples
    print("\n=== Sample Upcoded Claims ===")
    upcoded_claims = claims_df[claims_df['is_upcoded'] == True].head(3)
    for _, claim in upcoded_claims.iterrows():
        print(f"Claim ID: {claim['claim_id']}")
        print(f"Type: {claim['upcoding_type']} upcoding")
        print(f"Reason: {claim['upcoding_reason']}")
        print(f"Total Charges: ${claim['total_charges']:,.2f}")
        print("---")