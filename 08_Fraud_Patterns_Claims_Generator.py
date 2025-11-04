import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
from faker import Faker
import json

# Initialize Faker for generating realistic data
fake = Faker()

# Healthcare procedure codes and costs by specialty
PROCEDURE_CODES = {
    'cardiology': {
        'standard': [
            ('93306', 'Echocardiogram', 200),
            ('93000', 'Electrocardiogram', 150),
            ('93017', 'Cardiovascular stress test', 400),
            ('93453', 'Cardiac catheterization', 2500),
            ('33533', 'Coronary artery bypass', 15000)
        ],
        'high_cost': [
            ('93453', 'Cardiac catheterization', 2500),
            ('33533', 'Coronary artery bypass', 15000),
            ('33264', 'Pacemaker implantation', 12000),
            ('92928', 'Coronary angioplasty', 8000)
        ]
    },
    'dermatology': {
        'standard': [
            ('11100', 'Skin biopsy', 300),
            ('17000', 'Destruction of lesion', 200),
            ('11401', 'Excision of lesion', 400),
            ('96567', 'Photodynamic therapy', 800),
            ('15823', 'Blepharoplasty', 3000)
        ],
        'high_cost': [
            ('15823', 'Blepharoplasty', 3000),
            ('15876', 'Suction lipectomy', 4500),
            ('96567', 'Photodynamic therapy', 800),
            ('17311', 'Mohs surgery', 2200)
        ]
    },
    'orthopedics': {
        'standard': [
            ('27447', 'Knee arthroplasty', 8000),
            ('29881', 'Arthroscopy knee', 3500),
            ('73721', 'MRI lower extremity', 1200),
            ('20610', 'Joint injection', 300),
            ('97110', 'Physical therapy', 150)
        ],
        'high_cost': [
            ('27447', 'Knee arthroplasty', 8000),
            ('27130', 'Hip arthroplasty', 9500),
            ('22614', 'Spinal fusion', 12000),
            ('29881', 'Arthroscopy knee', 3500)
        ]
    },
    'mental_health': {
        'standard': [
            ('90834', 'Psychotherapy 45 min', 150),
            ('90837', 'Psychotherapy 60 min', 200),
            ('90901', 'Group therapy', 100),
            ('96116', 'Neurobehavioral assessment', 400),
            ('90791', 'Psychiatric evaluation', 350)
        ],
        'high_cost': [
            ('96116', 'Neurobehavioral assessment', 400),
            ('90901', 'Group therapy', 100),
            ('90834', 'Psychotherapy 45 min', 150),
            ('90837', 'Psychotherapy 60 min', 200)
        ]
    },
    'chiropractic': {
        'standard': [
            ('98940', 'Chiropractic manipulation', 100),
            ('97012', 'Mechanical traction', 80),
            ('97110', 'Therapeutic exercise', 75),
            ('29125', 'Application of splint', 200),
            ('98943', 'Spinal decompression', 400)
        ],
        'high_cost': [
            ('98943', 'Spinal decompression', 400),
            ('29125', 'Application of splint', 200),
            ('97012', 'Mechanical traction', 80)
        ]
    }
}

# Diagnosis codes
DIAGNOSIS_CODES = {
    'cardiology': ['I25.10', 'I48.91', 'I50.9', 'Z95.1', 'I10'],
    'dermatology': ['C44.92', 'L57.0', 'D23.9', 'L85.9', 'Q82.5'],
    'orthopedics': ['M25.561', 'M79.3', 'S72.001A', 'M23.90', 'M54.5'],
    'mental_health': ['F32.9', 'F41.9', 'F43.10', 'F90.9', 'F33.9'],
    'chiropractic': ['M54.5', 'M25.50', 'M79.3', 'S13.4XXA', 'M43.28']
}

class HealthcareClaimsGenerator:
    def __init__(self):
        self.claims_data = []
        self.providers = []
        self.patients = []
        self.claim_counter = 1
        
    def generate_providers(self, specialty, count=30):
        """Generate providers for a given specialty"""
        providers = []
        for _ in range(count):
            provider = {
                'provider_id': f"PRV{random.randint(100000, 999999)}",
                'npi': f"{random.randint(1000000000, 9999999999)}",
                'name': fake.company() + " Medical Center",
                'specialty': specialty,
                'address': fake.address(),
                'city': fake.city(),
                'state': fake.state_abbr(),
                'zip_code': fake.zipcode(),
                'latitude': fake.latitude(),
                'longitude': fake.longitude()
            }
            providers.append(provider)
        return providers
    
    def generate_patients(self, count=10000):
        """Generate patient demographics"""
        patients = []
        for _ in range(count):
            patient = {
                'patient_id': f"PT{random.randint(100000, 999999)}",
                'member_id': f"MBR{random.randint(1000000000, 9999999999)}",
                'first_name': fake.first_name(),
                'last_name': fake.last_name(),
                'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=85),
                'gender': random.choice(['M', 'F']),
                'address': fake.address(),
                'city': fake.city(),
                'state': fake.state_abbr(),
                'zip_code': fake.zipcode(),
                'latitude': fake.latitude(),
                'longitude': fake.longitude()
            }
            patients.append(patient)
        return patients
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate approximate distance between two coordinates"""
        return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 69  # Rough miles
    
    def generate_consistent_visit_pattern(self, provider, patients, start_date, end_date):
        """Pattern 1: Consistent patient visits per day"""
        claims = []
        daily_visits = random.choice([15, 20, 25, 30])  # Suspicious consistency
        
        current_date = start_date
        while current_date <= end_date:
            # Skip some weekends randomly but maintain the pattern most of the time
            if current_date.weekday() < 5 or random.random() < 0.3:
                for visit in range(daily_visits):
                    patient = random.choice(patients)
                    claim_lines = random.randint(3, 12)
                    
                    for line in range(claim_lines):
                        procedure = random.choice(PROCEDURE_CODES[provider['specialty']]['standard'])
                        diagnosis = random.choice(DIAGNOSIS_CODES[provider['specialty']])
                        
                        claim = self.create_claim_record(
                            provider, patient, current_date, procedure, diagnosis, line + 1
                        )
                        claims.append(claim)
            
            current_date += timedelta(days=1)
        
        return claims
    
    def generate_sudden_patient_influx(self, provider, patients, start_date, end_date):
        """Pattern 2: Sudden increase in patient volume"""
        claims = []
        normal_period_end = start_date + timedelta(days=180)
        
        # Normal period - fewer patients
        current_date = start_date
        while current_date <= normal_period_end:
            if random.random() < 0.7:  # 70% chance of seeing patients
                daily_patients = random.randint(3, 8)
                for _ in range(daily_patients):
                    patient = random.choice(patients[:2000])  # Limited patient pool
                    claim_lines = random.randint(2, 6)
                    
                    for line in range(claim_lines):
                        procedure = random.choice(PROCEDURE_CODES[provider['specialty']]['standard'])
                        diagnosis = random.choice(DIAGNOSIS_CODES[provider['specialty']])
                        
                        claim = self.create_claim_record(
                            provider, patient, current_date, procedure, diagnosis, line + 1
                        )
                        claims.append(claim)
            
            current_date += timedelta(days=1)
        
        # Sudden influx period - many new patients with expensive procedures
        while current_date <= end_date:
            if random.random() < 0.9:  # 90% chance of seeing patients
                daily_patients = random.randint(25, 40)  # Sudden increase
                for _ in range(daily_patients):
                    patient = random.choice(patients[5000:])  # New patient pool
                    claim_lines = random.randint(6, 12)
                    
                    for line in range(claim_lines):
                        # Higher chance of expensive procedures
                        procedure = random.choice(PROCEDURE_CODES[provider['specialty']]['high_cost'])
                        diagnosis = random.choice(DIAGNOSIS_CODES[provider['specialty']])
                        
                        claim = self.create_claim_record(
                            provider, patient, current_date, procedure, diagnosis, line + 1
                        )
                        claims.append(claim)
            
            current_date += timedelta(days=1)
        
        return claims
    
    def generate_geographic_anomalies(self, provider, patients, start_date, end_date):
        """Pattern 3: Geographic anomalies - distant patients"""
        claims = []
        
        # Filter patients that are far from provider (>100 miles)
        distant_patients = []
        for patient in patients:
            distance = self.calculate_distance(
                float(provider['latitude']), float(provider['longitude']),
                float(patient['latitude']), float(patient['longitude'])
            )
            if distance > 100:
                distant_patients.append(patient)
        
        if not distant_patients:
            distant_patients = patients[-1000:]  # Fallback
        
        current_date = start_date
        while current_date <= end_date:
            if random.random() < 0.6:
                daily_patients = random.randint(5, 15)
                for _ in range(daily_patients):
                    patient = random.choice(distant_patients)
                    claim_lines = random.randint(4, 10)
                    
                    for line in range(claim_lines):
                        procedure = random.choice(PROCEDURE_CODES[provider['specialty']]['high_cost'])
                        diagnosis = random.choice(DIAGNOSIS_CODES[provider['specialty']])
                        
                        claim = self.create_claim_record(
                            provider, patient, current_date, procedure, diagnosis, line + 1
                        )
                        claims.append(claim)
            
            current_date += timedelta(days=1)
        
        return claims
    
    def generate_excessive_single_procedure(self, provider, patients, start_date, end_date):
        """Pattern 4: Excessive billing for single procedure"""
        claims = []
        
        # Pick one high-cost procedure to overuse
        excessive_procedure = random.choice(PROCEDURE_CODES[provider['specialty']]['high_cost'])
        
        current_date = start_date
        while current_date <= end_date:
            if random.random() < 0.8:
                daily_patients = random.randint(10, 20)
                for _ in range(daily_patients):
                    patient = random.choice(patients)
                    claim_lines = random.randint(6, 12)
                    
                    for line in range(claim_lines):
                        # 80% chance of using the excessive procedure
                        if random.random() < 0.8:
                            procedure = excessive_procedure
                        else:
                            procedure = random.choice(PROCEDURE_CODES[provider['specialty']]['standard'])
                        
                        diagnosis = random.choice(DIAGNOSIS_CODES[provider['specialty']])
                        
                        claim = self.create_claim_record(
                            provider, patient, current_date, procedure, diagnosis, line + 1
                        )
                        claims.append(claim)
            
            current_date += timedelta(days=1)
        
        return claims
    
    def generate_identical_services_pattern(self, provider, patients, start_date, end_date):
        """Pattern 5: Identical services across diverse demographics"""
        claims = []
        
        # Define a standard "package" of services
        standard_package = PROCEDURE_CODES[provider['specialty']]['standard'][:5]
        
        current_date = start_date
        while current_date <= end_date:
            if random.random() < 0.7:
                daily_patients = random.randint(8, 16)
                for _ in range(daily_patients):
                    patient = random.choice(patients)
                    
                    # Every patient gets the exact same services
                    for line, procedure in enumerate(standard_package):
                        diagnosis = random.choice(DIAGNOSIS_CODES[provider['specialty']])
                        
                        claim = self.create_claim_record(
                            provider, patient, current_date, procedure, diagnosis, line + 1
                        )
                        claims.append(claim)
            
            current_date += timedelta(days=1)
        
        return claims
    
    def assign_fraud_flags(self, claim, pattern_type):
        """Assign fraud flags based on pattern type and claim characteristics"""
        if pattern_type == 'legitimate':
            return 'Clean'
        
        flag_mapping = {
            'consistent_visits': 'Volume_Anomaly',
            'sudden_influx': 'Patient_Mill',
            'geographic_anomaly': 'Geographic_Risk',
            'excessive_procedure': 'Procedure_Abuse',
            'identical_services': 'Cookie_Cutter'
        }
        
        return flag_mapping.get(pattern_type, 'Suspicious')
    
    def calculate_risk_score(self, claim, pattern_type):
        """Calculate risk score based on various factors"""
        if pattern_type == 'legitimate':
            return random.randint(1, 30)  # Low risk
        
        base_scores = {
            'consistent_visits': random.randint(70, 85),
            'sudden_influx': random.randint(85, 95),
            'geographic_anomaly': random.randint(75, 90),
            'excessive_procedure': random.randint(80, 95),
            'identical_services': random.randint(75, 88)
        }
        
        base_score = base_scores.get(pattern_type, 70)
        
        # Adjust based on claim characteristics
        if claim['charges'] > 5000:
            base_score += random.randint(3, 8)
        if claim['distance_miles'] > 100:
            base_score += random.randint(5, 10)
        if claim['procedure_code'] in ['33533', '27447', '22614']:  # High-cost procedures
            base_score += random.randint(3, 7)
        
        return min(base_score, 99)  # Cap at 99
    
    def create_claim_record(self, provider, patient, service_date, procedure, diagnosis, line_number):
        """Create a single claim record following UB04/CMS guidelines"""
        claim_id = f"CLM{self.claim_counter:010d}"
        
        # Add some variation to the procedure cost
        base_cost = procedure[2]
        actual_cost = base_cost * random.uniform(0.8, 1.3)
        
        claim = {
            # Header Information
            'claim_id': claim_id,
            'claim_line': line_number,
            'claim_type': 'Professional',
            'claim_status': random.choice(['Paid', 'Denied', 'Pending']),
            'submission_date': service_date + timedelta(days=random.randint(1, 30)),
            'processed_date': service_date + timedelta(days=random.randint(31, 60)),
            
            # Provider Information
            'provider_id': provider['provider_id'],
            'provider_npi': provider['npi'],
            'provider_name': provider['name'],
            'provider_specialty': provider['specialty'],
            'provider_address': provider['address'],
            'provider_city': provider['city'],
            'provider_state': provider['state'],
            'provider_zip': provider['zip_code'],
            
            # Patient Information
            'patient_id': patient['patient_id'],
            'member_id': patient['member_id'],
            'patient_name': f"{patient['first_name']} {patient['last_name']}",
            'patient_dob': patient['date_of_birth'],
            'patient_gender': patient['gender'],
            'patient_address': patient['address'],
            'patient_city': patient['city'],
            'patient_state': patient['state'],
            'patient_zip': patient['zip_code'],
            
            # Service Information
            'service_date': service_date,
            'procedure_code': procedure[0],
            'procedure_description': procedure[1],
            'diagnosis_code': diagnosis,
            'units': random.randint(1, 3),
            'charges': round(actual_cost, 2),
            'allowed_amount': round(actual_cost * random.uniform(0.7, 0.95), 2),
            'paid_amount': round(actual_cost * random.uniform(0.6, 0.9), 2),
            'patient_responsibility': round(actual_cost * random.uniform(0.1, 0.3), 2),
            
            # Additional Fields
            'place_of_service': random.choice(['11', '21', '22', '23']),  # Office, Hospital, etc.
            'modifier': random.choice(['', '25', '59', 'LT', 'RT']),
            'revenue_code': f"{random.randint(100, 999):03d}",
            'drg_code': f"{random.randint(1, 999):03d}" if random.random() < 0.3 else '',
            
            # Pattern Indicators (for analysis)
            'pattern_type': 'mixed',  # Will be updated by calling function
            'fraud_flag': 'Pending',  # Will be updated by calling function
            'risk_score': 50,  # Will be updated by calling function
            'distance_miles': round(self.calculate_distance(
                float(provider['latitude']), float(provider['longitude']),
                float(patient['latitude']), float(patient['longitude'])
            ), 2),
            
            # Additional Flags for Detection
            'high_cost_flag': 'Yes' if actual_cost > 2000 else 'No',
            'weekend_service': 'Yes' if service_date.weekday() >= 5 else 'No',
            'multiple_procedures': 'Yes',  # Will be updated based on claim lines
            'new_patient_flag': 'Unknown'  # Could be enhanced with patient history
        }
        
        if line_number == 1:
            self.claim_counter += 1
            
        return claim
        """Create a single claim record following UB04/CMS guidelines"""
        claim_id = f"CLM{self.claim_counter:010d}"
        
        # Add some variation to the procedure cost
        base_cost = procedure[2]
        actual_cost = base_cost * random.uniform(0.8, 1.3)
        
        claim = {
            # Header Information
            'claim_id': claim_id,
            'claim_line': line_number,
            'claim_type': 'Professional',
            'claim_status': random.choice(['Paid', 'Denied', 'Pending']),
            'submission_date': service_date + timedelta(days=random.randint(1, 30)),
            'processed_date': service_date + timedelta(days=random.randint(31, 60)),
            
            # Provider Information
            'provider_id': provider['provider_id'],
            'provider_npi': provider['npi'],
            'provider_name': provider['name'],
            'provider_specialty': provider['specialty'],
            'provider_address': provider['address'],
            'provider_city': provider['city'],
            'provider_state': provider['state'],
            'provider_zip': provider['zip_code'],
            
            # Patient Information
            'patient_id': patient['patient_id'],
            'member_id': patient['member_id'],
            'patient_name': f"{patient['first_name']} {patient['last_name']}",
            'patient_dob': patient['date_of_birth'],
            'patient_gender': patient['gender'],
            'patient_address': patient['address'],
            'patient_city': patient['city'],
            'patient_state': patient['state'],
            'patient_zip': patient['zip_code'],
            
            # Service Information
            'service_date': service_date,
            'procedure_code': procedure[0],
            'procedure_description': procedure[1],
            'diagnosis_code': diagnosis,
            'units': random.randint(1, 3),
            'charges': round(actual_cost, 2),
            'allowed_amount': round(actual_cost * random.uniform(0.7, 0.95), 2),
            'paid_amount': round(actual_cost * random.uniform(0.6, 0.9), 2),
            'patient_responsibility': round(actual_cost * random.uniform(0.1, 0.3), 2),
            
            # Additional Fields
            'place_of_service': random.choice(['11', '21', '22', '23']),  # Office, Hospital, etc.
            'modifier': random.choice(['', '25', '59', 'LT', 'RT']),
            'revenue_code': f"{random.randint(100, 999):03d}",
            'drg_code': f"{random.randint(1, 999):03d}" if random.random() < 0.3 else '',
            
            # Pattern Indicators (for analysis)
            'pattern_type': 'mixed',  # Will be updated by calling function
            'fraud_flag': 'Pending',  # Will be updated by calling function
            'risk_score': 50,  # Will be updated by calling function
            'distance_miles': round(self.calculate_distance(
                float(provider['latitude']), float(provider['longitude']),
                float(patient['latitude']), float(patient['longitude'])
            ), 2),
            
            # Additional Flags for Detection
            'high_cost_flag': 'Yes' if actual_cost > 2000 else 'No',
            'weekend_service': 'Yes' if service_date.weekday() >= 5 else 'No',
            'multiple_procedures': 'Yes',  # Will be updated based on claim lines
            'new_patient_flag': 'Unknown'  # Could be enhanced with patient history
        }
        
        if line_number == 1:
            self.claim_counter += 1
            
        return claim
    
    def generate_legitimate_claims(self, provider, patients, start_date, end_date):
        """Generate normal, legitimate claims without suspicious patterns"""
        claims = []
        
        current_date = start_date
        while current_date <= end_date:
            # Realistic variation in daily patient volume
            if current_date.weekday() < 5:  # Weekdays
                daily_patients = random.randint(5, 20)
            elif current_date.weekday() == 5:  # Saturday
                daily_patients = random.randint(2, 8) if random.random() < 0.4 else 0
            else:  # Sunday
                daily_patients = random.randint(1, 3) if random.random() < 0.1 else 0
            
            for _ in range(daily_patients):
                patient = random.choice(patients)
                
                # Realistic variation in services per visit
                claim_lines = random.choices([1, 2, 3, 4, 5, 6], weights=[30, 25, 20, 15, 8, 2])[0]
                
                for line in range(claim_lines):
                    # Mix of standard and high-cost procedures (realistic distribution)
                    if random.random() < 0.8:
                        procedure = random.choice(PROCEDURE_CODES[provider['specialty']]['standard'])
                    else:
                        procedure = random.choice(PROCEDURE_CODES[provider['specialty']]['high_cost'])
                    
                    diagnosis = random.choice(DIAGNOSIS_CODES[provider['specialty']])
                    
                    claim = self.create_claim_record(
                        provider, patient, current_date, procedure, diagnosis, line + 1
                    )
                    claim['pattern_type'] = 'legitimate'
                    claim['fraud_flag'] = 'Clean'
                    claim['risk_score'] = random.randint(1, 30)  # Low risk scores
                    claims.append(claim)
            
            current_date += timedelta(days=1)
        
        return claims
    
    def generate_all_claims(self):
        """Generate mix of legitimate and suspicious claims"""
        print("Generating providers and patients...")
        
        # Generate providers for each specialty (35+ each)
        all_providers = []
        for specialty in PROCEDURE_CODES.keys():
            providers = self.generate_providers(specialty, 35)
            all_providers.extend(providers)
        
        # Generate patients
        all_patients = self.generate_patients(15000)
        
        print(f"Generated {len(all_providers)} providers and {len(all_patients)} patients")
        
        # Date range for claims (1 year)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        all_claims = []
        
        # Pattern distribution - 70% legitimate, 30% suspicious
        patterns = [
            ('consistent_visits', self.generate_consistent_visit_pattern),
            ('sudden_influx', self.generate_sudden_patient_influx),
            ('geographic_anomaly', self.generate_geographic_anomalies),
            ('excessive_procedure', self.generate_excessive_single_procedure),
            ('identical_services', self.generate_identical_services_pattern)
        ]
        
        for i, provider in enumerate(all_providers):
            print(f"Processing provider {i+1}/{len(all_providers)}: {provider['name']}")
            
            # 70% chance of legitimate provider, 30% chance of suspicious
            if random.random() < 0.7:
                # Generate legitimate claims
                provider_claims = self.generate_legitimate_claims(provider, all_patients, start_date, end_date)
                pattern_name = 'legitimate'
            else:
                # Generate suspicious claims
                pattern_name, pattern_func = random.choice(patterns)
                provider_claims = pattern_func(provider, all_patients, start_date, end_date)
                
                # Update pattern flags for suspicious claims
                for claim in provider_claims:
                    claim['pattern_type'] = pattern_name
                    claim['fraud_flag'] = self.assign_fraud_flags(claim, pattern_name)
                    claim['risk_score'] = self.calculate_risk_score(claim, pattern_name)
            
            all_claims.extend(provider_claims)
            print(f"  Generated {len(provider_claims)} claims - Pattern: {pattern_name}")
        
        print(f"\nTotal claims generated: {len(all_claims):,}")
        return all_claims
    
    def save_claims_to_csv(self, claims, filename='healthcare_claims_with_fraud_patterns.csv'):
        """Save claims to CSV file"""
        df = pd.DataFrame(claims)
        df.to_csv(filename, index=False)
        print(f"Claims saved to {filename}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Total Claims: {len(df):,}")
        print(f"Unique Patients: {df['patient_id'].nunique():,}")
        print(f"Unique Providers: {df['provider_id'].nunique():,}")
        print(f"Total Charges: ${df['charges'].sum():,.2f}")
        print(f"Date Range: {df['service_date'].min()} to {df['service_date'].max()}")
        
        print(f"\nPattern Distribution:")
        pattern_counts = df['pattern_type'].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count:,} claims")
        
        print(f"\nFraud Flag Distribution:")
        fraud_flag_counts = df['fraud_flag'].value_counts()
        for flag, count in fraud_flag_counts.items():
            print(f"  {flag}: {count:,} claims")
        
        print(f"\nRisk Score Distribution:")
        print(f"  Low Risk (1-30): {len(df[df['risk_score'] <= 30]):,} claims")
        print(f"  Medium Risk (31-70): {len(df[(df['risk_score'] > 30) & (df['risk_score'] <= 70)]):,} claims")
        print(f"  High Risk (71-100): {len(df[df['risk_score'] > 70]):,} claims")
        
        print(f"\nHigh-Cost Claims:")
        high_cost_claims = len(df[df['high_cost_flag'] == 'Yes'])
        print(f"  Claims > $2,000: {high_cost_claims:,} ({high_cost_claims/len(df)*100:.1f}%)")
        
        print(f"\nSpecialty Distribution:")
        specialty_counts = df['provider_specialty'].value_counts()
        for specialty, count in specialty_counts.items():
            print(f"  {specialty}: {count:,} claims")

def main():
    """Main function to generate healthcare claims"""
    print("Healthcare Claims Generator with Fraud Patterns")
    print("=" * 50)
    
    generator = HealthcareClaimsGenerator()
    claims = generator.generate_all_claims()
    generator.save_claims_to_csv(claims)
    
    print("\nClaims generation completed successfully!")
    print("\nDataset includes:")
    print("• 70% legitimate claims (normal patterns)")
    print("• 30% suspicious claims with fraud patterns")
    print("\nFraud Detection Flags:")
    print("• Clean: Normal, legitimate claims")
    print("• Volume_Anomaly: Consistent daily visit patterns") 
    print("• Patient_Mill: Sudden patient volume increases")
    print("• Geographic_Risk: Patients traveling excessive distances")
    print("• Procedure_Abuse: Overuse of expensive procedures")
    print("• Cookie_Cutter: Identical services for all patients")
    print("\nRisk Scores: 1-30 (Low), 31-70 (Medium), 71-99 (High)")

if __name__ == "__main__":
    main()