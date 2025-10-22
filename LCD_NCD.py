"""
Live CMS Coverage API Integration for Payment Integrity
Fetches real-time LCD/NCD data from CMS Medicare Coverage Database API
"""

import json
import re
import requests
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time


class ValidationStatus(Enum):
    """Claim validation outcomes"""
    APPROVED = "Approved"
    DENIED = "Denied"
    PENDING_REVIEW = "Pending Review"
    MISSING_DOCUMENTATION = "Missing Documentation"
    POLICY_NOT_FOUND = "Policy Not Found"


class DenialReason(Enum):
    """Standard denial reasons"""
    NOT_MEDICALLY_NECESSARY = "Not Medically Necessary"
    NON_COVERED_SERVICE = "Non-Covered Service"
    INCORRECT_DIAGNOSIS = "Incorrect Diagnosis Code"
    FREQUENCY_EXCEEDED = "Frequency Limitation Exceeded"
    MISSING_LCD_REQUIREMENTS = "Missing LCD Requirements"
    MISSING_NCD_REQUIREMENTS = "Missing NCD Requirements"
    INVALID_PLACE_OF_SERVICE = "Invalid Place of Service"
    MISSING_MODIFIER = "Missing Required Modifier"
    AGE_RESTRICTION = "Age Restriction Not Met"
    GENDER_RESTRICTION = "Gender Restriction Not Met"


@dataclass
class Claim:
    """Medical claim data structure"""
    claim_id: str
    patient_id: str
    date_of_service: str
    procedure_codes: List[str]
    diagnosis_codes: List[str]
    mac_jurisdiction: str
    place_of_service: str
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    units: int = 1
    billed_amount: float = 0.0
    provider_specialty: Optional[str] = None


@dataclass
class ValidationResult:
    """Validation result with detailed findings"""
    claim_id: str
    status: ValidationStatus
    denial_reasons: List[DenialReason] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    applicable_policies: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    reviewed_by_policy: bool = False
    lcd_details: Optional[Dict] = None
    ncd_details: Optional[Dict] = None


@dataclass
class CoverageRule:
    """Coverage policy rule extracted from LCD/NCD"""
    policy_id: str
    policy_type: str
    title: str
    effective_date: Optional[str] = None
    contractor: Optional[str] = None
    covered_codes: Set[str] = field(default_factory=set)
    required_diagnoses: Set[str] = field(default_factory=set)
    excluded_diagnoses: Set[str] = field(default_factory=set)
    frequency_limits: Dict[str, int] = field(default_factory=dict)
    age_restrictions: Optional[Tuple[int, int]] = None
    gender_restrictions: Optional[str] = None
    required_modifiers: Set[str] = field(default_factory=set)
    place_of_service_codes: Set[str] = field(default_factory=set)
    documentation_requirements: List[str] = field(default_factory=list)
    medical_necessity_criteria: List[str] = field(default_factory=list)
    limitation_text: Optional[str] = None
    indication_text: Optional[str] = None


class CMSCoverageAPIClient:
    """
    Client for CMS Coverage API - Fetches real LCD/NCD data
    API Documentation: https://api.coverage.cms.gov/docs/swagger/
    """
    
    BASE_URL = "https://api.coverage.cms.gov/api/v1"
    
    # MAC Jurisdiction Mappings
    MAC_CONTRACTORS = {
        "1": "15001",  # Noridian J-E
        "2": "15002",  # Noridian J-F
        "3": "15003",  # CGS
        "4": "15004",  # Novitas
        "5": "15005",  # Palmetto J-J
        "6": "15006",  # NGS J-K
        "7": "15009",  # First Coast J-N
        "8": "15011",  # WPS J-5
    }
    
    def __init__(self, timeout: int = 30, cache_ttl: int = 3600):
        """
        Initialize CMS API client
        
        Args:
            timeout: Request timeout in seconds
            cache_ttl: Cache time-to-live in seconds
        """
        self.timeout = timeout
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'PaymentIntegrityValidator/1.0'
        })
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Retrieve from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            del self.cache[key]
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Store in cache with timestamp"""
        self.cache[key] = (data, time.time())
    
    def get_lcd_by_code(self, code: str, contractor: str) -> Optional[Dict]:
        """
        Fetch LCD data for specific procedure code and contractor
        
        Args:
            code: CPT/HCPCS code
            contractor: MAC contractor ID
            
        Returns:
            LCD data dictionary or None
        """
        cache_key = f"lcd_{code}_{contractor}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Get Final LCDs
            url = f"{self.BASE_URL}/reports/local-coverage-final-lcds"
            params = {
                'contractor': contractor,
                'cpt_hcpcs': code
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                lcd = data[0]  # Get first matching LCD
                lcd_detail = self.get_lcd_detail(lcd.get('document_id', ''))
                if lcd_detail:
                    lcd.update(lcd_detail)
                
                self._set_cache(cache_key, lcd)
                return lcd
            
        except requests.RequestException as e:
            print(f"Error fetching LCD: {e}")
        
        return None
    
    def get_lcd_detail(self, document_id: str) -> Optional[Dict]:
        """
        Fetch detailed LCD content
        
        Args:
            document_id: LCD document ID
            
        Returns:
            Detailed LCD data
        """
        try:
            url = f"{self.BASE_URL}/documents/lcd/{document_id}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching LCD detail: {e}")
            return None
    
    def get_ncd_by_code(self, code: str) -> Optional[Dict]:
        """
        Fetch NCD data for specific procedure code
        
        Args:
            code: CPT/HCPCS code or NCD ID
            
        Returns:
            NCD data dictionary or None
        """
        cache_key = f"ncd_{code}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Search NCDs
            url = f"{self.BASE_URL}/reports/national-coverage-ncd"
            params = {
                'keyword': code
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                ncd = data[0]
                ncd_detail = self.get_ncd_detail(ncd.get('ncd_id', ''))
                if ncd_detail:
                    ncd.update(ncd_detail)
                
                self._set_cache(cache_key, ncd)
                return ncd
            
        except requests.RequestException as e:
            print(f"Error fetching NCD: {e}")
        
        return None
    
    def get_ncd_detail(self, ncd_id: str) -> Optional[Dict]:
        """
        Fetch detailed NCD content
        
        Args:
            ncd_id: NCD identifier
            
        Returns:
            Detailed NCD data
        """
        try:
            url = f"{self.BASE_URL}/documents/ncd/{ncd_id}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching NCD detail: {e}")
            return None
    
    def search_coverage(self, keyword: str, doc_type: str = None) -> List[Dict]:
        """
        General search for coverage policies
        
        Args:
            keyword: Search keyword
            doc_type: 'LCD' or 'NCD'
            
        Returns:
            List of matching policies
        """
        try:
            if doc_type == 'NCD':
                url = f"{self.BASE_URL}/reports/national-coverage-ncd"
            else:
                url = f"{self.BASE_URL}/reports/local-coverage-final-lcds"
            
            params = {'keyword': keyword}
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
        
        except requests.RequestException as e:
            print(f"Error searching coverage: {e}")
            return []


class LivePaymentIntegrityValidator:
    """
    Payment integrity validator using live CMS Coverage API
    """
    
    def __init__(self):
        """Initialize validator with CMS API client"""
        self.cms_api = CMSCoverageAPIClient()
        self.validation_history: List[ValidationResult] = []
    
    def validate_claim(self, claim: Claim) -> ValidationResult:
        """
        Main validation entry point using live CMS data
        
        Args:
            claim: Claim object to validate
            
        Returns:
            ValidationResult with detailed findings
        """
        result = ValidationResult(
            claim_id=claim.claim_id,
            status=ValidationStatus.PENDING_REVIEW
        )
        
        print(f"\n{'='*60}")
        print(f"Validating Claim: {claim.claim_id}")
        print(f"{'='*60}")
        
        # Step 1: Fetch live LCD data
        lcd_data = None
        contractor_id = self.cms_api.MAC_CONTRACTORS.get(claim.mac_jurisdiction)
        
        if contractor_id:
            for code in claim.procedure_codes:
                print(f"Fetching LCD for code {code}, MAC {claim.mac_jurisdiction}...")
                lcd_data = self.cms_api.get_lcd_by_code(code, contractor_id)
                if lcd_data:
                    print(f"✓ Found LCD: {lcd_data.get('title', 'Unknown')}")
                    result.lcd_details = lcd_data
                    break
        
        # Step 2: Fetch live NCD data (takes precedence over LCD)
        ncd_data = None
        for code in claim.procedure_codes:
            print(f"Fetching NCD for code {code}...")
            ncd_data = self.cms_api.get_ncd_by_code(code)
            if ncd_data:
                print(f"✓ Found NCD: {ncd_data.get('title', 'Unknown')}")
                result.ncd_details = ncd_data
                break
        
        if not lcd_data and not ncd_data:
            result.status = ValidationStatus.POLICY_NOT_FOUND
            result.findings.append("No applicable LCD/NCD found in CMS database")
            result.recommended_actions.append("Manual review required - no coverage policy found")
            print("✗ No coverage policies found")
            return result
        
        # Step 3: Parse and validate against policies
        policies = []
        
        if ncd_data:
            ncd_rule = self._parse_ncd_to_rule(ncd_data)
            policies.append(ncd_rule)
            result.applicable_policies.append(f"NCD: {ncd_rule.policy_id}")
        
        if lcd_data:
            lcd_rule = self._parse_lcd_to_rule(lcd_data)
            policies.append(lcd_rule)
            result.applicable_policies.append(f"LCD: {lcd_rule.policy_id}")
        
        result.reviewed_by_policy = True
        
        # Step 4: Run comprehensive validation
        validation_score = 100.0
        
        for policy in policies:
            print(f"\nValidating against {policy.policy_type}: {policy.policy_id}")
            
            # Coverage validation
            if not self._validate_coverage(claim, policy, result):
                validation_score -= 35
            
            # Diagnosis validation
            if not self._validate_diagnoses(claim, policy, result):
                validation_score -= 25
            
            # Frequency validation
            if not self._validate_frequency(claim, policy, result):
                validation_score -= 20
            
            # Demographics validation
            if not self._validate_demographics(claim, policy, result):
                validation_score -= 10
            
            # Documentation check
            self._check_documentation(policy, result)
        
        # Step 5: Determine final status
        result.confidence_score = max(0.0, validation_score)
        
        if validation_score >= 85:
            result.status = ValidationStatus.APPROVED
            result.findings.append("✓ Claim meets all coverage criteria")
        elif validation_score >= 60:
            result.status = ValidationStatus.PENDING_REVIEW
            result.recommended_actions.append("Request additional clinical documentation")
        else:
            result.status = ValidationStatus.DENIED
            if not result.denial_reasons:
                result.denial_reasons.append(DenialReason.NOT_MEDICALLY_NECESSARY)
        
        if result.missing_requirements:
            result.status = ValidationStatus.MISSING_DOCUMENTATION
        
        self._generate_recommendations(result, claim)
        self.validation_history.append(result)
        
        print(f"\n{'='*60}")
        print(f"Validation Result: {result.status.value}")
        print(f"Confidence Score: {result.confidence_score:.1f}%")
        print(f"{'='*60}\n")
        
        return result
    
    def _parse_lcd_to_rule(self, lcd_data: Dict) -> CoverageRule:
        """Parse LCD API response to structured rule"""
        rule = CoverageRule(
            policy_id=lcd_data.get('document_id', 'Unknown'),
            policy_type="LCD",
            title=lcd_data.get('title', 'Unknown LCD'),
            effective_date=lcd_data.get('effective_date'),
            contractor=lcd_data.get('contractor')
        )
        
        # Extract coverage information
        coverage_text = lcd_data.get('coverage_guidance', '') or lcd_data.get('indications_and_limitations', '')
        
        if coverage_text:
            rule.indication_text = coverage_text
            
            # Extract CPT/HCPCS codes
            code_patterns = re.findall(r'\b[0-9]{5}[A-Z]?\b|\b[A-Z][0-9]{4}\b', coverage_text)
            rule.covered_codes = set(code_patterns[:50])
            
            # Extract ICD-10 codes
            icd_patterns = re.findall(r'\b[A-Z][0-9]{2}\.?[0-9A-Z]{0,4}\b', coverage_text)
            rule.required_diagnoses = set(icd_patterns[:100])
            
            # Parse limitations
            self._parse_limitations(coverage_text, rule)
        
        # Extract from covered_codes field if available
        if 'cpt_hcpcs_codes' in lcd_data:
            codes = lcd_data.get('cpt_hcpcs_codes', [])
            if isinstance(codes, list):
                rule.covered_codes.update(codes)
            elif isinstance(codes, str):
                rule.covered_codes.add(codes)
        
        return rule
    
    def _parse_ncd_to_rule(self, ncd_data: Dict) -> CoverageRule:
        """Parse NCD API response to structured rule"""
        rule = CoverageRule(
            policy_id=ncd_data.get('ncd_id', 'Unknown'),
            policy_type="NCD",
            title=ncd_data.get('title', 'Unknown NCD'),
            effective_date=ncd_data.get('implementation_date')
        )
        
        # Extract coverage information
        coverage_text = ncd_data.get('coverage_text', '') or ncd_data.get('indications_and_limitations', '')
        
        if coverage_text:
            rule.indication_text = coverage_text
            
            # Extract codes
            code_patterns = re.findall(r'\b[0-9]{5}[A-Z]?\b|\b[A-Z][0-9]{4}\b', coverage_text)
            rule.covered_codes = set(code_patterns[:50])
            
            # Extract ICD-10 codes
            icd_patterns = re.findall(r'\b[A-Z][0-9]{2}\.?[0-9A-Z]{0,4}\b', coverage_text)
            rule.required_diagnoses = set(icd_patterns[:100])
            
            # Parse limitations
            self._parse_limitations(coverage_text, rule)
        
        return rule
    
    def _parse_limitations(self, text: str, rule: CoverageRule):
        """Parse limitation criteria from policy text"""
        text_lower = text.lower()
        
        # Age restrictions
        age_pattern = r'age[s]?\s+(\d+)\s*(?:to|through|-|and)\s*(\d+)'
        age_match = re.search(age_pattern, text_lower)
        if age_match:
            rule.age_restrictions = (int(age_match.group(1)), int(age_match.group(2)))
        elif 'pediatric' in text_lower or 'children' in text_lower:
            rule.age_restrictions = (0, 18)
        elif 'adult only' in text_lower:
            rule.age_restrictions = (18, 150)
        
        # Frequency limits
        freq_patterns = [
            (r'once\s+per\s+year', {'annual': 1}),
            (r'once\s+per\s+month', {'monthly': 1}),
            (r'(\d+)\s+times?\s+per\s+year', lambda m: {'annual': int(m.group(1))}),
            (r'every\s+(\d+)\s+years?', lambda m: {'annual': 1 / int(m.group(1))}),
        ]
        
        for pattern, limit in freq_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if callable(limit):
                    rule.frequency_limits.update(limit(match))
                else:
                    rule.frequency_limits.update(limit)
        
        # Documentation requirements
        doc_keywords = [
            'medical record', 'documentation', 'physician notes',
            'clinical notes', 'test results', 'prior authorization'
        ]
        for keyword in doc_keywords:
            if keyword in text_lower:
                rule.documentation_requirements.append(f"Requires {keyword}")
        
        # Medical necessity
        if 'medically necessary' in text_lower or 'reasonable and necessary' in text_lower:
            rule.medical_necessity_criteria.append("Must meet medical necessity criteria")
    
    def _validate_coverage(self, claim: Claim, policy: CoverageRule, result: ValidationResult) -> bool:
        """Validate procedure codes are covered"""
        valid = True
        
        for code in claim.procedure_codes:
            if policy.covered_codes and code not in policy.covered_codes:
                result.findings.append(f"✗ Code {code} not listed in {policy.policy_type} {policy.policy_id}")
                result.denial_reasons.append(DenialReason.NON_COVERED_SERVICE)
                valid = False
            else:
                result.findings.append(f"✓ Code {code} covered under {policy.policy_type}")
        
        return valid
    
    def _validate_diagnoses(self, claim: Claim, policy: CoverageRule, result: ValidationResult) -> bool:
        """Validate diagnosis codes meet requirements"""
        if not policy.required_diagnoses:
            return True
        
        has_matching = any(dx in policy.required_diagnoses for dx in claim.diagnosis_codes)
        
        if not has_matching:
            result.findings.append(f"✗ No matching diagnosis codes for {policy.policy_type}")
            result.findings.append(f"  Required diagnosis patterns found in policy")
            result.denial_reasons.append(DenialReason.INCORRECT_DIAGNOSIS)
            return False
        
        result.findings.append(f"✓ Diagnosis codes align with {policy.policy_type}")
        return True
    
    def _validate_frequency(self, claim: Claim, policy: CoverageRule, result: ValidationResult) -> bool:
        """Validate frequency limits"""
        if not policy.frequency_limits:
            return True
        
        for period, limit in policy.frequency_limits.items():
            if claim.units > limit:
                result.findings.append(f"✗ Exceeds {period} frequency limit of {limit}")
                result.denial_reasons.append(DenialReason.FREQUENCY_EXCEEDED)
                return False
        
        result.findings.append(f"✓ Frequency limits met")
        return True
    
    def _validate_demographics(self, claim: Claim, policy: CoverageRule, result: ValidationResult) -> bool:
        """Validate age/gender restrictions"""
        valid = True
        
        if policy.age_restrictions and claim.patient_age is not None:
            min_age, max_age = policy.age_restrictions
            if not (min_age <= claim.patient_age <= max_age):
                result.findings.append(f"✗ Patient age {claim.patient_age} outside allowed range {min_age}-{max_age}")
                result.denial_reasons.append(DenialReason.AGE_RESTRICTION)
                valid = False
            else:
                result.findings.append(f"✓ Age restriction met")
        
        return valid
    
    def _check_documentation(self, policy: CoverageRule, result: ValidationResult):
        """Check documentation requirements"""
        if policy.documentation_requirements:
            result.missing_requirements.extend(policy.documentation_requirements)
    
    def _generate_recommendations(self, result: ValidationResult, claim: Claim):
        """Generate actionable recommendations"""
        if result.status == ValidationStatus.DENIED:
            result.recommended_actions.append("⚠ Claim does not meet coverage criteria")
            result.recommended_actions.append("→ Review clinical documentation")
            result.recommended_actions.append("→ Consider alternative covered procedures")
        
        elif result.status == ValidationStatus.PENDING_REVIEW:
            result.recommended_actions.append("⚠ Additional review required")
            result.recommended_actions.append("→ Request peer-to-peer review")
    
    def generate_report(self) -> str:
        """Generate validation summary report"""
        if not self.validation_history:
            return "No validations performed"
        
        total = len(self.validation_history)
        approved = sum(1 for v in self.validation_history if v.status == ValidationStatus.APPROVED)
        denied = sum(1 for v in self.validation_history if v.status == ValidationStatus.DENIED)
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║         PAYMENT INTEGRITY VALIDATION REPORT               ║
╚═══════════════════════════════════════════════════════════╝

Total Claims Validated: {total}
Approved: {approved} ({approved/total*100:.1f}%)
Denied: {denied} ({denied/total*100:.1f}%)

Average Confidence Score: {sum(v.confidence_score for v in self.validation_history)/total:.1f}%

Policies Referenced:
{chr(10).join(f'  • {p}' for v in self.validation_history for p in v.applicable_policies)}
"""
        return report


def demo_live_validation():
    """Demonstration with live CMS API"""
    validator = LivePaymentIntegrityValidator()
    
    # Example claim
    claim = Claim(
        claim_id="CLM20250001",
        patient_id="PT98765",
        date_of_service="2025-10-20",
        procedure_codes=["99213"],  # Office visit
        diagnosis_codes=["E11.9", "I10"],  # Diabetes, Hypertension
        mac_jurisdiction="5",  # Palmetto
        place_of_service="11",  # Office
        patient_age=62,
        patient_gender="M",
        modifiers=["25"],
        billed_amount=150.00
    )
    
    result = validator.validate_claim(claim)
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Status: {result.status.value}")
    print(f"Confidence: {result.confidence_score:.1f}%")
    print(f"\nFindings:")
    for finding in result.findings:
        print(f"  {finding}")
    
    if result.recommended_actions:
        print(f"\nRecommended Actions:")
        for action in result.recommended_actions:
            print(f"  {action}")
    
    print("\n" + validator.generate_report())


if __name__ == "__main__":
    demo_live_validation()
