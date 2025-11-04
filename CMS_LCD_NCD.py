"""
CMS Medicare Coverage Database Scraper for Payment Integrity
Uses web scraping and downloadable datasets from CMS MCD
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import csv
import io
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, quote
import time
from datetime import datetime


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
    policy_details: Dict = field(default_factory=dict)


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
    required_modifiers: Set[str] = field(default_factory=set)
    documentation_requirements: List[str] = field(default_factory=list)
    medical_necessity_criteria: List[str] = field(default_factory=list)
    full_text: str = ""


class CMSCoverageScraper:
    """
    Web scraper for CMS Medicare Coverage Database
    Works with the public-facing CMS MCD website
    """
    
    BASE_URL = "https://www.cms.gov/medicare-coverage-database"
    SEARCH_URL = f"{BASE_URL}/search/search.aspx"
    
    # State to MAC mapping
    STATE_TO_MAC = {
        "AL": "5", "FL": "7", "GA": "5", "NC": "5", "SC": "5", "TN": "5", "VA": "5", "WV": "5",
        "CT": "4", "DE": "4", "DC": "4", "MD": "4", "NJ": "4", "PA": "4",
        "IL": "6", "IN": "6", "KY": "6", "MI": "6", "MN": "6", "OH": "6", "WI": "6",
        "AK": "1", "AZ": "1", "CA": "1", "HI": "1", "ID": "1", "MT": "1", "ND": "1", 
        "OR": "1", "SD": "1", "UT": "1", "WA": "1", "WY": "1",
        "CO": "2", "KS": "2", "NM": "2", "OK": "2", "TX": "2",
        "AR": "4", "LA": "4", "MS": "4",
        "IA": "8", "MO": "8", "NE": "8",
        "MA": "4", "ME": "4", "NH": "4", "RI": "4", "VT": "4", "NY": "4"
    }
    
    def __init__(self, timeout: int = 30, cache_ttl: int = 3600):
        """Initialize scraper with caching"""
        self.timeout = timeout
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        })
        self.last_request = 0
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < 2.0:  # 2 second delay between requests
            time.sleep(2.0 - elapsed)
        self.last_request = time.time()
    
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
    
    def search_lcd(self, code: str, state: str) -> Optional[Dict]:
        """
        Search for LCD by procedure code and state
        
        Args:
            code: CPT/HCPCS code
            state: Two-letter state code
            
        Returns:
            LCD data dictionary or None
        """
        cache_key = f"lcd_{code}_{state}"
        cached = self._get_cached(cache_key)
        if cached:
            print(f"✓ Using cached LCD for {code}")
            return cached
        
        print(f"Searching CMS database for LCD: code={code}, state={state}")
        
        self._rate_limit()
        
        try:
            # Construct search URL with parameters
            params = {
                'KeyWord': code,
                'State': state,
                'DocType': 'All',
                'bc': 'AgAAAAAAAAA='
            }
            
            response = self.session.get(
                self.SEARCH_URL, 
                params=params, 
                timeout=self.timeout,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find search results
                results = soup.find_all('div', class_='results-row') or \
                         soup.find_all('tr', class_='results-row')
                
                for result in results:
                    title_elem = result.find('a', href=True)
                    if not title_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    
                    # Look for LCD documents
                    if 'LCD' in title or 'Local Coverage' in title:
                        lcd_url = title_elem.get('href', '')
                        if not lcd_url.startswith('http'):
                            lcd_url = urljoin(self.BASE_URL, lcd_url)
                        
                        # Extract document ID
                        doc_id_match = re.search(r'(L\d+|DL\d+)', title)
                        doc_id = doc_id_match.group(1) if doc_id_match else 'Unknown'
                        
                        lcd_data = {
                            'document_id': doc_id,
                            'title': title,
                            'url': lcd_url,
                            'type': 'LCD',
                            'code': code,
                            'state': state
                        }
                        
                        # Fetch full LCD details
                        full_data = self._fetch_lcd_details(lcd_url)
                        if full_data:
                            lcd_data.update(full_data)
                        
                        self._set_cache(cache_key, lcd_data)
                        print(f"✓ Found LCD: {title}")
                        return lcd_data
                
                print(f"✗ No LCD found for code {code}")
                
            else:
                print(f"✗ Search failed with status {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error searching LCD: {e}")
        
        return None
    
    def search_ncd(self, keyword: str) -> Optional[Dict]:
        """
        Search for NCD by keyword or code
        
        Args:
            keyword: Search keyword or code
            
        Returns:
            NCD data dictionary or None
        """
        cache_key = f"ncd_{keyword}"
        cached = self._get_cached(cache_key)
        if cached:
            print(f"✓ Using cached NCD for {keyword}")
            return cached
        
        print(f"Searching CMS database for NCD: keyword={keyword}")
        
        self._rate_limit()
        
        try:
            params = {
                'KeyWord': keyword,
                'DocType': 'NCD',
                'bc': 'AgAAAAAAAAA='
            }
            
            response = self.session.get(
                self.SEARCH_URL,
                params=params,
                timeout=self.timeout,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = soup.find_all('div', class_='results-row') or \
                         soup.find_all('tr', class_='results-row')
                
                for result in results:
                    title_elem = result.find('a', href=True)
                    if not title_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    
                    if 'NCD' in title or 'National Coverage' in title:
                        ncd_url = title_elem.get('href', '')
                        if not ncd_url.startswith('http'):
                            ncd_url = urljoin(self.BASE_URL, ncd_url)
                        
                        # Extract NCD ID
                        ncd_id_match = re.search(r'(\d+\.?\d*)', title)
                        ncd_id = ncd_id_match.group(1) if ncd_id_match else 'Unknown'
                        
                        ncd_data = {
                            'document_id': ncd_id,
                            'title': title,
                            'url': ncd_url,
                            'type': 'NCD',
                            'keyword': keyword
                        }
                        
                        # Fetch full NCD details
                        full_data = self._fetch_ncd_details(ncd_url)
                        if full_data:
                            ncd_data.update(full_data)
                        
                        self._set_cache(cache_key, ncd_data)
                        print(f"✓ Found NCD: {title}")
                        return ncd_data
                
                print(f"✗ No NCD found for keyword {keyword}")
                
            else:
                print(f"✗ Search failed with status {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error searching NCD: {e}")
        
        return None
    
    def _fetch_lcd_details(self, url: str) -> Optional[Dict]:
        """Fetch detailed LCD content from document page"""
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            details = {}
            
            # Extract coverage guidance section
            coverage_section = soup.find('div', {'id': 'CoverageGuidance'}) or \
                             soup.find('div', {'id': 'IndicationsandLimitations'})
            
            if coverage_section:
                details['coverage_text'] = coverage_section.get_text(separator='\n', strip=True)
            
            # Extract effective date
            effective_date = soup.find(string=re.compile(r'Effective Date', re.I))
            if effective_date:
                parent = effective_date.find_parent()
                if parent:
                    date_text = parent.get_text()
                    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', date_text)
                    if date_match:
                        details['effective_date'] = date_match.group(1)
            
            # Extract contractor info
            contractor = soup.find(string=re.compile(r'Contractor', re.I))
            if contractor:
                parent = contractor.find_parent()
                if parent:
                    details['contractor'] = parent.get_text(strip=True)
            
            return details
            
        except Exception as e:
            print(f"Error fetching LCD details: {e}")
            return None
    
    def _fetch_ncd_details(self, url: str) -> Optional[Dict]:
        """Fetch detailed NCD content from document page"""
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            details = {}
            
            # Extract coverage text
            coverage_section = soup.find('div', {'id': 'CoverageIndications'}) or \
                             soup.find('div', {'id': 'CoverageText'})
            
            if coverage_section:
                details['coverage_text'] = coverage_section.get_text(separator='\n', strip=True)
            
            # Extract implementation date
            impl_date = soup.find(string=re.compile(r'Implementation Date', re.I))
            if impl_date:
                parent = impl_date.find_parent()
                if parent:
                    date_text = parent.get_text()
                    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', date_text)
                    if date_match:
                        details['implementation_date'] = date_match.group(1)
            
            return details
            
        except Exception as e:
            print(f"Error fetching NCD details: {e}")
            return None


class PaymentIntegrityValidator:
    """
    Payment integrity validator using CMS web scraping
    """
    
    def __init__(self):
        """Initialize validator with scraper"""
        self.scraper = CMSCoverageScraper()
        self.validation_history: List[ValidationResult] = []
    
    def validate_claim(self, claim: Claim) -> ValidationResult:
        """
        Validate claim against LCD/NCD policies
        
        Args:
            claim: Claim object to validate
            
        Returns:
            ValidationResult with detailed findings
        """
        result = ValidationResult(
            claim_id=claim.claim_id,
            status=ValidationStatus.PENDING_REVIEW
        )
        
        print(f"\n{'='*70}")
        print(f"VALIDATING CLAIM: {claim.claim_id}")
        print(f"{'='*70}")
        
        # Step 1: Search for applicable policies
        policies = []
        
        # Determine state from MAC jurisdiction
        state = self._mac_to_state(claim.mac_jurisdiction)
        
        # Search LCD for each procedure code
        for code in claim.procedure_codes:
            lcd_data = self.scraper.search_lcd(code, state)
            if lcd_data:
                lcd_rule = self._parse_lcd_to_rule(lcd_data)
                policies.append(lcd_rule)
                result.applicable_policies.append(f"LCD: {lcd_rule.policy_id}")
                result.policy_details['lcd'] = lcd_data
        
        # Search NCD (takes precedence over LCD)
        for code in claim.procedure_codes:
            ncd_data = self.scraper.search_ncd(code)
            if ncd_data:
                ncd_rule = self._parse_ncd_to_rule(ncd_data)
                policies.insert(0, ncd_rule)  # NCDs have priority
                result.applicable_policies.insert(0, f"NCD: {ncd_rule.policy_id}")
                result.policy_details['ncd'] = ncd_data
                break
        
        if not policies:
            result.status = ValidationStatus.POLICY_NOT_FOUND
            result.findings.append("✗ No applicable LCD/NCD found in CMS database")
            result.recommended_actions.append("→ Manual review required")
            result.confidence_score = 50.0
            self.validation_history.append(result)
            return result
        
        # Step 2: Run validations
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
        
        # Step 3: Determine final status
        result.confidence_score = max(0.0, validation_score)
        
        if validation_score >= 85:
            result.status = ValidationStatus.APPROVED
            result.findings.append("✓ Claim meets coverage criteria")
        elif validation_score >= 60:
            result.status = ValidationStatus.PENDING_REVIEW
            result.recommended_actions.append("→ Request additional documentation")
        else:
            result.status = ValidationStatus.DENIED
            if not result.denial_reasons:
                result.denial_reasons.append(DenialReason.NOT_MEDICALLY_NECESSARY)
        
        if result.missing_requirements:
            result.status = ValidationStatus.MISSING_DOCUMENTATION
        
        self._generate_recommendations(result)
        self.validation_history.append(result)
        
        print(f"\n{'='*70}")
        print(f"RESULT: {result.status.value} (Confidence: {result.confidence_score:.1f}%)")
        print(f"{'='*70}\n")
        
        return result
    
    def _mac_to_state(self, mac: str) -> str:
        """Convert MAC jurisdiction to state code"""
        # Default mapping - in production, use actual state
        mac_to_states = {
            "1": "CA", "2": "TX", "3": "OH", "4": "NJ",
            "5": "SC", "6": "IL", "7": "FL", "8": "MO"
        }
        return mac_to_states.get(mac, "CA")
    
    def _parse_lcd_to_rule(self, lcd_data: Dict) -> CoverageRule:
        """Parse LCD data to structured rule"""
        rule = CoverageRule(
            policy_id=lcd_data.get('document_id', 'Unknown'),
            policy_type="LCD",
            title=lcd_data.get('title', 'Unknown LCD'),
            effective_date=lcd_data.get('effective_date'),
            contractor=lcd_data.get('contractor')
        )
        
        coverage_text = lcd_data.get('coverage_text', '')
        if coverage_text:
            rule.full_text = coverage_text
            self._parse_coverage_criteria(coverage_text, rule)
        
        return rule
    
    def _parse_ncd_to_rule(self, ncd_data: Dict) -> CoverageRule:
        """Parse NCD data to structured rule"""
        rule = CoverageRule(
            policy_id=ncd_data.get('document_id', 'Unknown'),
            policy_type="NCD",
            title=ncd_data.get('title', 'Unknown NCD'),
            effective_date=ncd_data.get('implementation_date')
        )
        
        coverage_text = ncd_data.get('coverage_text', '')
        if coverage_text:
            rule.full_text = coverage_text
            self._parse_coverage_criteria(coverage_text, rule)
        
        return rule
    
    def _parse_coverage_criteria(self, text: str, rule: CoverageRule):
        """Parse coverage criteria from policy text"""
        text_lower = text.lower()
        
        # Extract CPT/HCPCS codes
        code_patterns = re.findall(r'\b[0-9]{5}[A-Z]?\b|\b[A-Z][0-9]{4}\b', text)
        rule.covered_codes = set(code_patterns[:50])
        
        # Extract ICD-10 codes
        icd_patterns = re.findall(r'\b[A-Z][0-9]{2}\.?[0-9A-Z]{0,4}\b', text)
        rule.required_diagnoses = set(icd_patterns[:100])
        
        # Age restrictions
        if 'age' in text_lower:
            age_match = re.search(r'age[s]?\s+(\d+)\s*(?:to|through|-)\s*(\d+)', text_lower)
            if age_match:
                rule.age_restrictions = (int(age_match.group(1)), int(age_match.group(2)))
        
        # Frequency limits
        if 'once per year' in text_lower or 'annual' in text_lower:
            rule.frequency_limits['annual'] = 1
        elif 'once per month' in text_lower:
            rule.frequency_limits['monthly'] = 1
        
        # Documentation requirements
        doc_keywords = ['documentation required', 'medical record', 'physician notes']
        for keyword in doc_keywords:
            if keyword in text_lower:
                rule.documentation_requirements.append(f"Requires {keyword}")
        
        # Medical necessity
        if 'medically necessary' in text_lower:
            rule.medical_necessity_criteria.append("Must be medically necessary")
    
    def _validate_coverage(self, claim: Claim, policy: CoverageRule, result: ValidationResult) -> bool:
        """Validate procedure code coverage"""
        for code in claim.procedure_codes:
            if policy.covered_codes and code not in policy.covered_codes:
                result.findings.append(f"✗ Code {code} not explicitly listed in {policy.policy_type}")
                result.denial_reasons.append(DenialReason.NON_COVERED_SERVICE)
                return False
            else:
                result.findings.append(f"✓ Code {code} covered under {policy.policy_type}")
        return True
    
    def _validate_diagnoses(self, claim: Claim, policy: CoverageRule, result: ValidationResult) -> bool:
        """Validate diagnosis codes"""
        if not policy.required_diagnoses:
            return True
        
        has_matching = any(dx in policy.required_diagnoses for dx in claim.diagnosis_codes)
        if not has_matching:
            result.findings.append(f"✗ No matching diagnosis for {policy.policy_type}")
            result.denial_reasons.append(DenialReason.INCORRECT_DIAGNOSIS)
            return False
        
        result.findings.append(f"✓ Diagnosis codes align with policy")
        return True
    
    def _validate_frequency(self, claim: Claim, policy: CoverageRule, result: ValidationResult) -> bool:
        """Validate frequency limits"""
        if not policy.frequency_limits:
            return True
        
        for period, limit in policy.frequency_limits.items():
            if claim.units > limit:
                result.findings.append(f"✗ Exceeds {period} frequency limit")
                result.denial_reasons.append(DenialReason.FREQUENCY_EXCEEDED)
                return False
        
        result.findings.append(f"✓ Frequency limits met")
        return True
    
    def _validate_demographics(self, claim: Claim, policy: CoverageRule, result: ValidationResult) -> bool:
        """Validate age restrictions"""
        if policy.age_restrictions and claim.patient_age:
            min_age, max_age = policy.age_restrictions
            if not (min_age <= claim.patient_age <= max_age):
                result.findings.append(f"✗ Age {claim.patient_age} outside range {min_age}-{max_age}")
                result.denial_reasons.append(DenialReason.AGE_RESTRICTION)
                return False
            result.findings.append(f"✓ Age restriction met")
        return True
    
    def _check_documentation(self, policy: CoverageRule, result: ValidationResult):
        """Check documentation requirements"""
        if policy.documentation_requirements:
            result.missing_requirements.extend(policy.documentation_requirements)
    
    def _generate_recommendations(self, result: ValidationResult):
        """Generate actionable recommendations"""
        if result.status == ValidationStatus.DENIED:
            result.recommended_actions.append("→ Review clinical documentation")
            result.recommended_actions.append("→ Consider alternative procedures")
        elif result.status == ValidationStatus.PENDING_REVIEW:
            result.recommended_actions.append("→ Escalate to medical review")


def demo_validation():
    """Demonstration of payment integrity validation"""
    validator = PaymentIntegrityValidator()
    
    # Example claim
    claim = Claim(
        claim_id="CLM2025001",
        patient_id="PT12345",
        date_of_service="2025-10-22",
        procedure_codes=["99213"],  # Office visit
        diagnosis_codes=["E11.9", "I10"],  # Diabetes, Hypertension
        mac_jurisdiction="5",
        place_of_service="11",
        patient_age=62,
        billed_amount=150.00
    )
    
    result = validator.validate_claim(claim)
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Claim ID: {result.claim_id}")
    print(f"Status: {result.status.value}")
    print(f"Confidence Score: {result.confidence_score:.1f}%")
    print(f"\nPolicies Reviewed:")
    for policy in result.applicable_policies:
        print(f"  • {policy}")
    print(f"\nFindings:")
    for finding in result.findings:
        print(f"  {finding}")
    if result.recommended_actions:
        print(f"\nRecommended Actions:")
        for action in result.recommended_actions:
            print(f"  {action}")


if __name__ == "__main__":
    demo_validation()
