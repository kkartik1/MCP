"""
CMS NCCI (National Correct Coding Initiative) Validation Agent
WITH AUTOMATED DATA FETCHING FROM CMS WEBSITE

This agent validates medical billing codes against:
- PTP (Procedure-to-Procedure) edits for unbundling/mutually-exclusive codes
- MUE (Medically Unlikely Edits) for units-of-service limits

Data is automatically fetched from CMS quarterly releases.
"""

import json
import requests
import zipfile
import io
import csv
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EditType(Enum):
    """Types of NCCI edits"""
    UNBUNDLING = "unbundling"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    MUE_EXCEEDED = "mue_exceeded"


class ServiceType(Enum):
    """Service type categories"""
    PRACTITIONER = "PRA"
    HOSPITAL = "OPH"
    DME = "DME"


@dataclass
class NCCIResource:
    """NCCI resource configuration"""
    uri: str
    description: str


@dataclass
class PTPEdit:
    """Procedure-to-Procedure edit result"""
    column1_code: str
    column2_code: str
    modifier_indicator: str
    effective_date: str
    deletion_date: Optional[str]
    edit_type: str
    can_bypass_with_modifier: bool


@dataclass
class MUEEdit:
    """Medically Unlikely Edit result"""
    hcpcs_code: str
    mue_value: int
    mue_adjudication_indicator: str
    units_billed: int
    exceeds_limit: bool
    effective_date: str


@dataclass
class ValidationResult:
    """Overall validation result"""
    valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    ptp_edits: List[PTPEdit]
    mue_edits: List[MUEEdit]
    date_of_service: str


class CMSDataFetcher:
    """
    Fetches and parses NCCI data files from CMS website
    """
    
    # CMS file URL patterns
    BASE_URL = "https://www.cms.gov"
    PTP_PAGE = "/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-procedure-procedure-ptp-edits"
    MUE_PAGE = "/medicare/coding-billing/ncci-medicare/ncci-medicare-medically-unlikely-edits"
    
    # URL patterns for direct downloads
    PTP_URL_PATTERN = "/files/zip/medicare-ncci-{quarter}-{service_type}-ptp-edits-cci{service_code}-v{version}-f{file_num}.zip"
    MUE_URL_PATTERN = "/files/zip/medicare-ncci-{quarter}-{service_type}-mue-ccimu{service_code}-v{version}.zip"
    
    def __init__(self, cache_dir: str = "./ncci_cache"):
        """Initialize fetcher with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NCCI-Validation-Agent/1.0'
        })
    
    def get_current_quarter(self) -> Tuple[int, int]:
        """Get current year and quarter"""
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        return now.year, quarter
    
    def build_ptp_url(self, year: int, quarter: int, service_type: str = "practitioner", 
                      version: str = "313r0", file_num: int = 1) -> str:
        """
        Build PTP edit file URL
        
        Args:
            year: Year (e.g., 2025)
            quarter: Quarter (1-4)
            service_type: "practitioner" or "hospital"
            version: Version string (e.g., "313r0")
            file_num: File number (1-4, as files are split)
        """
        service_code = "pra" if service_type == "practitioner" else "oph"
        quarter_str = f"{year}q{quarter}"
        
        url = self.BASE_URL + self.PTP_URL_PATTERN.format(
            quarter=quarter_str,
            service_type=service_type,
            service_code=service_code,
            version=version,
            file_num=file_num
        )
        return url
    
    def build_mue_url(self, year: int, quarter: int, service_type: str = "practitioner",
                      version: str = "313r0") -> str:
        """
        Build MUE edit file URL
        
        Args:
            year: Year (e.g., 2025)
            quarter: Quarter (1-4)
            service_type: "practitioner", "hospital", or "dme"
            version: Version string
        """
        service_code_map = {
            "practitioner": "pra",
            "hospital": "oph",
            "dme": "dme"
        }
        service_code = service_code_map.get(service_type, "pra")
        quarter_str = f"{year}q{quarter}"
        
        url = self.BASE_URL + self.MUE_URL_PATTERN.format(
            quarter=quarter_str,
            service_type=service_type,
            service_code=service_code,
            version=version
        )
        return url
    
    def download_and_extract_zip(self, url: str) -> Dict[str, bytes]:
        """
        Download ZIP file and extract contents
        
        Returns:
            Dictionary mapping filenames to file contents
        """
        logger.info(f"Downloading: {url}")
        
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Extract ZIP contents
            files = {}
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for filename in zf.namelist():
                    files[filename] = zf.read(filename)
                    logger.info(f"  Extracted: {filename}")
            
            return files
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return {}
    
    def parse_ptp_file(self, file_content: bytes) -> List[Dict]:
        """
        Parse PTP edit file (tab-delimited text)
        
        Expected columns:
        - Column One Code
        - Column Two Code
        - Effective Date
        - Deletion Date
        - Modifier Indicator
        - PTP Edit Rationale
        """
        edits = []
        
        try:
            # Decode content
            text = file_content.decode('utf-8', errors='ignore')
            
            # Parse CSV
            reader = csv.DictReader(io.StringIO(text), delimiter='\t')
            
            for row in reader:
                # Clean up column names (remove BOM, whitespace)
                cleaned_row = {k.strip('\ufeff').strip(): v.strip() for k, v in row.items()}
                
                edits.append({
                    'column1': cleaned_row.get('Column One Code', ''),
                    'column2': cleaned_row.get('Column Two Code', ''),
                    'effective_date': cleaned_row.get('Effective Date', ''),
                    'deletion_date': cleaned_row.get('Deletion Date', ''),
                    'modifier_indicator': cleaned_row.get('Modifier Indicator', ''),
                    'rationale': cleaned_row.get('PTP Edit Rationale', '')
                })
            
            logger.info(f"Parsed {len(edits)} PTP edits")
            
        except Exception as e:
            logger.error(f"Failed to parse PTP file: {e}")
        
        return edits
    
    def parse_mue_file(self, file_content: bytes) -> Dict[str, Dict]:
        """
        Parse MUE edit file (tab-delimited text)
        
        Expected columns:
        - HCPCS/CPT Code
        - MUE Values
        - MUE Adjudication Indicator
        - Practitioner Services Edit Rationale
        """
        mues = {}
        
        try:
            text = file_content.decode('utf-8', errors='ignore')
            reader = csv.DictReader(io.StringIO(text), delimiter='\t')
            
            for row in reader:
                cleaned_row = {k.strip('\ufeff').strip(): v.strip() for k, v in row.items()}
                
                hcpcs = cleaned_row.get('HCPCS/CPT Code', '').strip()
                if not hcpcs:
                    continue
                
                try:
                    mue_value = int(cleaned_row.get('MUE Values', '0'))
                except (ValueError, TypeError):
                    mue_value = 0
                
                mues[hcpcs] = {
                    'mue': mue_value,
                    'mai': cleaned_row.get('MUE Adjudication Indicator', ''),
                    'rationale': cleaned_row.get('Practitioner Services Edit Rationale', ''),
                    'effective_date': cleaned_row.get('Effective Date', '')
                }
            
            logger.info(f"Parsed {len(mues)} MUE edits")
            
        except Exception as e:
            logger.error(f"Failed to parse MUE file: {e}")
        
        return mues
    
    def fetch_ptp_edits(self, year: int = None, quarter: int = None,
                       service_type: str = "practitioner") -> List[Dict]:
        """
        Fetch PTP edits for specified quarter
        
        Returns list of PTP edits
        """
        if year is None or quarter is None:
            year, quarter = self.get_current_quarter()
        
        all_edits = []
        
        # PTP files are split into multiple parts (usually 4)
        for file_num in range(1, 5):
            url = self.build_ptp_url(year, quarter, service_type, file_num=file_num)
            
            files = self.download_and_extract_zip(url)
            if not files:
                continue
            
            # Find the text file (usually .txt)
            for filename, content in files.items():
                if filename.endswith('.txt'):
                    edits = self.parse_ptp_file(content)
                    all_edits.extend(edits)
                    break
        
        return all_edits
    
    def fetch_mue_edits(self, year: int = None, quarter: int = None,
                       service_type: str = "practitioner") -> Dict[str, Dict]:
        """
        Fetch MUE edits for specified quarter
        
        Returns dictionary mapping HCPCS codes to MUE data
        """
        if year is None or quarter is None:
            year, quarter = self.get_current_quarter()
        
        url = self.build_mue_url(year, quarter, service_type)
        
        files = self.download_and_extract_zip(url)
        if not files:
            return {}
        
        # Find the text file
        for filename, content in files.items():
            if filename.endswith('.txt'):
                return self.parse_mue_file(content)
        
        return {}


class NCCIAgent:
    """
    CMS NCCI Validation Agent with automated data fetching
    """
    
    def __init__(self, config: Dict[str, Any], auto_fetch: bool = False):
        """
        Initialize NCCI agent
        
        Args:
            config: Configuration dictionary
            auto_fetch: If True, automatically fetch latest data from CMS
        """
        self.name = config.get("name", "cms-ncci")
        self.resources = [NCCIResource(**r) for r in config.get("resources", [])]
        self.tools = config.get("tools", [])
        
        # Data storage
        self.ptp_data: Dict[str, List[Dict]] = {}
        self.mue_data: Dict[str, Dict] = {}
        
        # Data fetcher
        self.fetcher = CMSDataFetcher()
        
        if auto_fetch:
            self.fetch_latest_data()
        else:
            self._load_sample_data()
    
    def fetch_latest_data(self, year: int = None, quarter: int = None):
        """
        Fetch latest NCCI data from CMS website
        
        Args:
            year: Year to fetch (default: current)
            quarter: Quarter to fetch (default: current)
        """
        logger.info("Fetching latest NCCI data from CMS...")
        
        # Fetch PTP edits
        ptp_edits = self.fetcher.fetch_ptp_edits(year, quarter, "practitioner")
        self._load_ptp_data(ptp_edits)
        
        # Fetch MUE edits
        mue_edits = self.fetcher.fetch_mue_edits(year, quarter, "practitioner")
        self.mue_data = mue_edits
        
        logger.info(f"Loaded {len(self.ptp_data)} PTP column1 codes")
        logger.info(f"Loaded {len(self.mue_data)} MUE codes")
    
    def _load_ptp_data(self, ptp_edits: List[Dict]):
        """Convert flat PTP list to indexed structure"""
        self.ptp_data = {}
        
        for edit in ptp_edits:
            col1 = edit['column1']
            if col1 not in self.ptp_data:
                self.ptp_data[col1] = []
            
            # Determine edit type from rationale
            rationale = edit.get('rationale', '').lower()
            edit_type = "mutually_exclusive" if "mutually" in rationale else "unbundling"
            
            self.ptp_data[col1].append({
                'column2': edit['column2'],
                'modifier_indicator': edit['modifier_indicator'],
                'effective_date': edit['effective_date'].replace('-', ''),
                'deletion_date': edit['deletion_date'].replace('-', '') if edit['deletion_date'] else None,
                'edit_type': edit_type
            })
    
    def _load_sample_data(self):
        """Load sample data for demo purposes"""
        logger.info("Loading sample NCCI data...")
        
        self.ptp_data = {
            "99213": [
                {
                    "column2": "36415",
                    "modifier_indicator": "1",
                    "effective_date": "20250101",
                    "deletion_date": None,
                    "edit_type": "unbundling"
                }
            ],
            "27447": [
                {
                    "column2": "27369",
                    "modifier_indicator": "0",
                    "effective_date": "20250101",
                    "deletion_date": None,
                    "edit_type": "unbundling"
                }
            ],
            "29881": [
                {
                    "column2": "29880",
                    "modifier_indicator": "1",
                    "effective_date": "20250101",
                    "deletion_date": None,
                    "edit_type": "mutually_exclusive"
                }
            ]
        }
        
        self.mue_data = {
            "99213": {"mue": 4, "mai": "2", "effective_date": "20250101"},
            "36415": {"mue": 2, "mai": "2", "effective_date": "20250101"},
            "27447": {"mue": 1, "mai": "1", "effective_date": "20250101"},
            "J0135": {"mue": 50, "mai": "2", "effective_date": "20250101"},
            "80053": {"mue": 1, "mai": "3", "effective_date": "20250101"},
        }
    
    def ptpcheck(self, hcpcs: List[str], dos: str) -> ValidationResult:
        """Check for PTP edits"""
        errors = []
        warnings = []
        ptp_edits = []
        
        dos_date = datetime.strptime(dos, "%Y%m%d")
        
        for i, code1 in enumerate(hcpcs):
            for code2 in hcpcs[i+1:]:
                edit = self._check_ptp_pair(code1, code2, dos_date)
                if edit:
                    ptp_edits.append(edit)
                    if edit.can_bypass_with_modifier:
                        warnings.append({
                            "type": edit.edit_type,
                            "message": f"PTP edit: {code1} bundles {code2}. May bypass with modifier.",
                            "codes": [code1, code2],
                            "modifier_indicator": edit.modifier_indicator
                        })
                    else:
                        errors.append({
                            "type": edit.edit_type,
                            "message": f"PTP edit: {code1} bundles {code2}. Cannot bypass.",
                            "codes": [code1, code2],
                            "modifier_indicator": edit.modifier_indicator
                        })
                
                edit_reverse = self._check_ptp_pair(code2, code1, dos_date)
                if edit_reverse:
                    ptp_edits.append(edit_reverse)
                    if edit_reverse.can_bypass_with_modifier:
                        warnings.append({
                            "type": edit_reverse.edit_type,
                            "message": f"PTP edit: {code2} bundles {code1}. May bypass with modifier.",
                            "codes": [code2, code1],
                            "modifier_indicator": edit_reverse.modifier_indicator
                        })
                    else:
                        errors.append({
                            "type": edit_reverse.edit_type,
                            "message": f"PTP edit: {code2} bundles {code1}. Cannot bypass.",
                            "codes": [code2, code1],
                            "modifier_indicator": edit_reverse.modifier_indicator
                        })
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            ptp_edits=ptp_edits,
            mue_edits=[],
            date_of_service=dos
        )
    
    def _check_ptp_pair(self, column1: str, column2: str, dos_date: datetime) -> Optional[PTPEdit]:
        """Check if code pair has PTP edit"""
        if column1 not in self.ptp_data:
            return None
        
        for edit in self.ptp_data[column1]:
            if edit["column2"] == column2:
                effective_date = datetime.strptime(edit["effective_date"], "%Y%m%d")
                deletion_date = datetime.strptime(edit["deletion_date"], "%Y%m%d") if edit["deletion_date"] else None
                
                if effective_date <= dos_date and (deletion_date is None or dos_date < deletion_date):
                    return PTPEdit(
                        column1_code=column1,
                        column2_code=column2,
                        modifier_indicator=edit["modifier_indicator"],
                        effective_date=edit["effective_date"],
                        deletion_date=edit["deletion_date"],
                        edit_type=edit["edit_type"],
                        can_bypass_with_modifier=(edit["modifier_indicator"] == "1")
                    )
        
        return None
    
    def muecheck(self, hcpcs: str, uom: int, dos: str) -> ValidationResult:
        """Check for MUE violations"""
        errors = []
        warnings = []
        mue_edits = []
        
        if hcpcs in self.mue_data:
            mue_info = self.mue_data[hcpcs]
            mue_value = mue_info["mue"]
            mai = mue_info["mai"]
            
            exceeds = uom > mue_value
            
            mue_edit = MUEEdit(
                hcpcs_code=hcpcs,
                mue_value=mue_value,
                mue_adjudication_indicator=mai,
                units_billed=uom,
                exceeds_limit=exceeds,
                effective_date=mue_info.get("effective_date", "")
            )
            mue_edits.append(mue_edit)
            
            if exceeds:
                mai_description = {
                    "1": "Line-level adjudication",
                    "2": "Claim-level adjudication",
                    "3": "Date of service adjudication"
                }
                
                errors.append({
                    "type": "mue_exceeded",
                    "message": f"MUE exceeded for {hcpcs}: {uom} units billed, limit is {mue_value}",
                    "code": hcpcs,
                    "units_billed": uom,
                    "mue_limit": mue_value,
                    "mai": mai,
                    "adjudication": mai_description.get(mai, "Unknown")
                })
        else:
            warnings.append({
                "type": "mue_not_found",
                "message": f"No MUE data found for {hcpcs}",
                "code": hcpcs
            })
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            ptp_edits=[],
            mue_edits=mue_edits,
            date_of_service=dos
        )
    
    def validate_claim(self, claim_data: Dict[str, Any]) -> ValidationResult:
        """Validate entire claim"""
        dos = claim_data["dos"]
        line_items = claim_data["line_items"]
        
        all_codes = [item["hcpcs"] for item in line_items]
        ptp_result = self.ptpcheck(all_codes, dos)
        
        all_mue_edits = []
        mue_errors = []
        mue_warnings = []
        
        for item in line_items:
            mue_result = self.muecheck(item["hcpcs"], item["units"], dos)
            all_mue_edits.extend(mue_result.mue_edits)
            mue_errors.extend(mue_result.errors)
            mue_warnings.extend(mue_result.warnings)
        
        return ValidationResult(
            valid=(len(ptp_result.errors) == 0 and len(mue_errors) == 0),
            errors=ptp_result.errors + mue_errors,
            warnings=ptp_result.warnings + mue_warnings,
            ptp_edits=ptp_result.ptp_edits,
            mue_edits=all_mue_edits,
            date_of_service=dos
        )
    
    def export_result(self, result: ValidationResult, format: str = "json") -> str:
        """Export validation result"""
        if format == "json":
            return json.dumps({
                "valid": result.valid,
                "date_of_service": result.date_of_service,
                "errors": result.errors,
                "warnings": result.warnings,
                "ptp_edits": [asdict(e) for e in result.ptp_edits],
                "mue_edits": [asdict(e) for e in result.mue_edits]
            }, indent=2)
        
        elif format == "text":
            lines = [
                f"NCCI Validation Result - DOS: {result.date_of_service}",
                f"Status: {'VALID' if result.valid else 'INVALID'}",
                ""
            ]
            
            if result.errors:
                lines.append("ERRORS:")
                for err in result.errors:
                    lines.append(f"  - {err['message']}")
                lines.append("")
            
            if result.warnings:
                lines.append("WARNINGS:")
                for warn in result.warnings:
                    lines.append(f"  - {warn['message']}")
                lines.append("")
            
            if result.ptp_edits:
                lines.append("PTP EDITS FOUND:")
                for edit in result.ptp_edits:
                    lines.append(f"  - {edit.column1_code} bundles {edit.column2_code} ({edit.edit_type})")
                    lines.append(f"    Modifier bypass: {'Yes' if edit.can_bypass_with_modifier else 'No'}")
                lines.append("")
            
            if result.mue_edits:
                lines.append("MUE CHECKS:")
                for edit in result.mue_edits:
                    status = "EXCEEDED" if edit.exceeds_limit else "OK"
                    lines.append(f"  - {edit.hcpcs_code}: {edit.units_billed}/{edit.mue_value} units [{status}]")
            
            return "\n".join(lines)
        
        return str(result)


# Example usage
if __name__ == "__main__":
    config = {
        "name": "cms-ncci",
        "resources": [
            {
                "uri": "https://www.cms.gov/medicare/coding-billing/ncci-medicare",
                "description": "PTP rules & manuals"
            },
            {
                "uri": "https://www.cms.gov/medicare/coding-billing/ncci-medically-unlikely-edits",
                "description": "MUE tables"
            }
        ]
    }
    
    # Initialize with sample data
    agent = NCCIAgent(config, auto_fetch=True)
    
    print("=" * 70)
    print("CMS NCCI Validation Agent - Demo with Sample Data")
    print("=" * 70)
    print()
    
    # Test with sample data
    result = agent.validate_claim({
        "dos": "20250315",
        "line_items": [
            {"hcpcs": "99213", "units": 1},
            {"hcpcs": "36415", "units": 1}
        ]
    })
    print(agent.export_result(result, "text"))
    print()
    
    # Example: Fetch real data (commented out - requires network access)
    print("To fetch real CMS data, use:")
    print("  agent = NCCIAgent(config, auto_fetch=True)")
    print("Or manually:")
    print("  agent.fetch_latest_data(year=2025, quarter=4)")