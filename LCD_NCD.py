"""
Medicare Coverage Policy Agent
Automates retrieval and search of NCD/LCD from CMS Medicare Coverage Database
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class PolicyType(Enum):
    """Coverage policy types"""
    NCD = "National Coverage Determination"
    LCD = "Local Coverage Determination"


@dataclass
class CoveragePolicy:
    """Coverage policy data structure"""
    policy_id: str
    title: str
    policy_type: PolicyType
    description: str
    effective_date: Optional[str] = None
    contractor: Optional[str] = None
    mac: Optional[str] = None
    full_text: Optional[str] = None
    url: Optional[str] = None


class MedicareCoverageAgent:
    """
    Agent for automated Medicare coverage policy retrieval and search
    """
    
    BASE_URL = "https://www.cms.gov/medicare-coverage-database"
    SEARCH_URL = f"{BASE_URL}/search.aspx"
    
    # MAC (Medicare Administrative Contractor) mappings
    MAC_MAPPINGS = {
        "1": "Noridian Healthcare Solutions (J-E)",
        "2": "Noridian Healthcare Solutions (J-F)",
        "3": "CGS Administrators",
        "4": "Novitas Solutions",
        "5": "Palmetto GBA (J-J)",
        "6": "National Government Services (J-K)",
        "7": "First Coast Service Options (J-N)",
        "8": "WPS Government Health Administrators (J-5)",
        "A": "Noridian Healthcare Solutions (J-E)",
        "B": "CGS Administrators (J-15)",
        "C": "National Government Services (J-6)",
        "D": "Novitas Solutions (J-H)",
        "E": "Palmetto GBA (J-J)",
        "F": "First Coast Service Options (J-N)",
    }
    
    def __init__(self, timeout: int = 30, rate_limit: float = 1.0):
        """
        Initialize the agent
        
        Args:
            timeout: Request timeout in seconds
            rate_limit: Minimum seconds between requests
        """
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _rate_limit_wait(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[str]:
        """
        Make HTTP request with error handling
        
        Args:
            url: Target URL
            params: Query parameters
            
        Returns:
            Response text or None on failure
        """
        self._rate_limit_wait()
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def lookup_lcd(self, code: str, mac: str) -> Optional[CoveragePolicy]:
        """
        Lookup LCD by procedure code and MAC jurisdiction
        
        Args:
            code: CPT/HCPCS code
            mac: MAC jurisdiction identifier
            
        Returns:
            CoveragePolicy object or None
        """
        print(f"Looking up LCD for code: {code}, MAC: {mac}")
        
        # Search parameters for LCD lookup
        params = {
            'KeyWord': code,
            'Jurisdiction': mac,
            'DocType': 'LCD',
            'bc': 'AAAAIAAAAAAAAA%3d%3d&'
        }
        
        html = self._make_request(self.SEARCH_URL, params)
        if not html:
            return None
        
        return self._parse_search_results(html, PolicyType.LCD)
    
    def lookup_ncd(self, keyword: str) -> Optional[CoveragePolicy]:
        """
        Lookup NCD by keyword or topic
        
        Args:
            keyword: Search keyword or NCD number
            
        Returns:
            CoveragePolicy object or None
        """
        print(f"Looking up NCD for keyword: {keyword}")
        
        params = {
            'KeyWord': keyword,
            'DocType': 'NCD',
            'bc': 'AAAAIAAAAAAAAA%3d%3d&'
        }
        
        html = self._make_request(self.SEARCH_URL, params)
        if not html:
            return None
        
        return self._parse_search_results(html, PolicyType.NCD)
    
    def search_coverage(
        self, 
        keyword: str, 
        policy_type: Optional[PolicyType] = None,
        contractor: Optional[str] = None
    ) -> List[CoveragePolicy]:
        """
        General coverage policy search
        
        Args:
            keyword: Search term
            policy_type: Filter by NCD or LCD
            contractor: Filter by MAC contractor
            
        Returns:
            List of matching CoveragePolicy objects
        """
        print(f"Searching coverage for: {keyword}")
        
        params = {'KeyWord': keyword}
        
        if policy_type:
            params['DocType'] = 'NCD' if policy_type == PolicyType.NCD else 'LCD'
        
        if contractor:
            params['Jurisdiction'] = contractor
        
        html = self._make_request(self.SEARCH_URL, params)
        if not html:
            return []
        
        return self._parse_multiple_results(html)
    
    def _parse_search_results(self, html: str, policy_type: PolicyType) -> Optional[CoveragePolicy]:
        """Parse first search result from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find first result entry
        result = soup.find('div', class_='results-row') or soup.find('tr', class_='results-row')
        
        if not result:
            print("No results found")
            return None
        
        # Extract policy details
        title_elem = result.find('a', href=True)
        if not title_elem:
            return None
        
        policy_id = self._extract_policy_id(title_elem.text)
        title = title_elem.text.strip()
        url = self.BASE_URL + '/' + title_elem['href'] if not title_elem['href'].startswith('http') else title_elem['href']
        
        # Extract description
        desc_elem = result.find('div', class_='result-description') or result.find_next('td')
        description = desc_elem.text.strip() if desc_elem else ""
        
        policy = CoveragePolicy(
            policy_id=policy_id,
            title=title,
            policy_type=policy_type,
            description=description,
            url=url
        )
        
        # Fetch full text if URL available
        if url:
            policy.full_text = self._fetch_policy_text(url)
        
        return policy
    
    def _parse_multiple_results(self, html: str) -> List[CoveragePolicy]:
        """Parse multiple search results"""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        result_rows = soup.find_all('div', class_='results-row') or soup.find_all('tr', class_='results-row')
        
        for row in result_rows[:10]:  # Limit to first 10 results
            title_elem = row.find('a', href=True)
            if not title_elem:
                continue
            
            title = title_elem.text.strip()
            url = self.BASE_URL + '/' + title_elem['href'] if not title_elem['href'].startswith('http') else title_elem['href']
            
            # Determine policy type from title or context
            policy_type = PolicyType.NCD if 'NCD' in title or 'National' in title else PolicyType.LCD
            
            policy_id = self._extract_policy_id(title)
            
            desc_elem = row.find('div', class_='result-description') or row.find_next('td')
            description = desc_elem.text.strip() if desc_elem else ""
            
            results.append(CoveragePolicy(
                policy_id=policy_id,
                title=title,
                policy_type=policy_type,
                description=description,
                url=url
            ))
        
        return results
    
    def _extract_policy_id(self, text: str) -> str:
        """Extract policy ID from text"""
        # Match NCD format (e.g., "210.1") or LCD format (e.g., "L12345")
        match = re.search(r'(NCD\s*)?(\d+\.\d+)|(L\d+)', text, re.IGNORECASE)
        return match.group(0) if match else "Unknown"
    
    def _fetch_policy_text(self, url: str) -> Optional[str]:
        """Fetch full policy text from detail page"""
        html = self._make_request(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for main content area
        content = soup.find('div', {'id': 'mainContent'}) or soup.find('div', class_='content')
        
        if content:
            return content.get_text(separator='\n', strip=True)
        
        return None
    
    def get_mac_name(self, mac_code: str) -> str:
        """Get MAC contractor name from code"""
        return self.MAC_MAPPINGS.get(mac_code.upper(), f"MAC {mac_code}")
    
    def export_policy(self, policy: CoveragePolicy, format: str = 'json') -> str:
        """
        Export policy to specified format
        
        Args:
            policy: CoveragePolicy object
            format: Output format ('json', 'text', 'html')
            
        Returns:
            Formatted policy data
        """
        if format == 'json':
            return json.dumps({
                'policy_id': policy.policy_id,
                'title': policy.title,
                'type': policy.policy_type.value,
                'description': policy.description,
                'effective_date': policy.effective_date,
                'contractor': policy.contractor,
                'mac': policy.mac,
                'url': policy.url,
                'full_text': policy.full_text[:500] + '...' if policy.full_text else None
            }, indent=2)
        
        elif format == 'text':
            return f"""
Policy ID: {policy.policy_id}
Title: {policy.title}
Type: {policy.policy_type.value}
Description: {policy.description}
URL: {policy.url}

Full Text:
{policy.full_text or 'Not available'}
"""
        
        elif format == 'html':
            return f"""
<div class="coverage-policy">
    <h2>{policy.title}</h2>
    <p><strong>Policy ID:</strong> {policy.policy_id}</p>
    <p><strong>Type:</strong> {policy.policy_type.value}</p>
    <p><strong>Description:</strong> {policy.description}</p>
    <a href="{policy.url}">View on CMS Website</a>
</div>
"""


def main():
    """Example usage"""
    agent = MedicareCoverageAgent()
    
    # Example 1: Lookup LCD by code and MAC
    print("\n=== Example 1: LCD Lookup ===")
    lcd = agent.lookup_lcd(code="99213", mac="5")
    if lcd:
        print(f"Found: {lcd.title}")
        print(agent.export_policy(lcd, format='json'))
    
    # Example 2: Lookup NCD
    print("\n=== Example 2: NCD Lookup ===")
    ncd = agent.lookup_ncd(keyword="cardiac pacemaker")
    if ncd:
        print(f"Found: {ncd.title}")
        print(f"Description: {ncd.description[:200]}...")
    
    # Example 3: General search
    print("\n=== Example 3: General Search ===")
    results = agent.search_coverage(keyword="diabetes", policy_type=PolicyType.NCD)
    print(f"Found {len(results)} results:")
    for i, policy in enumerate(results[:3], 1):
        print(f"{i}. {policy.title}")
    
    # Example 4: Get MAC info
    print("\n=== Example 4: MAC Information ===")
    print(f"MAC 5: {agent.get_mac_name('5')}")


if __name__ == "__main__":
    main()