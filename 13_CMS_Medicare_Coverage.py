"""
Enhanced Medicare Coverage Determination System
================================================
Processes NCD, LCD, and Article databases to determine Medicare coverage.

Requirements:
pip install pandas pyodbc chromadb ollama
"""

import os
import json
import pandas as pd
import pyodbc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import chromadb
from chromadb.config import Settings
import ollama
from datetime import datetime


@dataclass
class NCDDocument:
    """Structured representation of an NCD document"""
    ncd_id: str
    version_num: int
    ncd_section: str
    section_title: str
    coverage_level: str
    effective_date: str
    item_service_desc: Optional[str]
    indication_limitation: Optional[str]
    cross_reference: Optional[str]
    other_text: Optional[str]
    benefit_categories: List[str]
    keywords: Optional[str]
    doc_type: str = "NCD"
    
    def to_text(self) -> str:
        """Convert NCD document to searchable text"""
        parts = [
            f"[NCD Document]",
            f"NCD Section: {self.ncd_section}",
            f"Title: {self.section_title}",
            f"Coverage Level: {self.coverage_level}",
            f"Effective Date: {self.effective_date}",
        ]
        
        if self.item_service_desc:
            parts.append(f"\nItem/Service Description:\n{self.item_service_desc}")
        
        if self.indication_limitation:
            parts.append(f"\nIndications and Limitations:\n{self.indication_limitation}")
        
        if self.cross_reference:
            parts.append(f"\nCross References:\n{self.cross_reference}")
        
        if self.other_text:
            parts.append(f"\nAdditional Information:\n{self.other_text}")
        
        if self.benefit_categories:
            parts.append(f"\nBenefit Categories: {', '.join(self.benefit_categories)}")
        
        if self.keywords:
            parts.append(f"\nKeywords: {self.keywords}")
        
        return "\n".join(parts)


@dataclass
class LCDDocument:
    """Structured representation of an LCD document"""
    lcd_id: str
    lcd_version: int
    title: str
    indication: Optional[str]
    coding_guidelines: Optional[str]
    doc_reqs: Optional[str]
    source_info: Optional[str]
    effective_date: Optional[str]
    end_date: Optional[str]
    coverage_level: str
    hcpc_codes: List[str]
    icd10_covered: List[str]
    icd10_noncovered: List[str]
    contractor_info: Optional[str]
    keywords: Optional[str]
    doc_type: str = "LCD"
    
    def to_text(self) -> str:
        """Convert LCD document to searchable text"""
        parts = [
            f"[LCD Document]",
            f"LCD ID: L{self.lcd_id}",
            f"Title: {self.title}",
            f"Coverage Level: {self.coverage_level}",
        ]
        
        if self.effective_date:
            parts.append(f"Effective Date: {self.effective_date}")
        
        if self.hcpc_codes:
            parts.append(f"\nCPT/HCPCS Codes: {', '.join(self.hcpc_codes[:20])}")
            if len(self.hcpc_codes) > 20:
                parts.append(f"... and {len(self.hcpc_codes) - 20} more codes")
        
        if self.indication:
            parts.append(f"\nCoverage Indications:\n{self.indication}")
        
        if self.coding_guidelines:
            parts.append(f"\nCoding Guidelines:\n{self.coding_guidelines}")
        
        if self.doc_reqs:
            parts.append(f"\nDocumentation Requirements:\n{self.doc_reqs}")
        
        if self.icd10_covered:
            parts.append(f"\nCovered ICD-10 Codes: {', '.join(self.icd10_covered[:20])}")
        
        if self.icd10_noncovered:
            parts.append(f"\nNon-Covered ICD-10 Codes: {', '.join(self.icd10_noncovered[:10])}")
        
        if self.source_info:
            parts.append(f"\nSource Information:\n{self.source_info}")
        
        if self.keywords:
            parts.append(f"\nKeywords: {self.keywords}")
        
        return "\n".join(parts)


@dataclass
class ArticleDocument:
    """Structured representation of an Article document"""
    article_id: str
    article_version: int
    article_type: str
    title: str
    description: str
    effective_date: Optional[str]
    end_date: Optional[str]
    hcpc_codes: List[str]
    icd10_covered: List[str]
    icd10_noncovered: List[str]
    revenue_codes: List[str]
    bill_codes: List[str]
    contractor_info: Optional[str]
    keywords: Optional[str]
    doc_type: str = "Article"
    
    def to_text(self) -> str:
        """Convert Article document to searchable text"""
        parts = [
            f"[Article Document]",
            f"Article ID: A{self.article_id}",
            f"Article Type: {self.article_type}",
            f"Title: {self.title}",
        ]
        
        if self.effective_date:
            parts.append(f"Effective Date: {self.effective_date}")
        
        if self.hcpc_codes:
            parts.append(f"\nCPT/HCPCS Codes: {', '.join(self.hcpc_codes[:20])}")
        
        if self.description:
            parts.append(f"\nDescription:\n{self.description}")
        
        if self.icd10_covered:
            parts.append(f"\nCovered ICD-10 Codes: {', '.join(self.icd10_covered[:20])}")
        
        if self.icd10_noncovered:
            parts.append(f"\nNon-Covered ICD-10 Codes: {', '.join(self.icd10_noncovered[:10])}")
        
        if self.revenue_codes:
            parts.append(f"\nRevenue Codes: {', '.join(self.revenue_codes[:10])}")
        
        if self.keywords:
            parts.append(f"\nKeywords: {self.keywords}")
        
        return "\n".join(parts)


class DatabaseExtractor:
    """Extract data from Medicare .mdb databases"""
    
    def __init__(self, ncd_path: str, lcd_path: str = None, article_path: str = None):
        self.ncd_path = ncd_path
        self.lcd_path = lcd_path
        self.article_path = article_path
        self.coverage_levels = {
            '1': 'Full Coverage',
            '2': 'Restricted Coverage',
            '3': 'No Coverage'
        }
    
    def connect(self, db_path: str):
        """Create database connection"""
        conn_str = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            rf"DBQ={db_path};"
        )
        return pyodbc.connect(conn_str)
    
    def extract_ncds(self) -> List[NCDDocument]:
        """Extract all NCD documents"""
        conn = self.connect(self.ncd_path)
        cursor = conn.cursor()
        
        ncd_query = """
        SELECT 
            NCD_id, NCD_vrsn_num, NCD_mnl_sect, NCD_mnl_sect_title,
            cvrg_lvl_cd, NCD_efctv_dt, itm_srvc_desc, indctn_lmtn,
            xref_txt, othr_txt, ncd_keyword
        FROM NCD_TRKG
        ORDER BY NCD_id, NCD_vrsn_num DESC
        """
        
        ncds = []
        cursor.execute(ncd_query)

        for row in cursor.fetchall():
            ncd_id = row.NCD_id
            version = row.NCD_vrsn_num
            
            # Get benefit categories
            benefit_query = f"""
            SELECT b.bnft_ctgry_desc
            FROM NCD_TRKG_BNFT_XREF x
            INNER JOIN NCD_BNFT_CTGRY_REF b ON x.bnft_ctgry_cd = b.bnft_ctgry_cd
            WHERE x.NCD_id = {ncd_id} AND x.NCD_vrsn_num = {version}
            """
            cursor.execute(benefit_query)
            benefits = [b[0] for b in cursor.fetchall()]
            
            ncd = NCDDocument(
                ncd_id=str(ncd_id),
                version_num=version,
                ncd_section=row.NCD_mnl_sect or "",
                section_title=row.NCD_mnl_sect_title or "",
                coverage_level=self.coverage_levels.get(str(row.cvrg_lvl_cd), "Unknown"),
                effective_date=str(row.NCD_efctv_dt) if row.NCD_efctv_dt else "",
                item_service_desc=row.itm_srvc_desc,
                indication_limitation=row.indctn_lmtn,
                cross_reference=row.xref_txt,
                other_text=row.othr_txt,
                benefit_categories=benefits,
                keywords=row.ncd_keyword
            )
            ncds.append(ncd)
        
        conn.close()
        return ncds
    
    def extract_lcds(self) -> List[LCDDocument]:
        """Extract all LCD documents"""
        if not self.lcd_path:
            return []
        
        conn = self.connect(self.lcd_path)
        cursor = conn.cursor()
        
        # Get latest version of each LCD
        lcd_query = """
        SELECT 
            lcd_id, lcd_version, title, indication, coding_guidelines,
            doc_reqs, source_info, orig_det_eff_date, ent_det_end_date,
            status, keywords
        FROM LCD
        WHERE status = 'A'
        ORDER BY lcd_id, lcd_version DESC
        """
        
        lcds = []
        processed_ids = set()
        
        cursor.execute(lcd_query)
        for row in cursor.fetchall():
            lcd_id = str(row.lcd_id)
            
            # Only get latest version
            if lcd_id in processed_ids:
                continue
            processed_ids.add(lcd_id)
            
            lcd_version = row.lcd_version
            
            # Get HCPC codes
            hcpc_query = f"""
            SELECT DISTINCT hcpc_code_id
            FROM LCD_X_HCPC_CODE
            WHERE lcd_id = {lcd_id} AND lcd_version = {lcd_version}
            """
            cursor.execute(hcpc_query)
            hcpc_codes = [h[0] for h in cursor.fetchall() if h[0]]
            
            # Get ICD-10 covered codes
            icd10_cov_query = f"""
            SELECT DISTINCT icd10_code_id
            FROM LCD_X_ICD10_COVERED
            WHERE lcd_id = {lcd_id} AND lcd_version = {lcd_version}
            """
            cursor.execute(icd10_cov_query)
            icd10_covered = [i[0] for i in cursor.fetchall() if i[0]]
            
            # Get ICD-10 non-covered codes
            icd10_noncov_query = f"""
            SELECT DISTINCT icd10_code_id
            FROM LCD_X_ICD10_NONCOVERED
            WHERE lcd_id = {lcd_id} AND lcd_version = {lcd_version}
            """
            cursor.execute(icd10_noncov_query)
            icd10_noncovered = [i[0] for i in cursor.fetchall() if i[0]]
            
            # Determine coverage level
            coverage_level = "Full Coverage" if row.status == 'A' else "Unknown"
            if icd10_noncovered or (row.indication and "not covered" in row.indication.lower()):
                coverage_level = "Restricted Coverage"
            
            lcd = LCDDocument(
                lcd_id=lcd_id,
                lcd_version=lcd_version,
                title=row.title or "",
                indication=row.indication,
                coding_guidelines=row.coding_guidelines,
                doc_reqs=row.doc_reqs,
                source_info=row.source_info,
                effective_date=str(row.orig_det_eff_date) if row.orig_det_eff_date else None,
                end_date=str(row.ent_det_end_date) if row.ent_det_end_date else None,
                coverage_level=coverage_level,
                hcpc_codes=hcpc_codes,
                icd10_covered=icd10_covered,
                icd10_noncovered=icd10_noncovered,
                contractor_info=None,
                keywords=row.keywords
            )
            lcds.append(lcd)
        
        conn.close()
        return lcds
    
    def extract_articles(self) -> List[ArticleDocument]:
        """Extract all Article documents"""
        if not self.article_path:
            return []
        
        conn = self.connect(self.article_path)
        cursor = conn.cursor()
        
        # Get latest version of each article
        article_query = """
        SELECT 
            article_id, article_version, article_type, title, description,
            article_eff_date, article_end_date, status, keywords
        FROM ARTICLE
        WHERE status = 'A'
        ORDER BY article_id, article_version DESC
        """
        
        articles = []
        processed_ids = set()
        
        cursor.execute(article_query)
        for row in cursor.fetchall():
            article_id = str(row.article_id)
            
            # Only get latest version
            if article_id in processed_ids:
                continue
            processed_ids.add(article_id)
            
            article_version = row.article_version
            
            # Get article type description
            type_query = f"""
            SELECT description
            FROM ARTICLE_TYPE_LOOKUP
            WHERE article_type_id = {row.article_type}
            """
            cursor.execute(type_query)
            type_result = cursor.fetchone()
            article_type = type_result[0] if type_result else "Unknown"
            
            # Get HCPC codes
            hcpc_query = f"""
            SELECT DISTINCT hcpc_code_id
            FROM ARTICLE_X_HCPC_CODE
            WHERE article_id = {article_id} AND article_version = {article_version}
            """
            cursor.execute(hcpc_query)
            hcpc_codes = [h[0] for h in cursor.fetchall() if h[0]]
            
            # Get ICD-10 covered codes
            icd10_cov_query = f"""
            SELECT DISTINCT icd10_code_id
            FROM ARTICLE_X_ICD10_COVERED
            WHERE article_id = {article_id} AND article_version = {article_version}
            """
            cursor.execute(icd10_cov_query)
            icd10_covered = [i[0] for i in cursor.fetchall() if i[0]]
            
            # Get ICD-10 non-covered codes
            icd10_noncov_query = f"""
            SELECT DISTINCT icd10_code_id
            FROM ARTICLE_X_ICD10_NONCOVERED
            WHERE article_id = {article_id} AND article_version = {article_version}
            """
            cursor.execute(icd10_noncov_query)
            icd10_noncovered = [i[0] for i in cursor.fetchall() if i[0]]
            
            # Get revenue codes
            revenue_query = f"""
            SELECT DISTINCT revenue_code_id
            FROM ARTICLE_X_REVENUE_CODE
            WHERE article_id = {article_id} AND article_version = {article_version}
            """
            cursor.execute(revenue_query)
            revenue_codes = [r[0] for r in cursor.fetchall() if r[0]]
            
            # Get bill codes
            bill_query = f"""
            SELECT DISTINCT bill_code_id
            FROM ARTICLE_X_BILL_CODE
            WHERE article_id = {article_id} AND article_version = {article_version}
            """
            cursor.execute(bill_query)
            bill_codes = [b[0] for b in cursor.fetchall() if b[0]]
            
            article = ArticleDocument(
                article_id=article_id,
                article_version=article_version,
                article_type=article_type,
                title=row.title or "",
                description=row.description or "",
                effective_date=str(row.article_eff_date) if row.article_eff_date else None,
                end_date=str(row.article_end_date) if row.article_end_date else None,
                hcpc_codes=hcpc_codes,
                icd10_covered=icd10_covered,
                icd10_noncovered=icd10_noncovered,
                revenue_codes=revenue_codes,
                bill_codes=bill_codes,
                contractor_info=None,
                keywords=row.keywords
            )
            articles.append(article)
        
        conn.close()
        return articles
    
    def export_to_json(self, documents: List, output_path: str):
        """Export documents to JSON format"""
        data = [asdict(doc) for doc in documents]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(documents)} documents to {output_path}")


class TextChunker:
    """Split documents into manageable chunks"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_ncd(self, ncd: NCDDocument) -> List[Dict[str, Any]]:
        """Split NCD into semantic chunks"""
        chunks = []
        
        base_metadata = {
            'doc_type': 'NCD',
            'ncd_id': ncd.ncd_id,
            'version': ncd.version_num,
            'section': ncd.ncd_section,
            'title': ncd.section_title,
            'coverage_level': ncd.coverage_level,
            'effective_date': ncd.effective_date,
            'benefit_categories': ','.join(ncd.benefit_categories)
        }
        
        # Overview chunk
        overview = f"""[NCD] {ncd.ncd_section}: {ncd.section_title}
Coverage Level: {ncd.coverage_level}
Effective Date: {ncd.effective_date}
Benefit Categories: {', '.join(ncd.benefit_categories)}"""
        
        chunks.append({
            'text': overview,
            'metadata': {**base_metadata, 'chunk_type': 'overview'}
        })
        
        # Description chunk
        if ncd.item_service_desc:
            chunks.append({
                'text': f"[NCD Coverage Description]\n{ncd.item_service_desc}",
                'metadata': {**base_metadata, 'chunk_type': 'description'}
            })
        
        # Indications and Limitations (most important)
        if ncd.indication_limitation:
            indication_chunks = self._split_text(ncd.indication_limitation)
            for i, chunk_text in enumerate(indication_chunks):
                chunks.append({
                    'text': f"[NCD Indications/Limitations]\n{chunk_text}",
                    'metadata': {**base_metadata, 'chunk_type': 'indication_limitation', 'part': i+1}
                })
        
        # Cross References
        if ncd.cross_reference:
            chunks.append({
                'text': f"[NCD Cross References]\n{ncd.cross_reference}",
                'metadata': {**base_metadata, 'chunk_type': 'cross_reference'}
            })
        
        return chunks
    
    def chunk_lcd(self, lcd: LCDDocument) -> List[Dict[str, Any]]:
        """Split LCD into semantic chunks"""
        chunks = []
        
        base_metadata = {
            'doc_type': 'LCD',
            'lcd_id': lcd.lcd_id,
            'version': lcd.lcd_version,
            'title': lcd.title,
            'coverage_level': lcd.coverage_level,
            'effective_date': lcd.effective_date or '',
            'hcpc_codes': ','.join(lcd.hcpc_codes[:50])  # Limit metadata size
        }
        
        # Overview with codes
        overview = f"""[LCD] L{lcd.lcd_id}: {lcd.title}
Coverage Level: {lcd.coverage_level}
Effective Date: {lcd.effective_date or 'N/A'}
CPT/HCPCS Codes: {', '.join(lcd.hcpc_codes[:20])}"""
        if len(lcd.hcpc_codes) > 20:
            overview += f"\n... and {len(lcd.hcpc_codes) - 20} more codes"
        
        chunks.append({
            'text': overview,
            'metadata': {**base_metadata, 'chunk_type': 'overview'}
        })
        
        # Coverage Indications
        if lcd.indication:
            indication_chunks = self._split_text(lcd.indication)
            for i, chunk_text in enumerate(indication_chunks):
                chunks.append({
                    'text': f"[LCD Coverage Indications]\n{chunk_text}",
                    'metadata': {**base_metadata, 'chunk_type': 'indications', 'part': i+1}
                })
        
        # Coding Guidelines
        if lcd.coding_guidelines:
            coding_chunks = self._split_text(lcd.coding_guidelines)
            for i, chunk_text in enumerate(coding_chunks):
                chunks.append({
                    'text': f"[LCD Coding Guidelines]\n{chunk_text}",
                    'metadata': {**base_metadata, 'chunk_type': 'coding_guidelines', 'part': i+1}
                })
        
        # Documentation Requirements
        if lcd.doc_reqs:
            doc_chunks = self._split_text(lcd.doc_reqs)
            for i, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    'text': f"[LCD Documentation Requirements]\n{chunk_text}",
                    'metadata': {**base_metadata, 'chunk_type': 'documentation', 'part': i+1}
                })
        
        # ICD-10 Coverage Info
        if lcd.icd10_covered or lcd.icd10_noncovered:
            icd_text = ""
            if lcd.icd10_covered:
                icd_text += f"Covered ICD-10 Codes: {', '.join(lcd.icd10_covered[:30])}\n"
            if lcd.icd10_noncovered:
                icd_text += f"Non-Covered ICD-10 Codes: {', '.join(lcd.icd10_noncovered[:30])}"
            
            chunks.append({
                'text': f"[LCD ICD-10 Coverage]\n{icd_text}",
                'metadata': {**base_metadata, 'chunk_type': 'icd10_codes'}
            })
        
        return chunks
    
    def chunk_article(self, article: ArticleDocument) -> List[Dict[str, Any]]:
        """Split Article into semantic chunks"""
        chunks = []
        
        base_metadata = {
            'doc_type': 'Article',
            'article_id': article.article_id,
            'version': article.article_version,
            'article_type': article.article_type,
            'title': article.title,
            'effective_date': article.effective_date or '',
            'hcpc_codes': ','.join(article.hcpc_codes[:50])
        }
        
        # Overview
        overview = f"""[Article] A{article.article_id}: {article.title}
Article Type: {article.article_type}
Effective Date: {article.effective_date or 'N/A'}
CPT/HCPCS Codes: {', '.join(article.hcpc_codes[:20])}"""
        
        chunks.append({
            'text': overview,
            'metadata': {**base_metadata, 'chunk_type': 'overview'}
        })
        
        # Description
        if article.description:
            desc_chunks = self._split_text(article.description)
            for i, chunk_text in enumerate(desc_chunks):
                chunks.append({
                    'text': f"[Article Description]\n{chunk_text}",
                    'metadata': {**base_metadata, 'chunk_type': 'description', 'part': i+1}
                })
        
        # ICD-10 and Revenue Code Info
        if article.icd10_covered or article.icd10_noncovered or article.revenue_codes:
            code_text = ""
            if article.icd10_covered:
                code_text += f"Covered ICD-10 Codes: {', '.join(article.icd10_covered[:30])}\n"
            if article.icd10_noncovered:
                code_text += f"Non-Covered ICD-10 Codes: {', '.join(article.icd10_noncovered[:30])}\n"
            if article.revenue_codes:
                code_text += f"Revenue Codes: {', '.join(article.revenue_codes[:20])}"
            
            chunks.append({
                'text': f"[Article Coding Information]\n{code_text}",
                'metadata': {**base_metadata, 'chunk_type': 'coding_info'}
            })
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split long text into chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_words = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class OllamaEmbeddingService:
    """Handle Ollama embeddings and inference"""
    
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.client = ollama.Client()
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama"""
        try:
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {e}"


class ChromaDBVectorStore:
    """Manage ChromaDB vector storage"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "medicare_documents"
        self.collection = None
    
    def create_collection(self):
        """Create or get collection"""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add document chunks with embeddings to ChromaDB"""
        if not self.collection:
            self.create_collection()
        
        ids = [f"chunk_{i}_{chunks[i]['metadata'].get('doc_type', 'doc')}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Added {len(chunks)} chunks to ChromaDB")
    
    def search(self, query_embedding: List[float], n_results: int = 7) -> Dict:
        """Search for similar documents"""
        if not self.collection:
            self.create_collection()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results


class CoverageDeterminationEngine:
    """Determine Medicare coverage for claims"""
    
    def __init__(self, vector_store: ChromaDBVectorStore, ollama_service: OllamaEmbeddingService):
        self.vector_store = vector_store
        self.ollama = ollama_service
    
    def process_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single claim for coverage determination"""
        
        # Create query from claim details
        query = self._create_claim_query(claim)
        
        # Generate embedding for query
        query_embedding = self.ollama.generate_embedding(query)
        
        # Retrieve relevant documentation chunks
        search_results = self.vector_store.search(query_embedding, n_results=7)
        
        # Format retrieved documents
        context_docs = self._format_context(search_results)
        
        # Generate coverage determination
        prompt = self._create_determination_prompt(claim, context_docs)
        determination = self.ollama.generate_response(prompt)
        
        return {
            'claim': claim,
            'query': query,
            'retrieved_documents': context_docs,
            'determination': determination,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_claim_query(self, claim: Dict[str, Any]) -> str:
        """Create search query from claim details"""
        cpt = claim.get('cpt_hcpcs_code', '')
        icd10_codes = claim.get('icd10_codes', '').split(',')
        pos = claim.get('place_of_service', '')
        provider = claim.get('provider_type', '')
        
        query = f"""CPT/HCPCS Code: {cpt}
ICD-10 Diagnosis Codes: {', '.join(icd10_codes)}
Place of Service: {pos}
Provider Type: {provider}
Medicare coverage requirements, medical necessity criteria, coding guidelines, and documentation requirements"""
        
        return query
    
    def _format_context(self, search_results: Dict) -> List[Dict[str, Any]]:
        """Format retrieved documents for prompt"""
        docs = []
        if search_results and 'documents' in search_results:
            for i, doc in enumerate(search_results['documents'][0]):
                metadata = search_results['metadatas'][0][i] if 'metadatas' in search_results else {}
                docs.append({
                    'text': doc,
                    'metadata': metadata,
                    'distance': search_results['distances'][0][i] if 'distances' in search_results else None
                })
        return docs
    
    def _create_determination_prompt(self, claim: Dict[str, Any], context_docs: List[Dict[str, Any]]) -> str:
        """Create prompt for Ollama to determine coverage"""
        
        # Format claim details
        claim_details = f"""
Claim Line Details:
- CPT/HCPCS Code: {claim.get('cpt_hcpcs_code', 'N/A')}
- ICD-10 Diagnosis Code(s): {claim.get('icd10_codes', 'N/A')}
- Place of Service: {claim.get('place_of_service', 'N/A')}
- Provider Type: {claim.get('provider_type', 'N/A')}
- Service Date: {claim.get('service_date', 'N/A')}
- Patient Age: {claim.get('patient_age', 'N/A')}
- Patient Gender: {claim.get('patient_gender', 'N/A')}
- Modifier Codes: {claim.get('modifier_codes', 'N/A')}
- Units: {claim.get('units', 'N/A')}
"""
        
        # Format retrieved CMS documentation by type
        ncd_docs = []
        lcd_docs = []
        article_docs = []
        
        for i, doc in enumerate(context_docs):
            doc_type = doc['metadata'].get('doc_type', 'Unknown')
            doc_id = doc['metadata'].get('ncd_id') or doc['metadata'].get('lcd_id') or doc['metadata'].get('article_id', 'N/A')
            
            if doc_type == 'NCD':
                ncd_docs.append(f"NCD {doc['metadata'].get('section', doc_id)}:\n{doc['text']}")
            elif doc_type == 'LCD':
                lcd_docs.append(f"LCD L{doc_id}:\n{doc['text']}")
            elif doc_type == 'Article':
                article_docs.append(f"Article A{doc_id}:\n{doc['text']}")
        
        cms_docs = ""
        if ncd_docs:
            cms_docs += "=== National Coverage Determinations (NCDs) ===\n\n"
            cms_docs += "\n\n---\n\n".join(ncd_docs)
        
        if lcd_docs:
            cms_docs += "\n\n=== Local Coverage Determinations (LCDs) ===\n\n"
            cms_docs += "\n\n---\n\n".join(lcd_docs)
        
        if article_docs:
            cms_docs += "\n\n=== Billing & Coding Articles ===\n\n"
            cms_docs += "\n\n---\n\n".join(article_docs)
        
        if not cms_docs:
            cms_docs = "No directly relevant CMS documentation found in the database."
        
        prompt = f"""You are a Medicare coverage determination specialist. Your task is to determine if the following service is covered under Medicare guidelines based on the provided CMS documentation (NCDs, LCDs, and Articles).

{claim_details}

Relevant CMS Documentation:
{cms_docs}

Instructions:
1. Analyze the claim details against the retrieved CMS documentation
2. Consider the hierarchy: NCDs take precedence over LCDs, which take precedence over Articles
3. Determine if the service is:
   - COVERED: Service meets all coverage criteria
   - RESTRICTED: Service may be covered under specific conditions
   - NOT COVERED: Service does not meet coverage criteria
   - INSUFFICIENT INFORMATION: Need additional documentation to determine

4. Provide your determination with:
   - Clear coverage decision (COVERED/RESTRICTED/NOT COVERED/INSUFFICIENT INFORMATION)
   - Specific reasoning based on the documentation requirements
   - Citations to applicable NCD, LCD, or Article sections
   - Any conditions or limitations that apply
   - Required documentation if applicable
   - Coding guidelines that must be followed
   - Any modifier requirements

Coverage Determination:
"""
        
        return prompt


def main():
    """Main execution flow"""
    
    print("=" * 70)
    print("Enhanced Medicare Coverage Determination System")
    print("NCD + LCD + Article Processing")
    print("=" * 70)
    
    # Configuration
    NCD_PATH = r"C:\Users\kanis\OneDrive\Documents\PythonScripts\medicare\ncd.mdb"
    LCD_PATH = r"C:\Users\kanis\OneDrive\Documents\PythonScripts\medicare\current_lcd.mdb"  # Update path
    ARTICLE_PATH = r"C:\Users\kanis\OneDrive\Documents\PythonScripts\medicare\current_article.mdb"  # Update path
    CLAIMS_CSV = "sample_claims.csv"
    CHROMA_DIR = "./chroma_db"
    OLLAMA_MODEL = "llama3"
    
    # Step 1: Extract data from databases
    print("\n[1/7] Extracting data from Medicare databases...")
    extractor = DatabaseExtractor(NCD_PATH, LCD_PATH, ARTICLE_PATH)
    
    all_documents = []
    
    # Extract NCDs
    try:
        print("  Extracting NCDs...")
        ncds = extractor.extract_ncds()
        all_documents.extend(ncds)
        extractor.export_to_json(ncds, "ncd_documents.json")
        print(f"  ✓ Extracted {len(ncds)} NCD documents")
    except Exception as e:
        print(f"  ✗ Error extracting NCDs: {e}")
        ncds = []
    
    # Extract LCDs
    try:
        print("  Extracting LCDs...")
        lcds = extractor.extract_lcds()
        all_documents.extend(lcds)
        extractor.export_to_json(lcds, "lcd_documents.json")
        print(f"  ✓ Extracted {len(lcds)} LCD documents")
    except Exception as e:
        print(f"  ✗ Error extracting LCDs: {e}")
        lcds = []
    
    # Extract Articles
    try:
        print("  Extracting Articles...")
        articles = extractor.extract_articles()
        all_documents.extend(articles)
        extractor.export_to_json(articles, "article_documents.json")
        print(f"  ✓ Extracted {len(articles)} Article documents")
    except Exception as e:
        print(f"  ✗ Error extracting Articles: {e}")
        articles = []
    
    print(f"\nTotal documents extracted: {len(all_documents)}")
    
    # Step 2: Chunk documents
    print("\n[2/7] Chunking documents...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    all_chunks = []
    
    for ncd in ncds:
        chunks = chunker.chunk_ncd(ncd)
        all_chunks.extend(chunks)
    
    for lcd in lcds:
        chunks = chunker.chunk_lcd(lcd)
        all_chunks.extend(chunks)
    
    for article in articles:
        chunks = chunker.chunk_article(article)
        all_chunks.extend(chunks)
    
    print(f"✓ Created {len(all_chunks)} chunks from {len(all_documents)} documents")
    print(f"  - NCD chunks: {sum(1 for c in all_chunks if c['metadata'].get('doc_type') == 'NCD')}")
    print(f"  - LCD chunks: {sum(1 for c in all_chunks if c['metadata'].get('doc_type') == 'LCD')}")
    print(f"  - Article chunks: {sum(1 for c in all_chunks if c['metadata'].get('doc_type') == 'Article')}")
    
    # Step 3: Initialize Ollama
    print("\n[3/7] Initializing Ollama service...")
    try:
        ollama_service = OllamaEmbeddingService(model=OLLAMA_MODEL)
        print(f"✓ Ollama service initialized with model: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"✗ Error initializing Ollama: {e}")
        print("Ensure Ollama is running: ollama serve")
        return
    
    # Step 4: Generate embeddings and store in ChromaDB
    print("\n[4/7] Generating embeddings and storing in ChromaDB...")
    vector_store = ChromaDBVectorStore(persist_directory=CHROMA_DIR)
    vector_store.create_collection()
    
    if all_chunks:
        print("Generating embeddings (this may take a while)...")
        embeddings = []
        batch_size = 10
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            print(f"  Processing chunks {i+1}-{min(i+batch_size, len(all_chunks))}/{len(all_chunks)}...")
            
            for chunk in batch:
                embedding = ollama_service.generate_embedding(chunk['text'])
                embeddings.append(embedding)
        
        vector_store.add_documents(all_chunks, embeddings)
        print(f"✓ Stored {len(all_chunks)} chunks in ChromaDB")
    
    # Step 5: Load claims data
    print("\n[5/7] Loading claims data...")
    try:
        claims_df = pd.read_csv(CLAIMS_CSV)
        claims = claims_df.to_dict('records')
        print(f"✓ Loaded {len(claims)} claims")
    except Exception as e:
        print(f"✗ Error loading claims: {e}")
        return
    
    # Step 6: Process claims for coverage determination
    print("\n[6/7] Processing claims for coverage determination...")
    engine = CoverageDeterminationEngine(vector_store, ollama_service)
    
    results = []
    for i, claim in enumerate(claims):
        print(f"\n--- Processing Claim {i+1}/{len(claims)} ---")
        print(f"CPT: {claim.get('cpt_hcpcs_code')}, ICD-10: {claim.get('icd10_codes')}")
        print(f"Provider: {claim.get('provider_type')}, POS: {claim.get('place_of_service')}")
        
        result = engine.process_claim(claim)
        results.append(result)
        
        print(f"\nRetrieved {len(result['retrieved_documents'])} relevant documentation chunks:")
        doc_types = {}
        for doc in result['retrieved_documents']:
            doc_type = doc['metadata'].get('doc_type', 'Unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in doc_types.items():
            print(f"  - {doc_type}: {count} chunks")
        
        print(f"\nCoverage Determination:")
        print("-" * 70)
        determination_preview = result['determination'][:800]
        print(determination_preview + "..." if len(result['determination']) > 800 else determination_preview)
        print("-" * 70)
    
    # Step 7: Save results
    print("\n[7/7] Saving results...")
    output_file = "coverage_determinations.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Also create a readable summary report
    summary_file = "coverage_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Medicare Coverage Determination Summary Report\n")
        f.write("=" * 70 + "\n\n")
        
        for i, result in enumerate(results):
            claim = result['claim']
            f.write(f"CLAIM {i+1}\n")
            f.write("-" * 70 + "\n")
            f.write(f"CPT/HCPCS: {claim.get('cpt_hcpcs_code')}\n")
            f.write(f"ICD-10: {claim.get('icd10_codes')}\n")
            f.write(f"Provider: {claim.get('provider_type')}\n")
            f.write(f"Place of Service: {claim.get('place_of_service')}\n")
            f.write(f"Service Date: {claim.get('service_date')}\n\n")
            
            f.write("DETERMINATION:\n")
            f.write(result['determination'])
            f.write("\n\n" + "=" * 70 + "\n\n")
    
    print(f"\n{'=' * 70}")
    print(f"✓ All claims processed!")
    print(f"✓ Detailed results saved to: {output_file}")
    print(f"✓ Summary report saved to: {summary_file}")
    print(f"\nProcessed {len(results)} claims using:")
    print(f"  - {len(ncds)} NCDs")
    print(f"  - {len(lcds)} LCDs")
    print(f"  - {len(articles)} Articles")
    print(f"  - {len(all_chunks)} total documentation chunks")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()