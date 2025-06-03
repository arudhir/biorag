import json
import re
from pathlib import Path
from typing import Dict, Generator

import fitz  # PyMuPDF
import tiktoken
from tqdm import tqdm


class PDFProcessor:
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing excessive whitespace and fixing encoding."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        return text.strip()
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))
    
    def _create_chunks(self, text: str, metadata: Dict) -> Generator[Dict, None, None]:
        """Create overlapping chunks from text with metadata."""
        tokens = self.tokenizer.encode(text)
        start_idx = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Calculate character positions
            char_start = text.find(chunk_text)
            char_end = char_start + len(chunk_text)
            
            yield {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": f"{metadata['source']}_p{metadata['page']}_chunk{start_idx // (self.chunk_size - self.overlap)}",
                    "char_start": char_start,
                    "char_end": char_end
                }
            }
            
            if end_idx == len(tokens):
                break
                
            start_idx = end_idx - self.overlap
    
    def process_pdf(self, pdf_path: Path) -> Generator[Dict, None, None]:
        """Process a single PDF file and yield chunks with metadata."""
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                text = self._clean_text(text)
                
                if not text or self._count_tokens(text) < 50:  # Skip very short pages
                    continue
                
                metadata = {
                    "source": pdf_path.name,
                    "page": page_num + 1
                }
                
                yield from self._create_chunks(text, metadata)
                
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return
    
    def process_directory(self, input_dir: Path, output_file: Path) -> None:
        """Process all PDFs in a directory and save chunks to JSONL file."""
        pdf_files = list(input_dir.glob("*.pdf"))
        
        with open(output_file, 'w') as f:
            for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
                for chunk in self.process_pdf(pdf_path):
                    f.write(json.dumps(chunk) + '\n') 