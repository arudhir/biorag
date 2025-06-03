#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_processor import PDFProcessor

def main():
    parser = argparse.ArgumentParser(description="Extract text chunks from PDF files")
    parser.add_argument("--input", required=True, help="Input directory containing PDF files")
    parser.add_argument("--output", required=True, help="Output JSONL file for chunks")
    parser.add_argument("--chunk-size", type=int, default=800, help="Size of text chunks in tokens")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks in tokens")
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{args.input}' does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = PDFProcessor(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    # Process PDFs
    print(f"Processing PDFs in {input_dir}...")
    processor.process_directory(input_dir, output_file)
    print(f"Chunks saved to {output_file}")

if __name__ == "__main__":
    main() 