#!/usr/bin/env python3
"""
Test chunking strategies using sample_download_try.bib

This script:
1. Loads papers from the BibTeX file
2. Downloads PDFs for available papers
3. Parses and tests different chunking strategies
4. Reports statistics on chunk quality
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Direct imports to avoid heavy dependency chain
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import only what we need for chunking tests
import bibtexparser
from perspicacite.pipeline.download import get_pdf_with_fallback
from perspicacite.pipeline.parsers.pdf import PDFParser
from perspicacite.pipeline.chunking_advanced import (
    AdvancedChunker,
    split_into_sections,
    chunk_by_tokens,
    chunk_by_semantics,
    get_tokenizer,
)
from perspicacite.models.papers import Paper, Author, PaperSource


def parse_bibtex_manual(bib_path: Path) -> list:
    """Parse BibTeX file manually to avoid heavy imports."""
    with open(bib_path) as f:
        text = f.read()
    
    db = bibtexparser.loads(text)
    papers = []
    
    for entry in db.entries:
        if entry.get('ENTRYTYPE', '').lower() not in ('article', 'inproceedings'):
            continue
        
        # Parse authors
        author_str = entry.get('author', '')
        authors = []
        if author_str:
            for name in author_str.split(' and '):
                authors.append(Author(name=name.strip()))
        
        # Parse year
        year = None
        try:
            year = int(entry.get('year', 0))
        except (ValueError, TypeError):
            pass
        
        paper = Paper(
            id=entry.get('ID', ''),
            title=entry.get('title', ''),
            authors=authors,
            year=year,
            doi=entry.get('doi', ''),
            source=PaperSource.BIBTEX,
        )
        papers.append(paper)
    
    return papers


async def download_and_parse_papers(bib_path: Path, max_papers: int = 3):
    """Download and parse papers from BibTeX."""
    print(f"\n{'='*60}")
    print(f"Loading BibTeX: {bib_path}")
    print(f"{'='*60}")
    
    papers = parse_bibtex_manual(bib_path)
    
    print(f"Found {len(papers)} papers")
    
    parser = PDFParser()
    downloaded = []
    
    for i, paper in enumerate(papers[:max_papers], 1):
        print(f"\n[{i}/{min(max_papers, len(papers))}] {paper.title[:60]}...")
        print(f"    DOI: {paper.doi}")
        
        if not paper.doi:
            print("    ⚠️ No DOI, skipping")
            continue
        
        try:
            pdf_bytes = await get_pdf_with_fallback(
                paper.doi,
                unpaywall_email="test@example.com",
            )
            
            if not pdf_bytes:
                print("    ❌ PDF not available")
                continue
            
            print(f"    ✓ Downloaded {len(pdf_bytes)} bytes")
            
            # Parse PDF
            parsed = await parser.parse_bytes(pdf_bytes)
            if parsed and parsed.text:
                print(f"    ✓ Extracted {len(parsed.text)} characters")
                paper.full_text = parsed.text
                downloaded.append(paper)
            else:
                print("    ⚠️ No text extracted")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    return downloaded


def test_section_detection(paper: Paper):
    """Test section detection on a paper."""
    print(f"\n{'='*60}")
    print(f"SECTION DETECTION TEST")
    print(f"{'='*60}")
    print(f"Paper: {paper.title[:60]}...")
    
    sections = split_into_sections(paper.full_text)
    print(f"\nDetected {len(sections)} sections:")
    
    for i, (name, text) in enumerate(sections, 1):
        print(f"  {i}. {name[:50]:<50} ({len(text)} chars)")


async def test_token_chunking(paper: Paper):
    """Test token-based chunking."""
    print(f"\n{'='*60}")
    print(f"TOKEN-BASED CHUNKING TEST")
    print(f"{'='*60}")
    
    chunker = AdvancedChunker(
        method="token",
        max_tokens=500,
        overlap_tokens=50,
        provider="openai",  # Will use tiktoken
    )
    
    chunks = await chunker.chunk_text(paper.full_text, paper)
    
    print(f"Paper: {paper.title[:60]}...")
    print(f"Input: {len(paper.full_text)} characters")
    print(f"Output: {len(chunks)} chunks")
    print(f"\nChunk sizes (chars):")
    
    sizes = [len(c.text) for c in chunks]
    print(f"  Min: {min(sizes)}")
    print(f"  Max: {max(sizes)}")
    print(f"  Avg: {sum(sizes)//len(sizes)}")
    
    print(f"\nFirst chunk preview:")
    print(f"  {chunks[0].text[:200]}...")


async def test_semantic_chunking(paper: Paper):
    """Test semantic chunking (if sentence-transformers available)."""
    print(f"\n{'='*60}")
    print(f"SEMANTIC CHUNKING TEST")
    print(f"{'='*60}")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers available")
    except ImportError:
        print("⚠️ sentence-transformers not installed, skipping")
        print("  Install with: pip install sentence-transformers")
        return
    
    chunker = AdvancedChunker(
        method="semantic",
        max_tokens=800,
        overlap_tokens=120,
        semantic_threshold=0.65,
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    
    chunks = await chunker.chunk_text(paper.full_text, paper)
    
    print(f"Paper: {paper.title[:60]}...")
    print(f"Input: {len(paper.full_text)} characters")
    print(f"Output: {len(chunks)} chunks")
    print(f"\nChunk sizes (chars):")
    
    sizes = [len(c.text) for c in chunks]
    print(f"  Min: {min(sizes)}")
    print(f"  Max: {max(sizes)}")
    print(f"  Avg: {sum(sizes)//len(sizes)}")
    
    print(f"\nFirst chunk preview:")
    print(f"  {chunks[0].text[:200]}...")


async def test_section_aware_chunking(paper: Paper):
    """Test section-aware chunking."""
    print(f"\n{'='*60}")
    print(f"SECTION-AWARE CHUNKING TEST")
    print(f"{'='*60}")
    
    chunker = AdvancedChunker(
        method="token",
        max_tokens=500,
        overlap_tokens=50,
        section_aware=True,
    )
    
    chunks = await chunker.chunk_text(paper.full_text, paper)
    
    print(f"Paper: {paper.title[:60]}...")
    print(f"Input: {len(paper.full_text)} characters")
    print(f"Output: {len(chunks)} chunks")
    
    # Group by section
    sections = {}
    for c in chunks:
        sect = c.metadata.section or "Unknown"
        sections.setdefault(sect, []).append(c)
    
    print(f"\nChunks by section:")
    for sect, sect_chunks in sections.items():
        print(f"  {sect[:40]:<40} {len(sect_chunks)} chunks")


async def main():
    """Main test function."""
    bib_path = Path(__file__).parent / "tests" / "sample_download_try.bib"
    
    if not bib_path.exists():
        print(f"Error: {bib_path} not found")
        sys.exit(1)
    
    # Download papers
    papers = await download_and_parse_papers(bib_path, max_papers=3)
    
    if not papers:
        print("\n❌ No papers downloaded successfully")
        print("Check your internet connection and try again")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Successfully downloaded {len(papers)} papers")
    print(f"{'='*60}")
    
    # Use first paper for chunking tests
    test_paper = papers[0]
    
    # Run tests
    test_section_detection(test_paper)
    await test_token_chunking(test_paper)
    await test_semantic_chunking(test_paper)
    await test_section_aware_chunking(test_paper)
    
    print(f"\n{'='*60}")
    print(f"CHUNKING TESTS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
