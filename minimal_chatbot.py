#!/usr/bin/env python3
"""
Minimal Perspicacite Chatbot

Usage:
    python minimal_chatbot.py "Your research question"
"""

import argparse
import asyncio
import csv
import os
import subprocess
import sys
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Import perspicacite
from perspicacite.models.papers import Paper, Author
from perspicacite.pipeline.download import PDFDownloader


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="Research question")
    args = parser.parse_args()
    
    query = args.query or input("Question: ").strip()
    if not query:
        print("❌ No question")
        return
    
    print("="*70)
    print("🤖 MINIMAL PERSPICACITE CHATBOT")
    print("="*70)
    print(f"\n📝 Question: {query}")
    
    # Step 1: Search with SciLEx (or use existing results)
    print(f"\n🔍 Step 1: Searching SciLEx...")
    
    scilex_dir = BASE_DIR / "packages_to_use/SciLEx"
    output_dir = scilex_dir / "output" / "minimal_chatbot"
    csv_file = output_dir / "aggregated_results.csv"
    
    # Check if we have existing results
    if not csv_file.exists() or os.path.getsize(csv_file) < 100:
        # Run new search
        config_file = scilex_dir / "scilex" / "scilex.config.yml"
        config = f"""keywords:
  - ["{query}"]
years:
  - 2024
apis:
  - OpenAlex
collect_name: minimal_chatbot
quality_filters:
  max_papers: 5
"""
        with open(config_file, 'w') as f:
            f.write(config)
        
        os.chdir(scilex_dir)
        subprocess.run(["scilex-collect"], capture_output=True)
        subprocess.run(["scilex-aggregate"], capture_output=True)
        os.chdir(BASE_DIR)
    
    # Try to find any recent results if current is empty
    if not csv_file.exists() or os.path.getsize(csv_file) < 100:
        # Look for recent collections
        collections = list((scilex_dir / "output").glob("agentic_*"))
        if collections:
            latest = max(collections, key=lambda x: x.stat().st_mtime)
            csv_file = latest / "aggregated_results.csv"
            print(f"   Using existing collection: {latest.name}")
    
    papers = []
    if csv_file.exists():
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                authors = [Author(name=n.strip()) for n in row.get('authors', '').split(';') if n.strip()][:3]
                papers.append(Paper(
                    id=row.get('archiveID', '')[-12:] or 'paper_' + str(len(papers)),
                    title=row.get('title', ''),
                    authors=authors,
                    year=int(row.get('date', '')[:4]) if row.get('date') else None,
                    doi=row.get('DOI') or None,
                    abstract=row.get('abstract', ''),
                    pdf_url=row.get('pdf_url') if row.get('pdf_url') != 'NA' else None,
                ))
    
    print(f"   Found {len(papers)} papers")
    
    if not papers:
        print("❌ No papers found")
        return
    
    # Step 2: Download with Perspicacite PDFDownloader
    print(f"\n📥 Step 2: Downloading with Perspicacite PDFDownloader...")
    
    downloader = PDFDownloader()
    
    for paper in papers:
        if paper.pdf_url:
            try:
                # Use asyncio.wait_for to add timeout
                pdf_bytes = await asyncio.wait_for(
                    downloader.download(paper.pdf_url),
                    timeout=10.0
                )
                if pdf_bytes:
                    paper.full_text = f"[PDF: {len(pdf_bytes)} bytes]\n{paper.abstract}"
                    print(f"   ✅ {paper.title[:50]}... ({len(pdf_bytes)} bytes)")
                else:
                    paper.full_text = paper.abstract
                    print(f"   📄 {paper.title[:50]}... (abstract)")
            except asyncio.TimeoutError:
                paper.full_text = paper.abstract
                print(f"   ⏱️  {paper.title[:50]}... (timeout)")
            except Exception as e:
                paper.full_text = paper.abstract
                print(f"   📄 {paper.title[:50]}... (error)")
        else:
            paper.full_text = paper.abstract
            print(f"   📄 {paper.title[:50]}... (no PDF)")
        
        await asyncio.sleep(0.1)
    
    # Step 3: Answer (simplified - just show we have the data)
    print(f"\n🧠 Step 3: Papers ready for analysis")
    print(f"\n📚 Papers in Perspicacite Paper objects:")
    for i, p in enumerate(papers[:3], 1):
        print(f"\n{i}. {p.title}")
        print(f"   Authors: {', '.join(a.name for a in p.authors)}")
        print(f"   Year: {p.year}")
        print(f"   Has full text: {'Yes' if p.full_text else 'No'}")
    
    print("\n" + "="*70)
    print("✅ Perspicacite workflow complete!")
    print("="*70)
    print(f"\nNext: These Paper objects can be:")
    print("  • Added to DynamicKnowledgeBase")
    print("  • Processed by AgenticRAGMode")
    print("  • Assessed by PaperAssessor")
    print("  • Used to answer the question")


if __name__ == "__main__":
    asyncio.run(main())
