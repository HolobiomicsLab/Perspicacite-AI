# SciLEx + OpenAlex Guide

OpenAlex is a **completely free** academic API with generous rate limits (100k requests/day without key, more with key).

## Quick Start

### Step 1: Configure SciLEx for OpenAlex

```bash
cd /mnt/d/new_repos/perspicacite_v2/packages_to_use/SciLEx
```

Edit `scilex/scilex.config.yml`:

```yaml
# Search terms
keywords:
  - ["vision transformers", "medical imaging"]

# Years to search
years:
  - 2024
  - 2023

# APIs to use (OpenAlex is FREE - no key required!)
apis:
  - OpenAlex

# Project name
collect_name: my_research

# Quality filters
quality_filters:
  enable_itemtype_filter: false
  require_abstract: true
  require_doi: false
  apply_citation_filter: false
  max_papers: 20

# Disable HuggingFace enrichment (optional)
hf_enrichment:
  enabled: false
```

### Step 2: Run Collection

```bash
# Activate virtual environment
source ../../.venv/bin/activate

# Collect papers
scilex-collect
```

### Step 3: Aggregate Results

```bash
# Deduplicate and filter
scilex-aggregate
```

### Step 4: Export

```bash
# Export to BibTeX
scilex-export-bibtex

# Or check the CSV output
cat output/*/aggregated_results.csv
```

---

## Complete Example

### 1. Search for Papers

```bash
cd /mnt/d/new_repos/perspicacite_v2/packages_to_use/SciLEx

# Create config for your topic
cat > scilex/scilex.config.yml << 'EOF'
keywords:
  - ["CRISPR", "gene editing", "cancer therapy"]

years:
  - 2024
  - 2023
  - 2022

apis:
  - OpenAlex

collect_name: crispr_cancer_research

quality_filters:
  enable_itemtype_filter: false
  require_abstract: true
  require_doi: false
  apply_citation_filter: true
  max_papers: 50

hf_enrichment:
  enabled: false
EOF

# Run collection
scilex-collect
```

**Expected output:**
```
12:30:00 - INFO - SciLEx Systematic Review Collection
12:30:00 - INFO - Configuration: 3 keywords, 3 years, 1 APIs
...
OpenAlex            : 100%|██████████| 9/9 [00:15<00:00, 1.5s/query, papers=47]
============================================================
Collection Complete - Summary:
============================================================
OpenAlex            :   9 queries | 47 papers
```

### 2. Process Results

```bash
# Aggregate (deduplicate, filter, rank)
scilex-aggregate

# Output: output/crispr_cancer_research/aggregated_results.csv
```

### 3. View Results

```bash
# Check the CSV
cat output/crispr_cancer_research/aggregated_results.csv | head -20

# Or export to BibTeX
scilex-export-bibtex --output my_papers.bib
```

---

## Python Script for Automated Search

Use this script to search OpenAlex programmatically:

```python
#!/usr/bin/env python3
"""Search OpenAlex using SciLEx."""

import subprocess
import tempfile
import os
from pathlib import Path


def search_openalex(query: str, max_papers: int = 20):
    """Search OpenAlex for papers."""
    
    # Create config
    config = f"""
keywords:
  - ["{query}"]

years:
  - 2024
  - 2023

apis:
  - OpenAlex

collect_name: search_{query.replace(' ', '_')[:20]}

quality_filters:
  enable_itemtype_filter: false
  require_abstract: true
  require_doi: false
  apply_citation_filter: false
  max_papers: {max_papers}

hf_enrichment:
  enabled: false
"""
    
    # Save config
    scilex_dir = Path("packages_to_use/SciLEx")
    config_file = scilex_dir / "scilex" / "scilex.config.yml"
    
    with open(config_file, 'w') as f:
        f.write(config)
    
    # Run collection
    os.chdir(scilex_dir)
    result = subprocess.run(
        ["scilex-collect"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Collection successful!")
        # Run aggregate
        subprocess.run(["scilex-aggregate"], capture_output=True)
        return True
    else:
        print(f"❌ Error: {result.stderr}")
        return False


# Usage
if __name__ == "__main__":
    search_openalex("vision transformers medical imaging", max_papers=10)
```

---

## Understanding OpenAlex in SciLEx

### What is OpenAlex?

- **Completely free** academic database
- **200M+ papers** from all disciplines
- **Generous rate limits**: 100k requests/day (no key), more with key
- **Rich metadata**: citations, authors, venues, concepts

### How SciLEx Uses OpenAlex

1. **Builds queries** from your keywords
2. **Searches by year** (one query per year)
3. **Retrieves metadata**: title, abstract, authors, citations
4. **Handles pagination** automatically
5. **Respects rate limits** (built-in throttling)

### OpenAlex vs Other APIs

| Feature | OpenAlex | ArXiv | Semantic Scholar |
|---------|----------|-------|------------------|
| **Cost** | Free | Free | Free tier |
| **Key required** | No | No | Optional |
| **Rate limit** | 100k/day | 1/3 sec | 1/sec (free) |
| **Coverage** | All fields | Physics/CS/ML | CS/AI/Bio |
| **Peer-reviewed** | Yes | Preprints | Mixed |
| **Citations** | Yes | No | Yes |

### When to Use OpenAlex

✅ **Good for:**
- Broad literature searches
- Multi-disciplinary research
- Citation analysis
- Free, unlimited access

❌ **Not ideal for:**
- Latest preprints (use ArXiv)
- Specific CS/AI papers (use Semantic Scholar + OpenAlex)
- Medical/biomedical (add PubMed)

---

## Advanced Configuration

### Get OpenAlex API Key (Optional)

While not required, a key gives you higher limits:

1. Go to https://openalex.org/settings/api
2. Create a free account
3. Get your API key
4. Add to `scilex/api.config.yml`:

```yaml
OpenAlex:
  api_key: "your-key-here"
```

### Search Multiple APIs

```yaml
apis:
  - OpenAlex      # Broad coverage
  - Arxiv         # Latest preprints
  - SemanticScholar  # CS/AI focus (optional key)
```

### Filter by Document Type

```yaml
quality_filters:
  enable_itemtype_filter: true
  allowed_item_types:
    - journalArticle
    - conferencePaper
  require_abstract: true
  apply_citation_filter: true
  max_papers: 100
```

### Complex Keywords

```yaml
# Papers matching ANY in group 1 AND ANY in group 2
keywords:
  - ["transformer", "attention mechanism", "ViT"]
  - ["medical imaging", "radiology", "pathology"]

# Bonus keywords (boost relevance but don't filter)
bonus_keywords:
  - "deep learning"
  - "neural network"
```

---

## Troubleshooting

### "No papers found"

Try:
- Broader keywords
- More years
- Remove `require_abstract: true`

### "Rate limit exceeded"

OpenAlex has high limits, but if hit:
- Wait a few minutes
- Get a free API key
- Reduce `max_articles_per_query`

### "Import errors"

```bash
cd /mnt/d/new_repos/perspicacite_v2
source .venv/bin/activate
uv pip install -e packages_to_use/SciLEx
```

---

## Next Steps

1. **Run your first search:**
   ```bash
   cd packages_to_use/SciLEx
   scilex-collect
   ```

2. **Check results:**
   ```bash
   ls output/*/aggregated_results.csv
   ```

3. **Export for your workflow:**
   ```bash
   scilex-export-bibtex
   scilex-push-zotero  # If configured
   ```

4. **Analyze with Python:**
   ```python
   import pandas as pd
   df = pd.read_csv('output/*/aggregated_results.csv')
   print(df[['title', 'year', 'citation_count']].head())
   ```
