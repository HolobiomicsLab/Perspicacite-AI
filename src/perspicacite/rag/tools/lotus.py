"""
LOTUS Natural Products Database Search Tool

Integrates with https://lotus.naturalproducts.net/ API to search for:
- Natural products by name, SMILES, InChI, InChIKey
- Substructure and similarity searches
- Taxonomic data (organisms producing compounds)
- Literature references

Usage:
    tool = LotusSearchTool()
    results = await tool.execute(query="quercetin", search_type="simple")
"""

from __future__ import annotations

from typing import Any

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.rag.tools.lotus")


class LotusSearchTool:
    """
    Tool to search the LOTUS natural products database.
    
    LOTUS (Natural Products Online) is a comprehensive database of:
    - 200,000+ natural products
    - Chemical structures (SMILES, InChI)
    - Producing organisms (taxonomy)
    - Literature references (with DOIs)
    - Molecular descriptors
    
    API Docs: https://lotus.naturalproducts.net/documentation
    """
    
    name = "lotus_search"
    description = """Search for natural products in the LOTUS database.
    
    Use this tool when the query involves:
    - Specific chemical compounds (natural products)
    - Chemical structures or SMILES
    - Organisms that produce compounds
    - Biosynthesis or secondary metabolites
    - Traditional medicine compounds
    
    Returns: Compound name, structure, taxonomy, and literature references.
    """
    
    BASE_URL = "https://lotus.naturalproducts.net/api"
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
    
    async def execute(
        self,
        query: str,
        search_type: str = "simple",
        max_results: int = 10,
    ) -> str:
        """
        Execute a LOTUS search.
        
        Args:
            query: Search query (name, SMILES, InChI, etc.)
            search_type: Type of search - "simple", "exact", "substructure"
            max_results: Maximum number of results to return
            
        Returns:
            Formatted string with search results
        """
        try:
            if search_type == "simple":
                results = await self._simple_search(query, max_results)
            elif search_type == "exact":
                results = await self._exact_structure_search(query, max_results)
            elif search_type == "substructure":
                results = await self._substructure_search(query, max_results)
            else:
                return f"Error: Unknown search type '{search_type}'"
            
            if not results:
                return f"No natural products found for '{query}' in LOTUS database."
            
            return self._format_results(results, query)
            
        except Exception as e:
            logger.error("lotus_search_error", query=query, error=str(e))
            return f"Error searching LOTUS: {str(e)}"
    
    async def _simple_search(
        self,
        query: str,
        max_results: int,
    ) -> list[dict]:
        """Simple search by name, InChI, InChIKey, or formula."""
        url = f"{self.BASE_URL}/search/simple"
        params = {"query": query}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        products = data.get("naturalProducts", [])
        return products[:max_results]
    
    async def _exact_structure_search(
        self,
        smiles: str,
        max_results: int,
    ) -> list[dict]:
        """Search by exact SMILES structure."""
        url = f"{self.BASE_URL}/search/exact-structure"
        params = {"type": "smiles", "smiles": smiles}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        products = data.get("naturalProducts", [])
        return products[:max_results]
    
    async def _substructure_search(
        self,
        smiles: str,
        max_results: int,
        algorithm: str = "default",
    ) -> list[dict]:
        """
        Substructure search by SMILES.
        
        Args:
            smiles: SMILES of substructure to search for
            max_results: Max hits to return
            algorithm: "default" (Ullmann), "df" (depth-first), or "vf" (Vento-Foggia)
        """
        url = f"{self.BASE_URL}/search/substructure"
        params = {
            "type": algorithm,
            "max-hits": max_results,
            "smiles": smiles,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        products = data.get("naturalProducts", [])
        return products[:max_results]
    
    async def get_compound_details(self, lotus_id: str) -> dict | None:
        """
        Get detailed information about a specific compound.
        
        Args:
            lotus_id: LOTUS ID (e.g., "LTS0205097")
            
        Returns:
            Compound details or None if not found
        """
        # LOTUS simple search by ID returns the compound
        results = await self._simple_search(lotus_id, max_results=1)
        return results[0] if results else None
    
    def _format_results(self, products: list[dict], query: str) -> str:
        """Format search results for the LLM."""
        lines = [
            f"LOTUS Natural Products Database Search Results for '{query}':",
            f"Found {len(products)} compound(s)",
            "",
        ]
        
        for i, product in enumerate(products, 1):
            lines.extend(self._format_compound(product, i))
            lines.append("")  # Blank line between compounds
        
        return "\n".join(lines)
    
    def _format_compound(self, product: dict, index: int) -> list[str]:
        """Format a single compound."""
        lines = [
            f"{index}. {product.get('traditional_name', 'Unknown')}",
        ]
        
        # Basic info
        if product.get('iupac_name'):
            iupac = product['iupac_name']
            if len(iupac) > 100:
                iupac = iupac[:100] + "..."
            lines.append(f"   IUPAC: {iupac}")
        
        lines.append(f"   LOTUS ID: {product.get('lotus_id', 'N/A')}")
        
        # Chemical properties
        if product.get('molecular_formula'):
            lines.append(f"   Formula: {product.get('molecular_formula')}")
        
        if product.get('molecular_weight'):
            mw = float(product['molecular_weight'])
            lines.append(f"   Molecular Weight: {mw:.2f} Da")
        
        # Classification
        np_class = product.get('chemicalTaxonomyNPclassifierClass')
        if np_class:
            lines.append(f"   NP Class: {np_class}")
        
        superclass = product.get('chemicalTaxonomyNPclassifierSuperclass')
        if superclass:
            lines.append(f"   NP Superclass: {superclass}")
        
        # Structure
        if product.get('smiles2D'):
            smiles = product['smiles2D']
            if len(smiles) > 80:
                smiles = smiles[:80] + "..."
            lines.append(f"   SMILES: {smiles}")
        
        # Organisms (taxa)
        taxa = product.get('allTaxa', [])
        if taxa:
            lines.append(f"   Producing Organisms ({len(taxa)}):")
            # Show first 3 organisms
            for taxon in taxa[:3]:
                if isinstance(taxon, str):
                    lines.append(f"      - {taxon}")
                elif isinstance(taxon, dict):
                    organism = taxon.get('name', 'Unknown')
                    rank = taxon.get('rank', '')
                    if rank:
                        lines.append(f"      - {organism} ({rank})")
                    else:
                        lines.append(f"      - {organism}")
            if len(taxa) > 3:
                lines.append(f"      ... and {len(taxa) - 3} more")
        
        # References
        refs = product.get('taxonomyReferenceObjects', {})
        if refs and isinstance(refs, dict):
            lines.append(f"   Literature References: {len(refs)} with taxonomic data")
        
        # Synonyms
        synonyms = product.get('synonyms', [])
        if synonyms and len(synonyms) > 0:
            syn_str = ", ".join(synonyms[:3])
            lines.append(f"   Also known as: {syn_str}")
        
        return lines
    
    def _extract_search_terms(self, product: dict) -> list[str]:
        """
        Extract useful search terms from a compound for literature search.
        
        Returns list of terms to search in academic databases.
        """
        terms = []
        
        # Primary name
        if product.get('traditional_name'):
            terms.append(product['traditional_name'])
        
        # IUPAC name (shortened)
        if product.get('iupac_name'):
            iupac = product['iupac_name']
            # Extract base name without stereochemistry
            terms.append(iupac.split('(')[0].strip())
        
        # Synonyms
        terms.extend(product.get('synonyms', [])[:2])
        
        # Chemical class
        np_class = product.get('chemicalTaxonomyNPclassifierClass')
        if np_class:
            terms.append(np_class)
        
        return list(set(terms))  # Remove duplicates


# Backward compatibility alias
LotusTool = LotusSearchTool
