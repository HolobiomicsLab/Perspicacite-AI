"""Text chunking for RAG operations."""


class SimpleChunker:
    """Simple text chunker for RAG operations."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        if not text:
            return []
        
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        step = max(1, self.chunk_size - self.overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
        
        return chunks


def create_chunker(chunk_size: int = 1000, overlap: int = 200) -> SimpleChunker:
    """Create a simple text chunker."""
    return SimpleChunker(chunk_size=chunk_size, overlap=overlap)
