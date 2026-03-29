"""Document processing pipeline."""

__all__ = ["KBBuilder", "PDFDownloader", "DownloadResult", "ContentResult"]

# Lazy imports
def __getattr__(name):
    if name == "KBBuilder":
        from perspicacite.pipeline.kb_builder import KBBuilder
        return KBBuilder
    if name == "PDFDownloader":
        from perspicacite.pipeline.download.base import PDFDownloader
        return PDFDownloader
    if name == "DownloadResult":
        from perspicacite.pipeline.download.base import DownloadResult
        return DownloadResult
    if name == "ContentResult":
        from perspicacite.pipeline.download.base import ContentResult
        return ContentResult
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
