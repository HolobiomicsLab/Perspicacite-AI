"""Document processing pipeline."""

__all__ = ["KBBuilder", "PDFDownloader", "DownloadResult", "PaperParser", "Unpaywall"]

# Lazy imports
def __getattr__(name):
    if name == "KBBuilder":
        from perspicacite.pipeline.kb_builder import KBBuilder
        return KBBuilder
    if name == "PDFDownloader":
        from perspicacite.pipeline.download import PDFDownloader
        return PDFDownloader
    if name == "DownloadResult":
        from perspicacite.pipeline.download import DownloadResult
        return DownloadResult
    if name == "PaperParser":
        from perspicacite.pipeline.download import PaperParser
        return PaperParser
    if name == "Unpaywall":
        from perspicacite.pipeline.download import Unpaywall
        return Unpaywall
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
