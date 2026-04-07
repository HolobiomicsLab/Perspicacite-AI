"""Legacy fallback orchestrators — kept for backward compatibility.

NOTE: These functions are deprecated. Use ``retrieve_paper_content()`` from
``perspicacite.pipeline.download.unified`` instead, which implements the
correct quality-based priority flow (structured > PDF > abstract > discard).
"""

# These functions are preserved for any external code that may import them.
# They will be removed in a future version.

from .unified import retrieve_paper_content  # noqa: F401 — re-export


async def get_pdf_with_fallback(*args, **kwargs):
    """Deprecated: Use retrieve_paper_content() instead."""
    import warnings
    warnings.warn(
        "get_pdf_with_fallback is deprecated. Use retrieve_paper_content() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .unified import retrieve_paper_content
    result = await retrieve_paper_content(*args, **kwargs)
    if result.success and result.full_text:
        return result.full_text.encode("utf-8")
    return None


async def get_content_with_fallback(*args, **kwargs):
    """Deprecated: Use retrieve_paper_content() instead."""
    import warnings
    warnings.warn(
        "get_content_with_fallback is deprecated. Use retrieve_paper_content() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .base import ContentResult
    from .unified import retrieve_paper_content
    result = await retrieve_paper_content(*args, **kwargs)
    if result.success:
        return ContentResult(
            success=True,
            content=result.full_text,
            content_type="text" if result.content_type == "structured" else result.content_type,
            source=result.content_source,
            metadata={"sections": result.sections},
        )
    return ContentResult(
        success=False,
        content=None,
        content_type="unknown",
        source="none",
        error="Content not found",
    )
