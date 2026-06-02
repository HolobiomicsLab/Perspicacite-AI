import unittest


class TestTableChunks(unittest.TestCase):
    def test_table_records_become_table_chunks(self):
        from perspicacite.pipeline.parsers.docling_pdf import DoclingTable
        from perspicacite.pipeline.chunking_dispatch import table_records_to_chunks

        class _Paper:
            paper_id = "local:abc"
            title = "T"; doi = None; year = None
        tables = [DoclingTable(page=3, caption="Table 1. Params.",
                               markdown="| k | v |\n| a | 1 |", headers=["k", "v"], rows=[["a", "1"]])]
        chunks = table_records_to_chunks(tables, _Paper(), start_index=0)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.metadata.content_type == "table"
        assert c.metadata.page == 3
        assert "Table 1" in c.text and "| k | v |" in c.text

    def test_empty_tables_yield_no_chunks(self):
        from perspicacite.pipeline.chunking_dispatch import table_records_to_chunks
        class _Paper:
            paper_id = "p"; title = None; doi = None; year = None
        assert table_records_to_chunks([], _Paper(), start_index=5) == []
