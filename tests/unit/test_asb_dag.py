"""workflow_dag.json reader. Supports both 2026-05-15 and 2026-05-16
on-disk edge formats and preserves port labels from the newer format."""
from pathlib import Path

METLINKR = Path(__file__).parent.parent / "fixtures" / "asb" / "metlinkr_subset"
ARTICLE = Path(__file__).parent.parent / "fixtures" / "asb" / "article_878_v4_subset"


# ---------- 2026-05-15 format (list-of-pair edges) ----------

def test_load_dag_returns_nodes_and_edges_metlinkr():
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(METLINKR)
    assert "task_001" in dag.nodes
    assert "task_002" in dag.downstream("task_001")


def test_legacy_edges_have_no_port_label():
    """2026-05-15 edges are pairs — no port."""
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(METLINKR)
    port = dag.edge_port("task_001", "task_002")
    assert port is None


def test_dag_upstream_downstream_maps_metlinkr():
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(METLINKR)
    assert "task_002" in dag.downstream("task_001")
    assert "task_001" in dag.upstream("task_002")
    # task_001 has no upstream
    assert dag.upstream("task_001") == []


# ---------- 2026-05-16 format (dict edges with ports) ----------

def test_load_dag_handles_dict_edges_article():
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(ARTICLE)
    assert "task_001" in dag.nodes
    # task_001 fans out to multiple downstream tasks via "cleaned_library" port
    downstream = dag.downstream("task_001")
    assert "task_002" in downstream


def test_dict_edges_preserve_port_label_article():
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(ARTICLE)
    # task_001 → task_002 carries the "cleaned_library" data flow
    port = dag.edge_port("task_001", "task_002")
    assert port == "cleaned_library"


def test_dag_to_dict_roundtrip_article():
    """to_dict preserves ports so downstream KB-description carries them."""
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(ARTICLE)
    d = dag.to_dict()
    assert "nodes" in d and "edges" in d
    # Edges as list of dicts with from/to/port
    e0 = d["edges"][0]
    assert "from" in e0 and "to" in e0
    assert "port" in e0


# ---------- error paths ----------

def test_dag_missing_file_returns_empty(tmp_path):
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(tmp_path)
    assert dag.nodes == []
    assert dag.edges == []


def test_dag_invalid_json_returns_empty(tmp_path):
    (tmp_path / "workflow_dag.json").write_text("{not json}")
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(tmp_path)
    assert dag.nodes == []
    assert dag.edges == []


def test_dag_edge_port_unknown_pair_returns_none():
    from perspicacite.pipeline.asb.dag import load_workflow_dag
    dag = load_workflow_dag(ARTICLE)
    assert dag.edge_port("task_nonexistent", "task_other") is None
