import pytest


@pytest.mark.unit
def test_indicium_importable_and_validates():
    indicium = pytest.importorskip("indicium")
    import rdflib
    g = rdflib.Graph()
    g.parse(data=f'''
@prefix asb: <{indicium.ASB_BASE}> .
<urn:c> a asb:Claim ; asb:context "c" ; asb:subject "s" ;
    asb:qualifier "causes" ; asb:relation "r" ; asb:object "o" .
''', format="turtle")
    conforms, _ = indicium.validate_graph(g)
    assert conforms is True
