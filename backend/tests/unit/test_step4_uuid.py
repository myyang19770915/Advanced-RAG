from app.pipelines.step4_ingestion import point_id_for


def test_point_id_is_deterministic():
    a = point_id_for("doc.pdf", 0)
    b = point_id_for("doc.pdf", 0)
    assert a == b


def test_point_id_changes_with_index_and_file():
    base = point_id_for("doc.pdf", 0)
    assert base != point_id_for("doc.pdf", 1)
    assert base != point_id_for("other.pdf", 0)
