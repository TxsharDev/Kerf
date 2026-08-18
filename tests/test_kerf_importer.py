"""CSV importer: cutting lists from shop-floor files."""

import csv
import re
from pathlib import Path

import pytest

from kerf.importer import load_cutting_list
from kerf.models import CuttingList

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cutting_list.csv"


def _write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "list.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_sample_fixture(tmp_path):
    cl = load_cutting_list(FIXTURE, customer_ref="PO-123")
    assert isinstance(cl, CuttingList)
    assert cl.customer_ref == "PO-123"
    assert len(cl.parts) == 10
    first = cl.parts[0]
    assert (first.id, first.material_id, first.length, first.quantity) == (
        "p1",
        "al-6061",
        2500,
        2,
    )
    assert first.label == "main beam"
    last = cl.parts[-1]
    assert (last.id, last.material_id, last.length) == ("p10", "alu-tube", 600)
    assert cl.status == "open"


def test_default_id_shape_and_uniqueness():
    a = load_cutting_list(FIXTURE)
    b = load_cutting_list(FIXTURE)
    assert re.fullmatch(r"cl-[0-9a-f]{12}", a.id)
    assert re.fullmatch(r"cl-[0-9a-f]{12}", b.id)
    assert a.id != b.id


def test_explicit_cutting_list_id(tmp_path):
    cl = load_cutting_list(FIXTURE, cutting_list_id="cl-custom")
    assert cl.id == "cl-custom"


def test_empty_data_rows(tmp_path):
    path = _write_csv(tmp_path, "part_id,material_id,length_mm,quantity,label\n")
    cl = load_cutting_list(path)
    assert cl.parts == []


def test_blank_lines_skipped(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label\n"
        "p1,m1,100,1,a\n"
        "\n"
        "p2,m1,200,2,b\n",
    )
    cl = load_cutting_list(path)
    assert [p.id for p in cl.parts] == ["p1", "p2"]


def test_empty_label_becomes_empty_string(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label\n"
        "p1,m1,100,1,\n",
    )
    cl = load_cutting_list(path)
    assert cl.parts[0].label == ""


def test_missing_required_header_names_column(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,label\np1,m1,100,1,x\n",
    )
    with pytest.raises(ValueError, match="quantity"):
        load_cutting_list(path)


def test_extra_column_rejected(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label,notes\n"
        "p1,m1,100,1,x,n\n",
    )
    with pytest.raises(ValueError, match="notes"):
        load_cutting_list(path)


def test_bad_length_names_row_number(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label\n"
        "p1,m1,100,1,ok\n"
        "p2,m1,abc,2,bad\n",
    )
    with pytest.raises(ValueError, match=r"row 2"):
        load_cutting_list(path)


def test_noninteger_length_rejected(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label\n"
        "p1,m1,12.5,1,bad\n",
    )
    with pytest.raises(ValueError, match=r"row 1"):
        load_cutting_list(path)


def test_zero_length_rejected_with_row(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label\n"
        "p1,m1,0,1,bad\n",
    )
    with pytest.raises(ValueError, match=r"row 1"):
        load_cutting_list(path)


def test_zero_quantity_rejected_with_row(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label\n"
        "p1,m1,100,0,bad\n",
    )
    with pytest.raises(ValueError, match=r"row 1"):
        load_cutting_list(path)


def test_missing_required_cell_names_row(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label\n"
        "p1,m1,100,1,ok\n"
        ",m1,200,2,no id\n",
    )
    with pytest.raises(ValueError, match=r"row 2"):
        load_cutting_list(path)


def test_duplicate_part_id_names_row(tmp_path):
    path = _write_csv(
        tmp_path,
        "part_id,material_id,length_mm,quantity,label\n"
        "p1,m1,100,1,a\n"
        "p2,m1,200,2,b\n"
        "p1,m1,300,3,c\n",
    )
    with pytest.raises(ValueError, match=r"row 3"):
        load_cutting_list(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cutting_list(tmp_path / "nope.csv")
