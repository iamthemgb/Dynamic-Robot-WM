from __future__ import annotations

from pathlib import Path

import pytest

from dynamic_robot_dataset.common.arrow_schema import canonical_sidecar_table
from dynamic_robot_dataset.common.episode_writer import write_parquet_atomic


pytestmark = pytest.mark.integration


def test_declared_all_null_extension_has_no_per_episode_parquet_type_drift(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    declaration = {"diagnostic.vector": "list<float64>"}
    first = canonical_sidecar_table(
        pa,
        "high_rate",
        [{"episode_index": 0, "timestamp": 0.0, "diagnostic.vector": None}],
        declared_extra_fields=declaration,
    )
    second = canonical_sidecar_table(
        pa,
        "high_rate",
        [{"episode_index": 1, "timestamp": 0.0, "diagnostic.vector": [1, 2.5]}],
        declared_extra_fields=declaration,
    )
    first_path = write_parquet_atomic(tmp_path / "first.parquet", first)
    second_path = write_parquet_atomic(tmp_path / "second.parquet", second)

    first_schema = pq.ParquetFile(first_path).schema_arrow
    second_schema = pq.ParquetFile(second_path).schema_arrow
    assert first_schema == second_schema
    assert first_schema.metadata[b"schema_identity_sha256"] == second_schema.metadata[
        b"schema_identity_sha256"
    ]

