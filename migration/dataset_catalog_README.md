# Robotics dataset catalog

This directory is a generated, non-destructive index. The source datasets stay
at their original absolute paths; the category view contains symlinks only.
Nothing in a refresh moves, renames, unlinks, or deletes a dataset.

`manifest.json` is the machine-readable catalog and `manifest.tsv` is the flat
review table. Each content-addressed directory under `views/` is an immutable
snapshot grouped into canonical run states, current and historical plans,
legacy assisted or smoke data, previews, and read-only dependencies. Consult
the manifest's `symlink_view` field for the view produced by the latest scan.

Refresh with:

```bash
python3 /gpfs/radev/project/sous/zl664/dataset_generation/tools/catalog_datasets.py \
  --roots-config /gpfs/radev/project/sous/zl664/dataset_generation/migration/dataset_catalog_roots.json \
  --catalog-root /gpfs/radev/project/sous/zl664/dataset_catalog \
  --build-symlink-view
```

Git should track the scanner, roots declaration, and this policy document. Do
not add MP4, Parquet, NumPy, staging, cache, virtual-environment, or generated
dataset trees to Git. If the external catalog root later becomes its own Git
repository, track only its README and manifest files; its absolute symlink view
is machine-specific.

The lifecycle fields are intentionally independent: `sealed` means finalized
and immutable, not approved. A run enters training only when
`release_eligible=true`; the catalog keeps automated QC and human-review state
as separate columns. Legacy assisted datasets remain quarantined even when
their metadata says complete. Known qualitative failures are declared as
non-destructive catalog overrides in `dataset_catalog_roots.json`; this labels
the index without modifying an immutable sealed dataset.
