"""Reporting helpers: exact property-type comparisons and file provenance."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_type_context(screen, building_id):
    """Count exact named property types, never a pooled fallback category."""
    selected = screen[screen.building_id.eq(building_id)]
    if len(selected) != 1:
        raise ValueError("Expected exactly one screening row for the building")
    building = selected.iloc[0]
    group = screen[screen.three_year & screen.property_type.eq(building.property_type)]
    peers = group[~group.building_id.eq(building_id)]
    percentile = (100 * peers.mean_gap_kbtu.lt(building.mean_gap_kbtu).mean()
                  if len(peers) else None)
    return {"property_type": building.property_type,
            "same_type_three_year_buildings": len(group), "other_same_type_buildings": len(peers),
            "gap_percentile_strictly_lower": percentile}


def verify_analysis_files(folder):
    """Stop report generation when core files were replaced by another run."""
    folder = Path(folder)
    manifest_path = folder / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Missing run_manifest.json. Run the updated script 11 before scripts 12 and 13.")
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["artifact_sha256"].items():
        path = folder / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Analysis file changed or is missing: {name}. Rerun scripts 11, 12, and 13 in order.")
    return manifest
