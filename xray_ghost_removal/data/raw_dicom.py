"""Package an inventoried RAW batch as pixel-identical DICOM review copies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from xray_ghost_removal.common.dicom import save_dicom
from xray_ghost_removal.common.paths import require_current_m1_output
from xray_ghost_removal.common.provenance import sha256


EXPORT_KIND = "pixel-identical-raw-dicom-review"
FRAME_CLASSES = {"majority_55000": "High-value frames", "other": "Low-value frames"}


def _check_destination(output_dir: Path) -> None:
    """Replace only a complete export owned by this command."""
    require_current_m1_output(output_dir)
    if not output_dir.exists():
        return
    report_path = output_dir / "manifest.json"
    if output_dir.is_symlink() or not report_path.is_file():
        raise ValueError("Output directory is not a managed RAW DICOM export")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("export_kind") != EXPORT_KIND:
        raise ValueError("Output directory is not a managed RAW DICOM export")
    expected = {"manifest.json", *(r["dicom_file"] for r in report["files"])}
    actual = {p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in output_dir.rglob("*")):
        raise ValueError("Output directory contains files outside its export manifest")


def export_batch(inventory_path: Path, output_dir: Path) -> dict:
    """Read layout and hashes from intake; never infer acquisition metadata."""
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    data_dir = Path(inventory["input_root"]).resolve()
    destination = output_dir.resolve()
    if destination == data_dir or destination in data_dir.parents or data_dir in destination.parents:
        raise ValueError("RAW input and DICOM output directories must be separate")
    if inventory_path.resolve() == destination or destination in inventory_path.resolve().parents:
        raise ValueError("DICOM output must not contain the input inventory")
    _check_destination(output_dir)
    layout = inventory["assumed_layout"]
    shape = tuple(layout["shape"])
    if len(shape) != 2 or min(shape) < 1 or layout["dtype"] not in ("<u2", ">u2") or layout["header_bytes"] != 0:
        raise ValueError("Expected a headerless two-dimensional unsigned 16-bit RAW inventory")
    dtype = np.dtype(layout["dtype"])
    rows = [r for group in inventory["groups"] for r in group["files"]]
    names = {r["file"] for r in rows}
    actual = {p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*.raw")}
    if not rows or len(names) != len(rows) or names != actual or len(rows) != inventory["total_files"]:
        raise ValueError("RAW files differ from the input inventory; run inspect-raw again")
    for row in rows:
        path = data_dir / row["file"]
        if (not path.resolve().is_relative_to(data_dir)
                or path.stat().st_size != int(np.prod(shape)) * dtype.itemsize
                or row["observed_class"] not in FRAME_CLASSES):
            raise ValueError(f"Invalid inventoried RAW input: {row['file']}")

    records, studies = [], []
    groups = sorted(inventory["groups"], key=lambda g: min(r["filename_timestamp"] for r in g["files"]))
    with tempfile.TemporaryDirectory(prefix="raw-dicom-") as temporary:
        stage = Path(temporary)
        for group_index, group in enumerate(groups, 1):
            study_uid = generate_uid()
            group_rows = sorted(group["files"], key=lambda r: r["file"])
            series = {}
            for class_index, (frame_class, label) in enumerate(FRAME_CLASSES.items(), 1):
                members = [r for r in group_rows if r["observed_class"] == frame_class]
                if not members:
                    continue
                low = min(r["sample_quantiles"]["p1"] for r in members)
                high = max(r["sample_quantiles"]["p99"] for r in members)
                series[frame_class] = dict(
                    series_instance_uid=generate_uid(), series_number=class_index,
                    description=label, count=len(members),
                    window_center=float((low + high + 1) / 2),
                    window_width=float(max(high - low + 1, 2)),
                )
            for instance, row in enumerate(group_rows, 1):
                source = data_dir / row["file"]
                raw_bytes = source.read_bytes()
                if hashlib.sha256(raw_bytes).hexdigest() != row["sha256"]:
                    raise ValueError(f"RAW hash differs from intake: {row['file']}")
                image = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
                info = series[row["observed_class"]]
                template = Dataset()
                template.SpecificCharacterSet = "ISO_IR 192"
                template.PatientName = "RAW^REVIEW"
                template.PatientID = "XRAY-RAW-REVIEW"
                template.StudyID = f"RAW-G{group_index:02d}"
                template.StudyDescription = Path(group["directory"]).name or data_dir.name
                template.Modality = "OT"
                template.SeriesNumber = info["series_number"]
                template.InstanceNumber = instance
                template.Rows, template.Columns = shape
                template.BitsAllocated = template.BitsStored = 16
                template.HighBit = 15
                template.PixelRepresentation = 0
                template.SamplesPerPixel = 1
                template.PhotometricInterpretation = "MONOCHROME2"
                template.WindowCenter = info["window_center"]
                template.WindowWidth = info["window_width"]
                template.ImageComments = f"RAW source: {row['file']}"
                relative = Path(row["file"]).with_suffix(".dcm")
                audit = save_dicom(
                    image, template, stage / relative,
                    study_uid=study_uid, series_uid=info["series_instance_uid"],
                    series_description=info["description"],
                    derivation_description=(
                        "Pixel-identical RAW review copy; unconfirmed layout from intake. "
                        "Filename time is not acquisition time."
                    ),
                )
                if audit["clipped_low"] or audit["clipped_high"]:
                    raise RuntimeError("RAW review export must not clip stored pixels")
                records.append(dict(
                    source_file=row["file"], source_sha256=row["sha256"],
                    dicom_file=relative.as_posix(), dicom_sha256=sha256(stage / relative),
                    observed_class=row["observed_class"],
                    filename_timestamp=row["filename_timestamp"],
                    filename_counter=row["filename_counter"], instance_number=instance,
                    **audit,
                ))
            studies.append(dict(
                directory=group["directory"], study_instance_uid=study_uid,
                count=len(group_rows), series=series,
            ))
            print(f"{group['directory']}: {len(group_rows)} DICOMs verified", flush=True)
        uids = [r["sop_instance_uid"] for r in records]
        if len(set(uids)) != len(records):
            raise RuntimeError("DICOM instance UIDs must be unique")
        report = dict(
            export_kind=EXPORT_KIND, generated_at=datetime.now(timezone.utc).isoformat(),
            inventory_path=str(inventory_path.resolve()), inventory_sha256=sha256(inventory_path),
            exporter_sha256=sha256(Path(__file__)), input_root=str(data_dir),
            assumed_layout=layout, total_files=len(records), total_studies=len(studies),
            total_series=sum(len(g["series"]) for g in studies),
            pixel_round_trip_equal=all(r["round_trip_equal"] for r in records),
            notes=[
                "Unsigned 16-bit stored pixels preserved; no correction, scaling, rotation or resampling.",
                "Each original directory is a study with separate high-value and low-value series.",
                "InstanceNumber retains filename order within the original directory, including gaps after splitting series.",
                "Each series shares a display window spanning its input frames' sampled P1--P99 ranges; pixels are unchanged.",
                "High/low are observed value classes, not verified exposure/dark acquisition roles.",
                "Viewer patient/study identifiers are synthetic; no acquisition time, exposure or pixel spacing is inferred.",
                "Filename timestamps and counters are provenance only; layout and orientation remain unconfirmed.",
            ],
            studies=studies, files=records,
        )
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Publish only after every source hash and DICOM read-back has passed.
        _check_destination(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(stage), str(output_dir))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("outputs/raw_intake/inventory.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/raw_intake/dicom"))
    args = parser.parse_args()
    report = export_batch(args.inventory, args.output_dir)
    print(json.dumps({k: report[k] for k in ("total_files", "total_studies", "total_series", "pixel_round_trip_equal")}, indent=2))
    print(f"Import this entire directory into Horos: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
