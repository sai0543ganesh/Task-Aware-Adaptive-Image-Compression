"""Relocate an original ImageNet-100 input manifest archive without changing its split."""

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath
import tarfile


TABLES = ("sources.csv", "validation.csv", "test.csv")
MANIFEST = "ASSET_MANIFEST.json"
ASSETS = {*TABLES, "protocol.json", "grid_split_policy.json", "optimized_v28_features.csv"}


def read_assets(archive):
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        if len(members) != len(ASSETS) + 1 or {m.name for m in members} != ASSETS | {MANIFEST}:
            raise ValueError("Unexpected input package contents.")
        if any(not m.isfile() for m in members):
            raise ValueError("The input package must contain regular files only.")
        assets = {member.name: package.extractfile(member).read() for member in members}
    expected = json.loads(assets.pop(MANIFEST))["sha256"]
    if {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()} != expected:
        raise ValueError("Input manifest checksum mismatch.")
    return assets


def relocate_table(contents, data_root):
    reader = csv.DictReader(StringIO(contents.decode("utf-8"), newline=""))
    target = StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        for field in ("path", "original_path"):
            if not row.get(field):
                continue
            parts = PurePosixPath(row[field]).parts
            if "data" not in parts or ".." in parts:
                raise ValueError(f"Cannot relocate the {field} field for {row['id']}")
            relative = parts[parts.index("data") + 1:]
            row[field] = str(data_root.joinpath(*relative))
        writer.writerow(row)
    return target.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_archive", type=Path)
    parser.add_argument("data_root", type=Path, help="Linux directory containing clean/ and extracted/")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if not all((args.data_root / name).is_dir() for name in ("clean", "extracted")):
        parser.error("data_root must contain the clean/ and extracted/ directories.")
    data_root = args.data_root.resolve()
    assets = read_assets(args.original_archive)
    for name in TABLES:
        assets[name] = relocate_table(assets[name], data_root)
    assets[MANIFEST] = json.dumps({"sha256": {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()}}).encode()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "cross_backbone_inputs.tar.gz"
    with tarfile.open(destination, "x:gz") as package:
        for name, data in assets.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            package.addfile(info, BytesIO(data))
    print(destination)


if __name__ == "__main__":
    main()
