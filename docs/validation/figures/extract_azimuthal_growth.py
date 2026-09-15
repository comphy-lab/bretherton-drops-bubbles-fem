"""Write the compact azimuthal-growth CSV from S14 and S18 receipts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "s14-azimuthal-growth.csv"


def rows_from(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in payload.get("rows") or []:
        if row.get("role") not in {"production", "fine"} or not row.get("modes"):
            continue
        for mode in row["modes"]:
            if mode.get("rightmost_real") is None:
                continue
            out.append({
                "ca_in": row["ca_in"],
                "ca_b": row["ca_b"],
                "role": row["role"],
                "film_elements": row["film_elements"],
                "ndof": row["ndof"],
                "azimuthal_k": mode["azimuthal_k"],
                "rightmost_real": mode["rightmost_real"],
                "rightmost_imag": mode["rightmost_imag"],
                "growth_over_rate": mode["growth_over_rate"],
                "label": mode["label"],
            })
    return out


def main() -> int:
    rows = rows_from(ROOT / "receipts" / "s14_azimuthal_sweep.json")
    s18 = ROOT / "receipts" / "s18_azimuthal_k3.json"
    if s18.is_file():
        rows.extend(rows_from(s18))
    rows.sort(key=lambda r: (float(r["ca_in"]), r["role"], int(r["azimuthal_k"])))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
