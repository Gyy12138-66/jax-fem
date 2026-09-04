#!/usr/bin/env python3
"""Generate the V1-E bare-plate material config from the V1-P parity config.

V1-E (code-to-experiment leg promised in deviations D-V1-07) models the
NIST AMB2018-02 bare IN625 plate: solid metal everywhere, no powder-model
contribution. Transform relative to v1_material_config.json:

  * powder properties := solid properties (rho, k table, cp table) - the
    20 um top band of the mesh is then just plate surface material;
  * emissivity := 0.4 (Balbaa Table 1 solid-surface value; the parity
    config's 0.5312 is a POWDER-BED composite, wrong for a bare plate);
  * absorptivity := REQUIRED CLI input. The parity value 0.62 is Balbaa's
    powder DRS measurement and has no bare-plate provenance (review 4.4);
    the frozen V1-E protocol must supply a sourced value or interval.

Table paths are re-rooted with --table-root (the parity config pins the
box-159 absolute root).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--absorptivity", type=float, required=True,
                help="bare-plate absorptivity (sourced; see V1PE-RERUN-PLAN)")
ap.add_argument("--source", type=Path,
                default=Path(__file__).parent / "v1_material_config.json")
ap.add_argument("--output", type=Path,
                default=Path(__file__).parent / "v1e_material_config.json")
ap.add_argument("--table-root", default=None,
                help="replace the leading .../jax-fem in table paths")
args = ap.parse_args()

with open(args.source, encoding="utf-8") as fh:
    config = json.load(fh)

if not 0.0 < args.absorptivity <= 1.0:
    raise SystemExit("absorptivity must lie in (0, 1]")

config["_comment"] = (
    "V1-E bare-plate config derived from v1_material_config.json by "
    "make_v1e_material_config.py: powder properties collapsed to solid "
    "(no powder-model contribution, D-V1-07), solid-surface emissivity, "
    "bare-plate absorptivity supplied at generation time."
)
config["material_name"] = "IN625 - NIST AMB2018-02 bare plate (V1-E)"
config["rho_powder"] = config["rho_solid"]
config["k_table_powder"] = config["k_table_solid"]
config["cp_table_powder"] = config["cp_table_solid"]
config["emissivity"] = 0.4
config["_emissivity_note"] = (
    "solid-surface emissivity (Balbaa Table 1 es=0.4); the parity value "
    "0.5312 is a powder-bed composite and does not apply to a bare plate"
)
config["absorptivity"] = args.absorptivity
config["_absorptivity_note"] = (
    "bare-plate value supplied via make_v1e_material_config.py "
    "--absorptivity; provenance must be registered in the frozen V1-E "
    "protocol (the parity 0.62 is powder DRS and is NOT valid here)"
)

if args.table_root:
    root = args.table_root.rstrip("/")
    for key, value in list(config.items()):
        if key.endswith(("_table", "_table_solid", "_table_powder")) or (
            key.startswith(("k_table", "cp_table", "E_table", "alpha_table",
                            "yield_table", "hardening_table"))
        ):
            if isinstance(value, str) and "/jax-fem/" in value:
                config[key] = root + "/" + value.split("/jax-fem/", 1)[1]

with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(config, fh, indent=1)
    fh.write("\n")
print(f"wrote {args.output} (absorptivity={args.absorptivity})")
