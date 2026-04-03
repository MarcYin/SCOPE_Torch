from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import xarray as xr

from ..io import (
    NetCDFWriteOptions,
    ScopeInputFiles,
    prepare_scope_input_dataset,
    read_s2_bio_inputs,
    write_netcdf_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a runner-ready SCOPE input dataset from weather, observation, and Sentinel-2 bio inputs."
    )
    parser.add_argument("--weather", required=True, help="Path to the weather NetCDF dataset.")
    parser.add_argument(
        "--observation",
        required=True,
        help="Path to the observation NetCDF dataset carrying delta_time and viewing geometry.",
    )
    parser.add_argument("--bio-npz", required=True, help="Path to the historic Sentinel-2 bio NPZ bundle.")
    parser.add_argument("--year", required=True, type=int, help="Acquisition year used to decode NPZ DOY values.")
    parser.add_argument("--output", required=True, help="Output NetCDF path.")
    parser.add_argument("--s2-reference", help="Optional Sentinel-2 reference dataset used to copy CRS metadata.")
    parser.add_argument("--scope-root", help="Optional upstream SCOPE root used to resolve default file references.")
    parser.add_argument("--optipar-file", help="Optional FLUSPECT parameter file override.")
    parser.add_argument("--soil-file", help="Optional soil spectra file override.")
    parser.add_argument("--atmos-file", help="Optional atmosphere file override.")
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Optional spatial subset bounds.",
    )
    parser.add_argument("--time-start", help="Optional inclusive time subset start.")
    parser.add_argument("--time-end", help="Optional inclusive time subset end.")
    parser.add_argument(
        "--netcdf-engine",
        choices=("netcdf4", "h5netcdf", "scipy"),
        help="Optional NetCDF backend override.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Compression level for HDF5-backed NetCDF engines.",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable NetCDF variable compression even when the selected backend supports it.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> None:
    with xr.open_dataset(args.weather) as weather_ds, xr.open_dataset(args.observation) as observation_ds:
        reference_ds = xr.open_dataset(args.s2_reference) if args.s2_reference else None
        try:
            post_bio_da, post_bio_scale_da = read_s2_bio_inputs(
                args.bio_npz, year=args.year, reference_dataset=reference_ds
            )

            time_slice = None
            if args.time_start or args.time_end:
                time_slice = slice(args.time_start, args.time_end)

            dataset = prepare_scope_input_dataset(
                weather_ds,
                observation_ds,
                post_bio_da,
                post_bio_scale_da,
                scope_root_path=args.scope_root,
                scope_files=ScopeInputFiles(
                    optipar_file=args.optipar_file,
                    soil_file=args.soil_file,
                    atmos_file=args.atmos_file,
                ),
                bounds=args.bounds,
                time_slice=time_slice,
            )
            write_netcdf_dataset(
                dataset,
                Path(args.output),
                options=NetCDFWriteOptions(
                    engine=args.netcdf_engine,
                    compression=not args.no_compression,
                    compression_level=args.compression_level,
                ),
            )
        finally:
            if reference_ds is not None:
                reference_ds.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
