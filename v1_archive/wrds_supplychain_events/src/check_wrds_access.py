#!/usr/bin/env python3
"""Check WRDS connectivity and inspect available libraries/tables.

Usage:
  python wrds_supplychain_events/src/check_wrds_access.py
  python wrds_supplychain_events/src/check_wrds_access.py --show-all-libraries
  python wrds_supplychain_events/src/check_wrds_access.py --output-csv wrds_supplychain_events/data_processed/wrds_table_catalog.csv
  python wrds_supplychain_events/src/check_wrds_access.py --only-compustat-supplychain

Environment variables:
  WRDS_USERNAME
  WRDS_PASSWORD (optional; if omitted, WRDS may prompt)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable, List

import wrds

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _print_table(title: str, rows: Iterable[str], limit: int) -> None:
    items: List[str] = sorted(set(rows))
    shown = items[:limit]

    print(f"\n{title} ({len(items)} total)")
    if not shown:
        print("  - none found")
        return

    for name in shown:
        print(f"  - {name}")

    if len(items) > limit:
        print(f"  ... showing first {limit}")


def _table_catalog_query(libraries: List[str]) -> str:
    libs_sql = ", ".join(f"'{lib}'" for lib in libraries)
    return f"""
        SELECT
            n.nspname AS library,
            c.relname AS table_name,
            COALESCE(obj_description(c.oid), '') AS description
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n
            ON n.oid = c.relnamespace
        WHERE n.nspname IN ({libs_sql})
          AND c.relkind IN ('r', 'v', 'm', 'f', 'p')
        ORDER BY n.nspname, c.relname
    """


def _filter_supplychain_rows(rows):
    keywords = (
        "supplier",
        "customer",
        "supply",
        "chain",
        "relationship",
        "partner",
        "segment",
        "major customer",
    )
    mask = (
        rows["table_name"].fillna("").str.lower().str.contains("|".join(keywords))
        | rows["description"].fillna("").str.lower().str.contains("|".join(keywords))
    )
    return rows.loc[mask].copy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to WRDS, list libraries, and print table names for "
            "Compustat/SEC-related libraries."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max rows shown per section (default: 100)",
    )
    parser.add_argument(
        "--show-all-libraries",
        action="store_true",
        help="Print all libraries visible to your account (default shows matched libs).",
    )
    parser.add_argument(
        "--output-csv",
        default="wrds_supplychain_events/data_processed/wrds_table_catalog.csv",
        help=(
            "Output CSV path for table catalog with columns: library, table_name, description "
            "(default: wrds_supplychain_events/data_processed/wrds_table_catalog.csv)"
        ),
    )
    parser.add_argument(
        "--only-compustat-supplychain",
        action="store_true",
        help=(
            "Keep only likely Compustat supplier-customer/supply-chain tables based on "
            "table name and description keywords."
        ),
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")

    if not username:
        print("Missing WRDS_USERNAME env var.")
        print("Set it in shell or .env, e.g. WRDS_USERNAME=your_netid")
        return 2

    print(f"Connecting to WRDS as '{username}'...")

    db = None
    try:
        db = wrds.Connection(wrds_username=username, wrds_password=password)

        libraries = db.list_libraries()
        compustat_candidates = sorted(
            lib for lib in libraries if ("comp" in lib.lower() or lib.lower().startswith("comp"))
        )
        sec_candidates = sorted(
            lib for lib in libraries if ("sec" in lib.lower() or "audit" in lib.lower())
        )

        if args.show_all_libraries:
            _print_table("All WRDS libraries", libraries, args.limit)
        else:
            matched_libraries = sorted(set(compustat_candidates + sec_candidates))
            _print_table("Matched Compustat/SEC libraries", matched_libraries, args.limit)

        target_libraries = sorted(set(compustat_candidates + sec_candidates))
        if not target_libraries:
            print("\nNo Compustat/SEC-like libraries found for this account.")
            return 0

        print("\nFetching table catalog (library/table_name/description)...")
        query = _table_catalog_query(target_libraries)
        catalog_df = db.raw_sql(query)

        if args.only_compustat_supplychain:
            comp_df = catalog_df.loc[catalog_df["library"].str.lower().str.contains("comp")].copy()
            filtered_df = _filter_supplychain_rows(comp_df)
            print(
                f"Compustat supplier-customer filter: {len(filtered_df)} "
                f"tables matched from {len(comp_df)} Compustat tables."
            )
            catalog_df = filtered_df

        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_df.to_csv(output_path, index=False)

        print(f"\nSaved table catalog to: {output_path}")

        if catalog_df.empty:
            print("No rows matched the selected filter.")
        else:
            preview = (
                catalog_df[["library", "table_name", "description"]]
                .fillna("")
                .head(args.limit)
                .to_dict(orient="records")
            )
            _print_table(
                "Preview table names in CSV",
                [f"{row['library']}.{row['table_name']}" for row in preview],
                args.limit,
            )

        print("\nWRDS access check completed.")
        return 0

    except Exception as exc:
        print(f"WRDS connection failed: {exc}")
        return 1

    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
