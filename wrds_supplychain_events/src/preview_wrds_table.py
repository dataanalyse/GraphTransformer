#!/usr/bin/env python3
"""Preview a WRDS table: columns, row count, and sample rows.

Examples:
  python wrds_supplychain_events/src/preview_wrds_table.py --library compseg --table seg_customer
  python wrds_supplychain_events/src/preview_wrds_table.py --library comp --table wrds_seg_customer --limit 50
"""

from __future__ import annotations

import argparse
import os
import sys

import wrds
from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview metadata and sample rows from a WRDS table.")
    parser.add_argument("--library", required=True, help="WRDS library/schema name (e.g., compseg)")
    parser.add_argument("--table", required=True, help="WRDS table name (e.g., seg_customer)")
    parser.add_argument("--limit", type=int, default=20, help="Number of sample rows (default: 20)")
    args = parser.parse_args()

    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")

    if not username:
        print("Missing WRDS_USERNAME env var.")
        return 2

    db = None
    try:
        print(f"Connecting to WRDS as '{username}'...")
        db = wrds.Connection(wrds_username=username, wrds_password=password)

        cols = db.raw_sql(
            f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{args.library}'
              AND table_name = '{args.table}'
            ORDER BY ordinal_position
            """
        )

        if cols.empty:
            print(f"No columns found for {args.library}.{args.table}. Check library/table name.")
            return 1

        count_df = db.raw_sql(f"SELECT COUNT(*) AS n_rows FROM {args.library}.{args.table}")
        sample_df = db.raw_sql(f"SELECT * FROM {args.library}.{args.table} LIMIT {args.limit}")

        print(f"\n=== TABLE: {args.library}.{args.table} ===")
        print("\n=== COLUMNS ===")
        print(cols.to_string(index=False))
        print("\n=== ROW COUNT ===")
        print(count_df.to_string(index=False))
        print(f"\n=== SAMPLE ({args.limit} rows) ===")
        print(sample_df.to_string(index=False))
        return 0

    except Exception as exc:
        print(f"Failed to preview table {args.library}.{args.table}: {exc}")
        return 1

    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
