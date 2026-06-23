"""
scripts/init_database.py

Initialises the Supabase PostgreSQL database by running
the full schema from config/supabase_schema.sql.

Requires:
  - SUPABASE_URL and SUPABASE_SERVICE_KEY set in .env
  - psycopg2-binary installed
  - DATABASE_URL set (postgres connection string)

Usage:
    python scripts/init_database.py
    python scripts/init_database.py --dry-run   # prints SQL only
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from config.settings import settings


SCHEMA_FILE = Path(__file__).parent.parent / "config" / "supabase_schema.sql"


def read_schema() -> str:
    if not SCHEMA_FILE.exists():
        print(f"❌ Schema file not found: {SCHEMA_FILE}")
        sys.exit(1)
    return SCHEMA_FILE.read_text(encoding="utf-8")


def run_via_psycopg2(sql: str) -> None:
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2-binary not installed.")
        print("   pip install psycopg2-binary")
        sys.exit(1)

    if not settings.database_url:
        print("❌ DATABASE_URL not set in .env")
        sys.exit(1)

    print(f"Connecting to database...")
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    cur  = conn.cursor()

    # Split by semicolons and run each statement
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    ok = 0
    failed = 0
    for stmt in statements:
        try:
            cur.execute(stmt)
            ok += 1
        except Exception as e:
            # Many are IF NOT EXISTS — log but continue
            print(f"  ⚠️  {str(e)[:80]}")
            failed += 1
            conn.rollback()

    cur.close()
    conn.close()
    print(f"\n✅ Schema applied: {ok} ok, {failed} skipped/failed")


def run_via_supabase(sql: str) -> None:
    """Alternative: run via supabase-py (requires service key)."""
    try:
        from supabase import create_client
    except ImportError:
        print("❌ supabase-py not installed.")
        sys.exit(1)

    if not settings.supabase_url or not settings.supabase_service_key:
        print("❌ SUPABASE_URL / SUPABASE_SERVICE_KEY not set in .env")
        sys.exit(1)

    print("Note: For full schema (extensions, enums), use psycopg2 or the Supabase SQL editor.")
    print("Supabase client path only runs basic table creation.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Print SQL only, do not execute")
    p.add_argument("--method",  choices=["psycopg2","print"], default="psycopg2")
    args = p.parse_args()

    sql = read_schema()

    if args.dry_run or args.method == "print":
        print("=" * 60)
        print("SCHEMA SQL (dry run — not executed)")
        print("=" * 60)
        print(sql[:3000], "...[truncated]")
        print("\nTo apply: copy-paste the full content of config/supabase_schema.sql")
        print("into the Supabase dashboard → SQL Editor → Run")
        return

    run_via_psycopg2(sql)
    print("\nNext steps:")
    print("  python scripts/generate_sample_data.py")
    print("  python scripts/train_models.py")
    print("  python scripts/start_api.py")


if __name__ == "__main__":
    main()
