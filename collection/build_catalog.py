"""Build first-gym collection catalogs from Heatz policy directory."""

from __future__ import annotations

import argparse

from collection.catalog import write_catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatz-policy-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_catalog(args.heatz_policy_dir, args.output)
    print(f"Wrote catalog to {args.output}")


if __name__ == "__main__":
    main()
