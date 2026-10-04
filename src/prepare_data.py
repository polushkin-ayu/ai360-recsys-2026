"""Download, prepare, then verify the complete Team 1 data contract."""

import argparse
import json

from .check_data import make_report
from .data import ROOT, load_config, prepare, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/team1-data.json")
    parser.add_argument("--report", default="results/team1-data-checks.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_config(config_path)
    output, _ = prepare(config, config_path)
    command = f"python -m src.prepare_data --config {args.config} --report {args.report}"
    report = make_report(output, config, command, config_path)
    write_json(ROOT / args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
