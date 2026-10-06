"""Run all repository tests and persist their actual status and numeric errors."""
import argparse
import io
from pathlib import Path
import sys
import unittest

from src.data import ROOT
from src.run_svdpp import atomic_json, now, source_hashes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", default="results/svdpp")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    text = stream.getvalue()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "checks.log").write_text(text, encoding="utf-8")
    diagnostic = getattr(sys.modules.get("test_svdpp"), "DIAGNOSTICS", {})
    svdpp_skipped = sum(test.__class__.__module__ == "test_svdpp" for test, _ in result.skipped)
    svdpp_expected_failures = sum(test.__class__.__module__ == "test_svdpp" for test, _ in result.expectedFailures)
    command = "python -m src.check_svdpp"
    if args.output_dir != "results/svdpp":
        command += f" --output-dir {args.output_dir}"
    report = dict(command=command, checked_at=now(),
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), successful=result.wasSuccessful(),
                  expected_failures=len(result.expectedFailures),
                  svdpp_skipped=svdpp_skipped, svdpp_expected_failures=svdpp_expected_failures,
                  numerical_checks=diagnostic, source_sha256=source_hashes())
    atomic_json(output / "checks.json", report)
    print(text)
    if not result.wasSuccessful() or svdpp_skipped or svdpp_expected_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
