"""Run all repository tests and persist their actual status and numeric errors."""
import io
from pathlib import Path
import sys
import unittest

from src.data import ROOT
from src.run_svdpp import atomic_json, now, source_hashes


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    text = stream.getvalue()
    output = ROOT / "results/svdpp"
    output.mkdir(parents=True, exist_ok=True)
    (output / "checks.log").write_text(text, encoding="utf-8")
    diagnostic = getattr(sys.modules.get("test_svdpp"), "DIAGNOSTICS", {})
    report = dict(command="python -m src.check_svdpp", checked_at=now(),
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), successful=result.wasSuccessful(),
                  numerical_checks=diagnostic, source_sha256=source_hashes())
    atomic_json(output / "checks.json", report)
    print(text)
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
