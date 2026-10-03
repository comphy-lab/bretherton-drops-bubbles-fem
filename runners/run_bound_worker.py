"""Invoke one checksum-bound worker from a private run configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.configuration.read_text())
    root = args.configuration.resolve().parent
    if args.case.resolve() != Path(config['case']).resolve():
        raise ValueError('case selector differs from configuration')
    worker = (root / config['entrypoint']).resolve(strict=True)
    if not worker.is_relative_to(root) or not worker.is_file():
        raise ValueError('entry point must be a regular file inside the run root')
    expected = config['files'].get(str(worker.relative_to(root)))
    if expected is None or digest(worker) != expected:
        raise ValueError('entry point checksum differs from configuration')
    argv = [sys.executable, str(worker), '--case', str(args.case),
            '--configuration', str(args.configuration), '--output', str(args.output)]
    if args.preflight_only:
        argv.append('--preflight-only')
    return subprocess.run(argv, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
