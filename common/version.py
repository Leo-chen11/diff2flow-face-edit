"""Which code a run or an eval was produced with.

The project is often run from an rsync'ed copy without .git, so the commit is
read from git when there is one, else from a VERSION file at the project root
(written at sync time: git -C <clone> rev-parse --short HEAD > <clone>/VERSION).
"""
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def code_version():
    try:
        out = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--untracked-files=no'],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            return out.stdout.strip() + ('+dirty' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with open(os.path.join(ROOT, 'VERSION')) as f:
            return f.read().strip() or 'unknown'
    except OSError:
        return 'unknown'
