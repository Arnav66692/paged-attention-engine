"""Marks `tests` as a real package.

Not cosmetic. Without this file `tests` is a PEP 420 namespace package, and
namespace packages are the import system's LAST resort: if any entry on sys.path
provides a regular `tests` package or a `tests.py`, that wins outright, no matter
that ours is first on the path. On this laptop nothing else provides one, so
`from tests.tiny_model import ...` resolved and the suite passed. On Kaggle's
image something does, and Session 0's first run died with

    ModuleNotFoundError: No module named 'tests.tiny_model'

in collection, taking the entire suite with it -- 19 deselected, 0 executed, and
no information at all about the port it was run to verify.

An empty __init__.py makes `tests` a regular package, which the import system
resolves from the first sys.path entry that has it and stops. Deterministic on
any host, whatever else is installed.
"""
