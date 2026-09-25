"""JSON Schema -> Lark frontend shared by the PSC and GreatGramma adapters.

Neither engine accepts JSON Schema natively; both take LALR(1) Lark grammars.
We reuse the frontend published with PSC (github.com/Gompyn/PSC,
``benchmark/src/json_schema_to_lark.py``), which is the protocol PSC used to
compare itself against GreatGramma on MaskBench: the MaskBench llama.cpp GBNF
converter, llguidance's ``gbnf_to_lark``, lexical-rule priorities and a
``SPACE`` terminal. PSC ships no license, so it is loaded from a checkout
(``PSC_ROOT``) rather than vendored here.
"""
import hashlib
import importlib
import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

_PKG = "_psc_benchmark_src"
_FRONTEND_FILES = ("json_schema_to_lark.py", "json_schema_to_grammar.py", "json_schema.py")


class AdapterError(ValueError):
    """Failure attributed to one pipeline stage, recorded by the runner."""

    def __init__(self, stage, message, outcome="error"):
        super().__init__(message)
        self.stage = stage
        self.outcome = outcome


def psc_root() -> Path:
    env = os.environ.get("PSC_ROOT")
    if env:
        return Path(env)
    # Default: PSC cloned next to the InstaGrammar checkout.
    return Path(__file__).resolve().parents[4] / "PSC"


def _src_dir() -> Path:
    src = psc_root() / "benchmark" / "src"
    if not (src / "json_schema_to_lark.py").is_file():
        raise AdapterError(
            "dependency", f"PSC frontend not found under {src}; set PSC_ROOT"
        )
    return src


def frontend_identity() -> dict:
    """Source hashes so results can be tied to an exact frontend."""
    src = _src_dir()
    return {
        "source": "Gompyn/PSC benchmark/src/json_schema_to_lark.py",
        "sha256": {
            name: hashlib.sha256((src / name).read_bytes()).hexdigest()
            for name in _FRONTEND_FILES
        },
    }


def _load():
    if _PKG + ".json_schema_to_lark" in sys.modules:
        return sys.modules[_PKG + ".json_schema_to_lark"]
    src = _src_dir()
    # Load PSC's benchmark/src as a uniquely named package so its relative
    # imports resolve without shadowing any other top-level ``src``.
    spec = importlib.util.spec_from_file_location(
        _PKG, src / "__init__.py", submodule_search_locations=[str(src)]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_PKG] = pkg
    spec.loader.exec_module(pkg)
    try:
        return importlib.import_module(_PKG + ".json_schema_to_lark")
    except Exception as exc:
        raise AdapterError("dependency", repr(exc)) from exc


def json_schema_to_lark(schema: dict) -> str:
    module = _load()
    try:
        return module.json_schema_to_lark(deepcopy(schema), llguidance_variant=False)
    except Exception as exc:
        raise AdapterError("schema_conversion", repr(exc), "unsupported") from exc
