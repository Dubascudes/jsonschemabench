"""Runner behaviour for grammar workloads, EOS checks and checkpoints.

A stand-in engine exercises the runner contract without any backend; the
InstaGrammar tests run the real adapter end to end with a 257-token byte
tokenizer (the only InstaGrammar compile that is safe on a workstation) and
are skipped when the package or its C extension is unavailable.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maskbench import runner
from maskbench.engine import Engine, StageError


class ByteTokenizer:
    """Token id i < 256 is byte i; 256 is EOS."""
    eos_token_id = 256

    def __len__(self):
        return 257

    def token_bytes(self, token_id):
        return b"" if token_id == 256 else bytes([token_id])

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, **_):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class PrefixEngine(Engine):
    """Accepts any prefix of one of ``words``; complete iff a whole word."""
    grammar_format = "words"

    def __init__(self):
        super().__init__()
        self.tokenizer = ByteTokenizer()

    def get_id(self):
        return "prefix"

    def init(self):
        pass

    def compile_text_grammar(self, grammar):
        if grammar == "boom":
            raise StageError("grammar_parser", "bad grammar")
        self.words = grammar.split()

    def reset(self):
        self.buf = b""

    def compute_mask(self):
        pass

    def commit_token(self, t):
        nxt = self.buf + bytes([t])
        if any(w.encode().startswith(nxt) for w in self.words):
            self.buf = nxt
            return True
        return False

    def check_complete(self):
        return self.buf.decode() in self.words


def _run(tmp_path, engine, doc, name="w.json"):
    src = tmp_path / name
    src.write_text(json.dumps(doc))
    runner.output_path = str(tmp_path / "out")
    Path(runner.output_path).mkdir(exist_ok=True)
    engine.multi = True
    runner.process_file(engine, str(src))
    return json.loads((tmp_path / "out" / name).read_text())


def test_grammar_workload_eos_and_outcomes(tmp_path):
    doc = {"grammars": {"words": "select sel"}, "tests": [
        {"text": "select", "valid": True, "kind": "original"},
        {"text": "selec", "valid": False, "kind": "truncate"},   # valid prefix
        {"text": "selx", "valid": False, "kind": "drop_token"},
        {"text": "sel", "valid": True, "kind": "original"},
    ]}
    row = _run(tmp_path, PrefixEngine(), doc)
    assert row["complete"] and row["workload_kind"] == "grammar"
    res = row["test_results"]
    assert [r["accepted"] for r in res] == [True, False, False, True]
    assert res[1]["eof_checked"] and res[1]["rejected_at"] == 5      # at EOS
    assert res[2]["rejected_at"] == 3                                 # at 'x'
    assert row["num_valid_tests"] == 2 and row["num_invalid_tests"] == 2
    assert len(row["all_mask_us"]) == len(row["all_mask_cpu_us"]) == row["num_tokens"]
    assert "validation_error" not in row


def test_false_accept_is_reported(tmp_path):
    doc = {"grammars": {"words": "select"}, "tests": [
        {"text": "select", "valid": False, "kind": "dup_token"}]}
    row = _run(tmp_path, PrefixEngine(), doc)
    assert row["outcome"] == "validation_error"
    assert "should reject" in row["validation_error"]


def test_unsupported_format_and_compile_error(tmp_path):
    row = _run(tmp_path, PrefixEngine(), {"grammars": {"gbnf": "x"},
                                          "tests": [{"text": "x", "valid": True}]})
    assert row["outcome"] == "unsupported" and row["error_stage"] == "grammar_format"
    row = _run(tmp_path, PrefixEngine(), {"grammars": {"words": "boom"},
                                          "tests": [{"text": "x", "valid": True}]}, "b.json")
    assert row["error_stage"] == "grammar_parser" and row["complete"]


def test_checkpoint_written_mid_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHECKPOINT_S", -1.0)   # checkpoint after every test
    writes = []
    real = runner._write_json
    monkeypatch.setattr(runner, "_write_json",
                        lambda p, o: (writes.append(o.get("tests_done")), real(p, o)))
    doc = {"grammars": {"words": "ab"},
           "tests": [{"text": "ab", "valid": True}] * 3}
    _run(tmp_path, PrefixEngine(), doc)
    assert writes[:3] == [1, 2, 3]


def test_json_schema_rows_unchanged(tmp_path):
    class JsonEngine(PrefixEngine):
        def compile_grammar(self, schema):
            self.words = ['{"x": 1}']   # json.dumps default separators
    doc = {"schema": {"type": "object"},
           "tests": [{"data": {"x": 1}, "valid": True}, {"data": {"x": 2}, "valid": False}]}
    row = _run(tmp_path, JsonEngine(), doc)
    assert row["workload_kind"] == "json_schema"
    assert [r["accepted"] for r in row["test_results"]] == [True, False]


HAVE_IG = importlib.util.find_spec("instagrammar") is not None


def _ig_engine(monkeypatch, prewarm="capped"):
    from maskbench import instagrammar_lp_engine as mod
    monkeypatch.setenv("INSTAGRAMMAR_PREWARM", prewarm)
    try:
        eng = mod.InstaGrammarLPEngine()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    eng.tokenizer = ByteTokenizer()
    return eng


ARITH = r"""root ::= expr
expr ::= term (("+" | "-") term)*
term ::= [0-9]+ | "(" expr ")"
"""


@pytest.mark.skipif(not HAVE_IG, reason="instagrammar not importable")
def test_instagrammar_grammar_workload(tmp_path, monkeypatch):
    eng = _ig_engine(monkeypatch)
    doc = {"grammars": {"ig_ebnf": ARITH}, "tests": [
        {"text": "1+(2-3)", "valid": True},
        {"text": "1+(2-3", "valid": False},     # valid prefix: rejected at EOS
        {"text": "1+-", "valid": False},
        {"text": "42", "valid": True},
    ]}
    row = _run(tmp_path, eng, doc)
    assert row["outcome"] == "supported", row.get("validation_error")
    assert [r["accepted"] for r in row["test_results"]] == [True, False, False, True]
    bd = row["compile_breakdown"]
    assert {"compile_us", "template_us", "prewarm_us", "prewarm_states"} <= set(bd)
    path = row["replay_stats"]["mask_path"]
    assert len(path) == row["num_tokens"] and set(path) <= set("WCP")


@pytest.mark.skipif(not HAVE_IG, reason="instagrammar not importable")
def test_instagrammar_without_prewarm_is_lazy(tmp_path, monkeypatch):
    eng = _ig_engine(monkeypatch, prewarm="none")
    doc = {"grammars": {"ig_ebnf": ARITH},
           "tests": [{"text": "1+2+3", "valid": True}]}
    row = _run(tmp_path, eng, doc)
    assert row["compile_breakdown"]["prewarm_states"] == 0
    assert row["replay_stats"]["cold"] >= 1


@pytest.mark.skipif(not HAVE_IG, reason="instagrammar not importable")
def test_instagrammar_warm_replay_on_grammar_workload(tmp_path, monkeypatch):
    eng = _ig_engine(monkeypatch)
    monkeypatch.setattr(runner, "warm_replay", True)
    doc = {"grammars": {"ig_ebnf": ARITH}, "tests": [
        {"text": "1+(2-3)", "valid": True},
        {"text": "1+(2-3", "valid": False},     # rejected only at EOS
    ]}
    row = _run(tmp_path, eng, doc)
    assert row["outcome"] == "supported", row.get("validation_error")
    assert row["warm_outcome_mismatches"] == 0
    assert row["warm_num_tokens"] == row["num_tokens"]
    # the warm pass must not leak into the cold pass's per-token path
    assert len(row["replay_stats"]["mask_path"]) == row["num_tokens"]
    assert row["ig_prewarm_mode"] == "capped"
