"""CPU-only adapter checks for PSC and GreatGramma; no model downloads.

Contract tests use stand-in backend states. Tests marked ``real`` run the
installed backends on a 16-token byte-level fixture vocabulary and are skipped
when ``fast_constraint`` / ``alignment`` or the PSC checkout are unavailable.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maskbench import lark_frontend
from maskbench.greatgramma_engine import GreatGrammaEngine
from maskbench.lark_frontend import AdapterError
from maskbench.psc_engine import PscEngine

ENGINES = [PscEngine, GreatGrammaEngine]
HAVE = {
    PscEngine: importlib.util.find_spec("fast_constraint") is not None,
    GreatGrammaEngine: importlib.util.find_spec("alignment") is not None,
}
HAVE_FRONTEND = (lark_frontend.psc_root() / "benchmark/src/json_schema_to_lark.py").is_file()


class _PscState:
    def __init__(self, mask, fail=False):
        self.mask = mask
        self.fail = fail
        self.committed = []

    def commit_token(self, t):
        if self.fail:
            raise RuntimeError("parser failed")
        self.committed.append(t)
        return _PscState(self.mask)


def test_psc_rejected_token_does_not_advance():
    engine = PscEngine()
    state = _PscState([1, 0, 1])
    engine._inst = state
    assert engine.commit_token(1) is False
    assert engine.commit_token(7) is False
    assert engine._inst is state and state.committed == []
    assert engine.commit_token(2) is True
    assert state.committed == [2]


def test_psc_admitted_token_failure_is_advance_error():
    engine = PscEngine()
    engine._inst = _PscState([1], fail=True)
    with pytest.raises(AdapterError) as exc:
        engine.commit_token(0)
    assert exc.value.stage == "advance"


class _GgState:
    def __init__(self, acceptance, fail=False):
        self._acceptance = acceptance
        self.fail = fail

    @property
    def acceptance(self):
        return self._acceptance

    def feed_token(self, t):
        if self.fail:
            raise RuntimeError("parser failed")
        return _GgState({})


def _gg(acceptance, fail=False):
    engine = GreatGrammaEngine()
    engine._vocab_size = 4
    engine._state = _GgState(acceptance, fail)
    return engine


def test_greatgramma_mask_matches_acceptance_and_empty_stays_empty():
    engine = _gg({1: [0], 3: [0]})
    engine.compute_mask()
    assert engine._mask.tolist() == [False, True, False, True]
    assert engine.commit_token(0) is False
    engine = _gg({})
    engine.compute_mask()
    assert not engine._mask.any()
    assert engine.commit_token(0) is False


def test_greatgramma_requires_fresh_mask_and_propagates_advance_errors():
    engine = _gg({1: [0]})
    with pytest.raises(AdapterError):
        engine.commit_token(1)
    engine = _gg({1: [0]}, fail=True)
    engine.compute_mask()
    with pytest.raises(AdapterError) as exc:
        engine.commit_token(1)
    assert exc.value.stage == "advance"


@pytest.mark.parametrize("cls", ENGINES)
def test_missing_frontend_is_recorded_as_dependency_row(cls, tmp_path, monkeypatch):
    from maskbench import runner
    monkeypatch.setenv("PSC_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr(runner, "output_path", str(tmp_path), raising=False)
    source = tmp_path / "input" / "fixture.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"schema": {"type": "boolean"},
                                  "tests": [{"data": True, "valid": True}]}))
    engine = cls()
    engine.multi = True
    engine.init()
    result = runner.process_file(engine, str(source))
    assert result["error_stage"] == "dependency"
    assert result["num_tests"] == 1


# ---------------------------------------------------------------- real backends

@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocabulary = ['<eos>', 'a', 'b', 'ab', 'x', 'Ġ', 't', 'r', 'u', 'e', 'f', 'l', 's', '1', '2', '12']
    backend = Tokenizer(models.WordLevel({v: i for i, v in enumerate(vocabulary)}, unk_token='<eos>'))
    backend.decoder = decoders.ByteLevel()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>')


def _real(cls, tokenizer):
    if not HAVE[cls]:
        pytest.skip(f"{cls.__name__} backend not installed")
    engine = cls()
    engine.multi = True
    engine.tokenizer = tokenizer
    engine.init()
    return engine


def _replay(engine, text):
    engine.reset()
    for char in text:
        engine.compute_mask()
        if not engine.commit_token(engine.tokenizer.convert_tokens_to_ids(char)):
            return False
    return engine.check_complete()


@pytest.mark.parametrize("cls", ENGINES)
def test_real_lark_grammar_mask_advance_and_eos(cls, tokenizer):
    engine = _real(cls, tokenizer)
    engine.compile_lark('start: A B\nA: "a"\nB: "b"')
    engine.reset()
    engine.compute_mask()
    assert not engine.commit_token(tokenizer.convert_tokens_to_ids('x'))
    engine.compute_mask()
    assert engine.commit_token(tokenizer.convert_tokens_to_ids('a'))
    assert not engine.check_complete()
    engine.compute_mask()
    assert engine.commit_token(tokenizer.convert_tokens_to_ids('b'))
    assert engine.check_complete()
    # Reset must not carry state from the previous replay.
    assert _replay(engine, 'ab')
    engine.reset()
    assert not engine.check_complete()


@pytest.mark.parametrize("cls", ENGINES)
def test_real_malformed_grammar_is_parser_error(cls, tokenizer):
    engine = _real(cls, tokenizer)
    with pytest.raises(AdapterError) as exc:
        engine.compile_lark('start: MISSING_TERMINAL')
    assert exc.value.stage == "grammar_parser"


@pytest.mark.parametrize("cls", ENGINES)
def test_real_schema_through_runner_checks_eos(cls, tokenizer, tmp_path, monkeypatch):
    if not HAVE_FRONTEND:
        pytest.skip("PSC checkout (frontend) not found; set PSC_ROOT")
    from maskbench import runner
    engine = _real(cls, tokenizer)
    monkeypatch.setattr(runner, "output_path", str(tmp_path), raising=False)
    source = tmp_path / "input" / "fixture.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"schema": {"const": 12},
                                  "tests": [{"data": 1, "valid": False},
                                            {"data": 12, "valid": True}]}))
    result = runner.process_file(engine, str(source))
    assert result["outcome"] == "supported", result
    assert [row["accepted"] for row in result["test_results"]] == [False, True]
    assert "frontend_sha256" in result["adapter"]
