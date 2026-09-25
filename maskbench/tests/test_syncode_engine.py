"""CPU-only adapter contract checks; no downloads or language model required."""
import json
import sys
from pathlib import Path

import pytest
import torch
from lark import Lark

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maskbench.syncode_engine import SyncodeEngine, gbnf_to_lark


def test_translation_preserves_named_primitives_and_whitespace():
    parser = Lark(gbnf_to_lark('root ::= string space\nstring ::= "x"\nspace ::= " "?'))
    parser.parse('x ')
    with pytest.raises(Exception):
        parser.parse('"arbitrary"')
    with pytest.raises(Exception):
        parser.parse(' x')


@pytest.mark.parametrize('body,valid,invalid', [
    ('[a-z]{0,2}', 'ab', 'abc'),
    ('"a"{2,}', 'aaa', 'a'),
    ('("a" "b"){1,2}', 'abab', 'ababab'),
])
def test_translation_quantifiers(body, valid, invalid):
    parser = Lark(gbnf_to_lark('root ::= ' + body))
    parser.parse(valid)
    with pytest.raises(Exception):
        parser.parse(invalid)


@pytest.mark.parametrize('grammar', ['root ::= [abc', 'root ::= "oops', 'root ::= x\nx ::= "a"\nx ::= "b"', 'root ::= "x"\n%ignore " "'])
def test_translation_rejects_malformed_or_unrepresentable_input(grammar):
    with pytest.raises(ValueError):
        gbnf_to_lark(grammar)


def test_rejected_token_does_not_advance():
    engine = SyncodeEngine()
    engine._accept_mask = torch.tensor([True, False])
    engine._input_ids = []
    assert engine.commit_token(1) is False
    assert engine._input_ids == []


def test_empty_mask_is_not_unconstrained():
    engine = SyncodeEngine()
    engine._parse_result = object()
    class EmptyStore:
        def get_accept_mask(self, result):
            return torch.tensor([False, False])
    engine._mask_store = EmptyStore()
    engine.compute_mask()
    assert not engine._accept_mask.any()
    assert engine.commit_token(0) is False


def test_unsupported_schema_has_distinct_outcome():
    engine = SyncodeEngine()
    with pytest.raises(Exception) as exc:
        engine.compile_grammar({'type': 'array', 'uniqueItems': True})
    assert exc.value.stage == 'schema_conversion'
    assert exc.value.outcome == 'unsupported'


@pytest.mark.parametrize('schema', [
    {'type': 'array', 'minItems': 2},
    {'type': 'object', 'required': ['missing']},
    {'type': 'integer', 'minimum': 1, 'exclusiveMinimum': 4},
    {'properties': {'a': {'type': 'string'}}},
    {'type': ['object', 'null'], 'required': ['missing']},
])
def test_converter_constraints_are_not_silently_discarded(schema):
    with pytest.raises(Exception) as exc:
        SyncodeEngine().compile_grammar(schema)
    assert exc.value.outcome == 'unsupported'


def test_runner_persists_unsupported_and_denominator(tmp_path, monkeypatch):
    from maskbench import runner
    monkeypatch.setattr(runner, 'output_path', str(tmp_path), raising=False)
    source = tmp_path / 'input' / 'fixture.json'
    source.parent.mkdir()
    source.write_text(json.dumps({'schema': {'type': 'array', 'uniqueItems': True},
                                  'tests': [{'data': [], 'valid': True}]}))
    result = runner.process_file(SyncodeEngine(), str(source))
    assert result['outcome'] == 'unsupported'
    assert result['error_stage'] == 'schema_conversion'
    assert result['num_tests'] == 1
    assert json.loads((tmp_path / source.name).read_text()) == result


@pytest.fixture
def real_engine():
    from tokenizers import Tokenizer, models, decoders, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    # Byte-level vocabulary with a deliberate cross-terminal token and space.
    vocabulary = ['<eos>', 'a', 'b', 'ab', 'x', 'Ġ', 't', 'r', 'u', 'e', 'f', 'l', 's', '1', '2', '12']
    backend = Tokenizer(models.WordLevel({v: i for i, v in enumerate(vocabulary)}, unk_token='<eos>'))
    backend.decoder = decoders.ByteLevel()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    engine = SyncodeEngine()
    engine.multi = True
    engine.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>')
    return engine


def test_real_syncode_cross_terminal_mask_is_not_overridden(real_engine):
    engine = real_engine
    engine.compile_lark('start: "a" "b"')
    engine.reset()
    engine.compute_mask()
    assert not engine.commit_token(engine.tokenizer.eos_token_id)
    assert not engine.commit_token(engine.tokenizer.convert_tokens_to_ids('x'))
    # 0.4.0 strict masks reject this valid fused token. Preserve that native
    # result instead of overriding it with an is_valid candidate parse.
    engine._parser.base_parser.parse('ab')
    fused = engine.tokenizer.convert_tokens_to_ids('ab')
    native = engine._mask_store.get_accept_mask(engine._parse_result)[fused].item()
    assert engine.commit_token(fused) is native
    if not native:
        for char in 'ab':
            engine.compute_mask()
            assert engine.commit_token(engine.tokenizer.convert_tokens_to_ids(char))
    engine.compute_mask()
    assert engine.commit_token(engine.tokenizer.eos_token_id)


def test_real_syncode_reset_and_schema_compile(real_engine):
    engine = real_engine
    engine.compile_grammar({'type': 'boolean'})
    for value in ('true', 'false'):
        engine.reset()
        for char in value:
            engine.compute_mask()
            assert engine.commit_token(engine.tokenizer.convert_tokens_to_ids(char))
        engine.compute_mask()
        assert engine.commit_token(engine.tokenizer.eos_token_id)


def test_real_syncode_malformed_grammar_is_parser_error(real_engine):
    with pytest.raises(Exception) as exc:
        real_engine.compile_lark('start: MISSING_TERMINAL')
    assert exc.value.stage == 'grammar_parser'


def test_real_syncode_parser_failure_propagates(real_engine, monkeypatch):
    engine = real_engine
    engine.compile_lark('start: "a" "b"')
    engine.reset()
    engine.compute_mask()
    def fail(text):
        raise RuntimeError('parser failed')
    monkeypatch.setattr(engine._parser, 'get_acceptable_next_terminals', fail)
    with pytest.raises(Exception) as exc:
        engine.commit_token(engine.tokenizer.convert_tokens_to_ids('a'))
    assert exc.value.stage == 'advance'


def test_real_syncode_complete_and_incomplete_prefix(real_engine):
    engine = real_engine
    engine.compile_lark('start: "a" "b"')
    engine.reset()
    engine.compute_mask()
    assert engine.commit_token(engine.tokenizer.convert_tokens_to_ids('a'))
    assert not engine.check_complete()
    engine.compute_mask()
    assert engine.commit_token(engine.tokenizer.convert_tokens_to_ids('b'))
    assert engine.check_complete()


def test_runner_keeps_errors_separate_from_correct_rejections(tmp_path, monkeypatch, real_engine):
    from maskbench import runner
    monkeypatch.setattr(runner, 'output_path', str(tmp_path), raising=False)
    source = tmp_path / 'input' / 'fixture.json'
    source.parent.mkdir()
    source.write_text(json.dumps({'schema': {'type': 'boolean'},
                                  'tests': [{'data': True, 'valid': True},
                                            {'data': 'bad', 'valid': False}]}))
    def fail():
        raise RuntimeError('reset failed')
    monkeypatch.setattr(real_engine, 'reset', fail)
    result = runner.process_file(real_engine, str(source))
    assert result['outcome'] == 'error'
    assert result['num_tests'] == 2
    assert result['num_invalid_tests'] == 0
    assert len(result['test_results']) == 2
    assert all(row['accepted'] is None for row in result['test_results'])


def test_runner_checks_eos_for_prefix_valid_json(tmp_path, monkeypatch, real_engine):
    from maskbench import runner
    monkeypatch.setattr(runner, 'output_path', str(tmp_path), raising=False)
    source = tmp_path / 'input' / 'fixture.json'
    source.parent.mkdir()
    source.write_text(json.dumps({'schema': {'const': 12},
                                  'tests': [{'data': 1, 'valid': False},
                                            {'data': 12, 'valid': True}]}))
    result = runner.process_file(real_engine, str(source))
    assert result['outcome'] == 'supported'
    assert result['num_valid_tests'] == result['num_invalid_tests'] == 1
    assert [row['accepted'] for row in result['test_results']] == [False, True]
    assert all(row['eof_checked'] for row in result['test_results'])
    assert len(result['all_mask_us']) == result['num_tokens'] == 2
