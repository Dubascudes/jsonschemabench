"""SynCode mask/advance adapter for MaskBench's schema-derived GBNF profile.

This frontend is owned by the benchmark, not a native SynCode JSON Schema API.
It preserves the converter's grammar, and reports conversion/parser/DFA failures
separately. No logits-processor fallback or candidate-only validation is used.
"""
from copy import deepcopy
import importlib
import importlib.util
import re
import sys
import types

from .engine import Engine
from .json_schema import json_schema_to_gbnf


class SyncodeAdapterError(ValueError):
    def __init__(self, stage, message, outcome="error"):
        super().__init__(message)
        self.stage = stage
        self.outcome = outcome


def _import_syncode():
    # The 0.4.0 package initializer loads optional evaluation data absent from
    # its wheel. Import only the grammar APIs, with an accurate package spec.
    if "syncode" not in sys.modules:
        spec = importlib.util.find_spec("syncode")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("SynCode is not installed")
        pkg = types.ModuleType("syncode")
        pkg.__path__ = list(spec.submodule_search_locations)
        pkg.__package__ = "syncode"
        pkg.__spec__ = spec
        sys.modules["syncode"] = pkg
    importlib.import_module("syncode.parsers")
    importlib.import_module("syncode.mask_store.mask_store")


def gbnf_to_lark(gbnf: str) -> str:
    """Translate the converter's GBNF subset without changing its primitives.

    Unsupported syntax is an explicit frontend failure. In particular no
    generic JSON preamble, whitespace ignore, or grammar folding is inserted.
    """
    rules = {}
    for line in gbnf.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_-]*)\s*::=\s*(.*)', line)
        if not match or match[1] in rules:
            raise ValueError(f"Invalid or duplicate GBNF rule: {line!r}")
        rules[match[1]] = match[2]
    if 'root' not in rules:
        raise ValueError("GBNF requires a root rule")
    names = {name: f'r_{i}' for i, name in enumerate(rules)}

    def translate(body):
        pos = 0
        def sequence(in_group=False):
            nonlocal pos
            parts = []
            while pos < len(body):
                c = body[pos]
                if c.isspace():
                    pos += 1
                    continue
                if c == ')':
                    if not in_group:
                        raise ValueError("Unmatched closing parenthesis")
                    pos += 1
                    return ' '.join(parts)
                if c == '|':
                    parts.append('|')
                    pos += 1
                    continue
                if c == '(':
                    pos += 1
                    atom = '(' + sequence(True) + ')'
                elif c in '["':
                    start = pos
                    end = ']' if c == '[' else '"'
                    pos += 1
                    while pos < len(body):
                        if body[pos] == '\\':
                            pos += 2
                        elif body[pos] == end:
                            pos += 1
                            break
                        else:
                            pos += 1
                    else:
                        raise ValueError("Unterminated GBNF literal or character class")
                    atom = body[start:pos]
                    if c == '[':
                        atom = '/' + atom.replace('/', r'\/') + '/'
                else:
                    match = re.match(r'[A-Za-z_][A-Za-z0-9_-]*', body[pos:])
                    if not match or match[0] not in names:
                        raise ValueError(f"Unsupported GBNF syntax/reference: {body[pos:]!r}")
                    pos += len(match[0])
                    atom = names[match[0]]
                while pos < len(body) and body[pos].isspace():
                    pos += 1
                if pos < len(body) and body[pos] in '?*+':
                    atom += body[pos]
                    pos += 1
                elif pos < len(body) and body[pos] == '{':
                    match = re.match(r'\{(\d+)(?:,(\d*))?\}', body[pos:])
                    if not match:
                        raise ValueError("Invalid GBNF quantifier")
                    pos += len(match[0])
                    lo, hi = int(match[1]), match[2]
                    if hi == '':
                        atom = f'({atom} ~ {lo} {atom}*)' if lo else f'{atom}*'
                    elif hi is None:
                        atom = f'{atom} ~ {lo}'
                    else:
                        if int(hi) < lo:
                            raise ValueError("Reversed GBNF quantifier")
                        atom = f'{atom} ~ {lo}..{hi}'
                parts.append(atom)
            if in_group:
                raise ValueError("Unterminated GBNF group")
            return ' '.join(parts)
        return sequence()
    return '\n'.join(['start: ' + names['root']] +
                     [f'{names[name]}: {translate(body)}' for name, body in rules.items()])


# Deliberately bounded converter admission profile, not SynCode capabilities.
# The llama.cpp-derived converter has additional limitations (see status doc).
_ANNOTATIONS = {'$schema', '$id', 'title', 'description', 'default', 'examples',
                '$comment', 'deprecated', 'readOnly', 'writeOnly'}
_KEYWORDS = {'type', 'properties', 'required', 'additionalProperties', 'items',
             'minItems', 'maxItems', 'minLength', 'maxLength', 'minimum', 'maximum',
             'exclusiveMinimum', 'exclusiveMaximum', 'enum', 'const', '$defs',
             'definitions', '$ref', 'anyOf'}


def _check_schema(schema, path='$'):
    def unsupported(message):
        raise SyncodeAdapterError('schema_conversion', f'{path}: {message}', 'unsupported')
    if not isinstance(schema, dict):
        unsupported('boolean/non-object schemas are outside this converter profile')
    unknown = set(schema) - _ANNOTATIONS - _KEYWORDS
    if unknown:
        unsupported('unsupported converter keywords: ' + ', '.join(sorted(unknown)))
    # Converter branch precedence otherwise silently discards these siblings.
    constraints = set(schema) - _ANNOTATIONS - {'$defs', 'definitions'}
    if isinstance(schema.get('type'), list) and constraints - {'type'}:
        unsupported('type unions with sibling constraints are not admitted')
    for exclusive in ('$ref', 'anyOf', 'enum', 'const'):
        if exclusive in schema and constraints - {exclusive}:
            unsupported(f'{exclusive} with sibling constraints is not admitted')
    bounds = {'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum'} & set(schema)
    if bounds and schema.get('type') != 'integer':
        unsupported('numeric bounds require explicit integer type in this converter')
    for lower, exclusive in (('minimum', 'exclusiveMinimum'), ('maximum', 'exclusiveMaximum')):
        if lower in schema and exclusive in schema:
            unsupported('simultaneous inclusive/exclusive bounds are not admitted')
    if {'minItems', 'maxItems'} & set(schema):
        if schema.get('type') != 'array' or not isinstance(schema.get('items'), dict):
            unsupported('item bounds require array type and an items schema')
    if {'minLength', 'maxLength'} & set(schema) and schema.get('type') != 'string':
        unsupported('length bounds require explicit string type')
    if 'required' in schema and schema.get('type') == 'object':
        if not set(schema['required']) <= set(schema.get('properties', {})):
            unsupported('required keys must have declared properties')
    if 'type' not in schema and constraints & {'properties', 'required', 'additionalProperties', 'items'}:
        unsupported('structural constraints require an explicit type')
    for key in ('properties', '$defs', 'definitions'):
        for name, child in schema.get(key, {}).items():
            _check_schema(child, f'{path}/{key}/{name}')
    for key in ('items', 'additionalProperties'):
        if key in schema and isinstance(schema[key], dict):
            _check_schema(schema[key], f'{path}/{key}')
    if 'items' in schema and not isinstance(schema['items'], dict):
        unsupported('tuple/boolean items are outside this converter profile')
    for idx, child in enumerate(schema.get('anyOf', [])):
        _check_schema(child, f'{path}/anyOf/{idx}')


class SyncodeEngine(Engine):
    benchmark_contract = {
        'frontend': 'benchmark-llamacpp-gbnf-conservative-profile-v1',
        'mode': 'grammar_strict', 'parser': 'lalr', 'cache': 'disabled',
        'timed_operation': 'native-mask-plus-attempted-token-advance',
        'acceptance': 'complete replay; EOS check outside timed token samples',
    }

    def __init__(self):
        super().__init__()
        self._accept_mask = None
        self._parse_result = None
        self._input_ids = []
        self._finished = False

    def get_id(self):
        return 'syncode'

    def get_name(self):
        return 'SynCode (strict; benchmark GBNF frontend)'

    def get_module(self):
        return 'syncode'

    def init(self):
        # Defer dependency errors to compile_lark so the runner writes a row
        # for each schema instead of aborting before any output is created.
        pass

    def _load_runtime(self):
        try:
            _import_syncode()
        except Exception as exc:
            raise SyncodeAdapterError('dependency', str(exc)) from exc

    def compile_grammar(self, schema):
        self._accept_mask = None
        self._parse_result = None
        _check_schema(schema)
        try:
            gbnf = json_schema_to_gbnf(deepcopy(schema))
        except Exception as exc:
            raise SyncodeAdapterError('schema_conversion', str(exc)) from exc
        try:
            grammar_text = gbnf_to_lark(gbnf)
        except ValueError as exc:
            raise SyncodeAdapterError('grammar_translation', str(exc), 'unsupported') from exc
        self.compile_lark(grammar_text)

    # PSC's matching: the reference Lark grammar (SynCode's own format).
    grammar_format = "lark"

    def compile_text_grammar(self, grammar_text):
        self.compile_lark(grammar_text)

    def compile_lark(self, grammar_text):
        """Grammar-only entrypoint to isolate frontend and backend failures."""
        self._load_runtime()
        from syncode.parsers import create_base_parser
        from syncode.parsers.grammars import Grammar
        from syncode.parsers.incremental_parser import IncrementalParser
        from syncode.mask_store.mask_store import MaskStore
        self.log_single(grammar_text)
        try:
            grammar = Grammar(grammar_text)
            base = create_base_parser(grammar, parser='lalr')
            self._parser = IncrementalParser(base, ignore_whitespace=False)
        except Exception as exc:
            raise SyncodeAdapterError('grammar_parser', str(exc)) from exc
        try:
            # Direct constructor avoids upstream cache keys that only identify
            # tokenizers by class and vocabulary size. Compilation is cold.
            self._mask_store = MaskStore(
                base.terminals, self.tokenizer, simplifications=grammar.simplifications(),
                mode='grammar_strict', ignore_terminals=base.ignore_tokens,
                parse_table=base.parser.parser._parse_table,
            )
            self._byte_tokenizer = self._mask_store.byte_tokenizer
            if any(not isinstance(v, bytes) for v in self._byte_tokenizer.vocab_byte.values()):
                raise ValueError('SynCode byte tokenizer returned non-byte vocabulary entries')
        except Exception as exc:
            raise SyncodeAdapterError('mask_store', str(exc)) from exc

    def _parse(self):
        raw = self._byte_tokenizer.decode(self._input_ids, skip_special_tokens=True)
        try:
            text = raw.decode('utf-8')
            remainder = b''
        except UnicodeDecodeError as exc:
            if exc.reason != 'unexpected end of data':
                raise
            text, remainder = raw[:exc.start].decode('utf-8'), raw[exc.start:]
        result = self._parser.get_acceptable_next_terminals(text)
        result.remainder = result.remainder.encode('utf-8') + remainder
        if remainder:
            from syncode.parse_result import RemainderState
            result.remainder_state = RemainderState.INCOMPLETE
        return result

    def reset(self):
        self._input_ids = []
        self._finished = False
        self._accept_mask = None
        self._parser.reset()
        self._parse_result = self._parse()

    def compute_mask(self):
        try:
            self._accept_mask = self._mask_store.get_accept_mask(self._parse_result).clone()
        except Exception as exc:
            raise SyncodeAdapterError('mask', str(exc)) from exc

    def commit_token(self, token):
        if self._accept_mask is None:
            raise SyncodeAdapterError('advance', 'compute_mask must precede commit_token')
        if self._finished or token < 0 or token >= len(self._accept_mask) or not self._accept_mask[token]:
            return False
        self._input_ids.append(int(token))
        self._accept_mask = None
        if token == self.tokenizer.eos_token_id:
            self._finished = True
            return True
        try:
            self._parse_result = self._parse()
        except Exception as exc:
            # A mask-admitted token that cannot advance is a backend error,
            # not a valid rejection and never an unconstrained continuation.
            raise SyncodeAdapterError('advance', str(exc)) from exc
        return True

    def check_complete(self):
        """Check native EOS eligibility without consuming a speculative token."""
        self.compute_mask()
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise SyncodeAdapterError('completion', 'tokenizer has no EOS token')
        return 0 <= eos < len(self._accept_mask) and bool(self._accept_mask[eos])
