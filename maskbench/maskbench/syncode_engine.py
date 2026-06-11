from .engine import Engine
from .json_schema import json_schema_to_gbnf
import re
import sys
import torch


def _import_syncode():
    """Import syncode submodules while bypassing the broken top-level __init__.

    syncode's ``__init__.py`` eagerly imports evaluation/dataset code
    that requires data files not shipped with the pip package.  We work
    around this by registering a stub package in ``sys.modules`` before
    importing only the submodules we need.
    """
    if "syncode.grammar_decoder" in sys.modules:
        return  # already imported

    import types, importlib

    if "syncode" not in sys.modules:
        pkg = types.ModuleType("syncode")
        import importlib.util
        spec = importlib.util.find_spec("syncode")
        if spec and spec.submodule_search_locations:
            pkg.__path__ = list(spec.submodule_search_locations)
        else:
            pkg.__path__ = []
        pkg.__package__ = "syncode"
        sys.modules["syncode"] = pkg

    importlib.import_module("syncode.common")
    importlib.import_module("syncode.parsers")
    importlib.import_module("syncode.parsers.grammars")
    importlib.import_module("syncode.mask_store")
    importlib.import_module("syncode.grammar_decoder")


def _tokenize_gbnf_body(body: str):
    """Tokenize a GBNF rule body into (kind, text) chunks.

    Single-pass left-to-right: at each position decide between a [...]
    char class, a "..." string, an identifier, or anything else.  Char
    classes can contain ``"`` which would otherwise confuse a string-
    aware split, so they are recognized first.
    """
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == '[':
            j = i + 1
            while j < n:
                if body[j] == '\\' and j + 1 < n:
                    j += 2; continue
                if body[j] == ']':
                    j += 1; break
                j += 1
            cls = body[i:j]
            if j < n and body[j] == '{':
                k = body.index('}', j) + 1
                yield ('class_q', cls, body[j:k]); i = k
            else:
                yield ('class', cls); i = j
        elif c == '"':
            j = i + 1
            while j < n:
                if body[j] == '\\' and j + 1 < n:
                    j += 2; continue
                if body[j] == '"':
                    j += 1; break
                j += 1
            yield ('str', body[i:j]); i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < n and (body[j].isalnum() or body[j] in '_-'):
                j += 1
            yield ('ident', body[i:j]); i = j
        else:
            yield ('other', c); i += 1


# Standard JSON primitives expressed as Lark terminals (regex) instead
# of recursive character-level rules.  Syncode's terminal-overapproximation
# mask only handles cross-terminal tokens (e.g. the LLaMA token ``{"``)
# correctly when strings are a single terminal.  The GBNF emitter builds
# these out of ``char``/``string``/``number`` rules that don't fold into
# terminals; we replace those rules wholesale with this preamble.
_JSON_PRIMITIVES_PREAMBLE = r'''
STRING: /"(?:[^"\\\x00-\x1f]|\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4}))*"/
NUMBER: /-?(?:0|[1-9][0-9]{0,15})(?:\.[0-9]{1,16})?(?:[eE][-+]?[0-9]{1,16})?/
BOOLEAN: "true" | "false"
NULL: "null"
%ignore /[ \t\n\r]+/
?value: object | array | STRING | NUMBER | BOOLEAN | NULL
array: "[" [value ("," value)*] "]"
object: "{" [_pair ("," _pair)*] "}"
_pair: STRING ":" value
'''.strip()

# Rule names emitted by ``json_schema_to_gbnf`` that we replace with the
# preamble above.  Any GBNF rule with one of these names is dropped from
# the output; references to them are remapped to the preamble's
# definitions (``space`` is replaced by Lark's ``%ignore`` whitespace).
_PRIMITIVE_RULES = {
    'array', 'boolean', 'char', 'decimal-part', 'integral-part',
    'null', 'number', 'object', 'space', 'string', 'value',
}


def gbnf_to_lark(gbnf: str) -> str:
    """Translate llama.cpp-style GBNF to a Lark grammar string.

    Schema-specific rules are translated mechanically (``::=`` → ``:``,
    kebab-case → snake_case, char classes wrapped in ``/.../``).  The
    universal JSON-primitive rules emitted by ``json_schema_to_gbnf``
    (string, number, char, object, array, …) are dropped and replaced
    by a regex-terminal preamble so Syncode's lexer can handle tokens
    like ``{"`` that span a literal-and-string boundary.
    """
    rule_names = []
    for line in gbnf.splitlines():
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_-]*)\s*::=', line)
        if m:
            rule_names.append(m.group(1))
    rename = {n: n.lower().replace('-', '_') for n in rule_names}

    out = []
    for line in gbnf.splitlines():
        if not line.strip():
            out.append(line); continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_-]*)\s*::=\s*(.*)$', line)
        if not m:
            out.append(line); continue
        if m.group(1) in _PRIMITIVE_RULES:
            continue  # preamble defines this; skip the GBNF version
        head = rename[m.group(1)]
        parts = []
        for chunk in _tokenize_gbnf_body(m.group(2)):
            kind = chunk[0]
            if kind == 'class':
                parts.append('/' + chunk[1] + '/')
            elif kind == 'class_q':
                # Translate {m,n} into Lark's term-level ~m..n so zero-min
                # quantifiers don't produce a zero-width regex terminal,
                # which Lark's lexer rejects.
                cls, q = chunk[1], chunk[2]
                qm = re.match(r'\{(\d+)(?:,(\d*))?\}', q)
                lo = qm.group(1)
                hi = qm.group(2)
                if hi is None:
                    parts.append(f'/{cls}/ ~ {lo}')
                else:
                    parts.append(f'/{cls}/ ~ {lo}..{hi}')
            elif kind == 'ident':
                ref = chunk[1]
                if ref == 'space':
                    # Whitespace is `%ignore`d in the preamble; drop refs.
                    continue
                if ref in _PRIMITIVE_RULES:
                    # Map to the preamble's name (rule lowercase, terminals upper).
                    parts.append({'string': 'STRING', 'number': 'NUMBER',
                                  'boolean': 'BOOLEAN', 'null': 'NULL'}.get(
                                      ref, ref))
                else:
                    parts.append(rename.get(ref, ref))
            else:
                parts.append(chunk[1])
        out.append(f'{head}: {"".join(parts)}')
    # Lark requires a top-level rule named ``start``; GBNF uses ``root``.
    if 'root' in rename.values() and 'start' not in rename.values():
        out.insert(0, 'start: root')
    return _JSON_PRIMITIVES_PREAMBLE + '\n' + '\n'.join(out)


class SyncodeEngine(Engine):
    def __init__(self):
        super().__init__()
        self._input_ids = None
        self._accept_mask = None
        self._processor = None
        self._processor_factory = None

    def get_id(self):
        return "syncode"

    def get_name(self):
        return "Syncode"

    def get_module(self):
        return "syncode"

    def init(self):
        _import_syncode()
        from syncode.grammar_decoder import SyncodeLogitsProcessor
        from syncode.parsers.grammars import Grammar

        # Defer processor construction to compile_grammar so the LALR
        # tables and DFA mask cache are built per schema.
        def make_processor(grammar_obj):
            return SyncodeLogitsProcessor(
                grammar=grammar_obj,
                tokenizer=self.tokenizer,
                use_cache=True,
                parse_output_only=False,
                num_samples=1,
                parser="lalr",
                mode="grammar_mask",
            )

        self._Grammar = Grammar
        self._processor_factory = make_processor

    def compile_grammar(self, schema: dict):
        gbnf = json_schema_to_gbnf(schema)
        lark = gbnf_to_lark(gbnf)
        self.log_single(lark)
        grammar = self._Grammar(lark)
        self._processor = self._processor_factory(grammar)

    def reset(self):
        self._input_ids = torch.zeros((1, 0), dtype=torch.long)
        self._accept_mask = None
        self._processor.reset()

    def compute_mask(self):
        vocab_size = len(self.tokenizer)
        scores = torch.zeros((1, vocab_size), dtype=torch.float32)
        masked_scores = self._processor(self._input_ids, scores.clone())
        self._accept_mask = masked_scores[0] > -float("inf")

    def commit_token(self, t: int) -> bool:
        # Syncode's ``__call__`` mask is a fast over-approximation that
        # excludes valid cross-terminal tokens (e.g. ``{"`` at step 0).
        # Its actual generation pipeline gates on ``is_valid``, which
        # does a full incremental parse including the candidate token
        # and uses ``is_valid_prefix`` instead of the upfront mask.  We
        # mirror that behavior here so the maskbench accept/reject
        # decision matches what Syncode would actually do during
        # generation.
        next_token = torch.tensor([t], dtype=torch.long)
        ok = self._processor.is_valid(self._input_ids, next_token)
        self._input_ids = torch.cat(
            [self._input_ids, next_token.unsqueeze(0)], dim=1
        )
        return bool(ok)
