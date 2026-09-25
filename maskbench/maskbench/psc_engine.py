"""PSC (Parser Stack Classification, Li et al. 2026) MaskBench adapter.

Uses the reference implementation from github.com/Gompyn/PSC
(``fast_constraint`` package) with the PSC benchmark's JSON Schema -> Lark
frontend. Preprocessing is cold: PSC's on-disk runner cache is not used, so
compile/load includes the full stack-classifier construction.

PSC materializes the complete mask when a parser state is created. Hence
``compute_mask`` is a no-op and each timed ``commit_token`` checks the
current mask, advances the lexer/parser and builds the next mask (the same
single-call pattern as the InstaGrammar adapter). The initial mask is built
in ``reset``, outside the timed samples.
"""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from .engine import Engine
from .lark_frontend import AdapterError, frontend_identity, json_schema_to_lark


class PscEngine(Engine):
    benchmark_contract = {
        "frontend": "psc-json-schema-to-lark",
        "parser": "lalr+basic-lexer",
        "cache": "disabled",
        "timed_operation": "mask-membership-plus-advance-plus-next-mask",
        "acceptance": "complete replay; EOS check outside timed token samples",
    }

    def __init__(self):
        super().__init__()
        self._runner = None
        self._inst = None

    def get_id(self):
        return "psc"

    def get_name(self):
        return "PSC"

    def get_module(self):
        return "fast-constraint"

    def init(self):
        # Dependency errors surface in compile_grammar so the runner still
        # writes one row per schema. Hashing here keeps it out of TTFM.
        try:
            self.benchmark_contract = dict(
                self.benchmark_contract,
                frontend_sha256=frontend_identity()["sha256"],
            )
        except AdapterError:
            pass

    def compile_grammar(self, schema: dict):
        self._runner = None
        self._inst = None
        grammar = json_schema_to_lark(schema)
        self.compile_lark(grammar)

    # PSC's matching: the reference Lark grammar.
    grammar_format = "lark"

    def compile_text_grammar(self, grammar: str):
        self.compile_lark(grammar)

    def compile_lark(self, grammar: str):
        """Grammar-only entrypoint to isolate frontend and backend failures."""
        try:
            from lark import Lark
            from fast_constraint.classify import ParserClassifier, clear_caches
            from fast_constraint.runner import Runner
        except Exception as exc:
            raise AdapterError("dependency", repr(exc)) from exc
        self.log_single(grammar)
        # PSC memoizes construction globally; drop the previous schema's
        # entries so memory does not accumulate across a worker's files.
        clear_caches()
        try:
            parser = Lark(grammar, parser="lalr", lexer="basic")
        except Exception as exc:
            raise AdapterError("grammar_parser", repr(exc), "unsupported") from exc
        # Construction prints progress bars and phase timings.
        with StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            try:
                self._runner = Runner.from_others(
                    ParserClassifier.from_others(parser, self.tokenizer)
                )
            except Exception as exc:
                raise AdapterError("preprocess", repr(exc)) from exc

    def reset(self):
        from fast_constraint.runner import RunningInstance

        self._inst = RunningInstance.from_runner(self._runner)

    def compute_mask(self):
        # No-op: the mask for the current state already exists.
        pass

    def commit_token(self, t: int) -> bool:
        mask = self._inst.mask
        if t < 0 or t >= len(mask) or not mask[t]:
            return False
        try:
            self._inst = self._inst.commit_token(t)
        except Exception as exc:
            # A mask-admitted token that cannot advance is a backend error,
            # not a correct rejection.
            raise AdapterError("advance", repr(exc)) from exc
        return True

    def check_complete(self) -> bool:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise AdapterError("completion", "tokenizer has no EOS token")
        mask = self._inst.mask
        return 0 <= eos < len(mask) and bool(mask[eos])
