"""GreatGramma (Park, Zhou & D'Antoni, ICML 2025) MaskBench adapter.

Uses the reference implementation from github.com/large-loris-models/greatgramma
(``alignment.monitor.grammar.CFGMonitor``, Cython modules compiled) with the
PSC benchmark's JSON Schema -> Lark frontend, as in PSC's GreatGramma
comparison.

GreatGramma's native mask is a dict {token_id: next parser stack}. As in its
own ``mask_logits`` and PSC's harness, ``compute_mask`` also materializes a
vocabulary-sized boolean mask, so TBM covers the work needed to mask logits.
``reset`` builds a fresh initial state: ``CFGMonitorState`` memoizes
successors, and reusing it would let later test instances hit that cache.
"""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from .engine import Engine
from .lark_frontend import AdapterError, frontend_identity, json_schema_to_lark


class GreatGrammaEngine(Engine):
    benchmark_contract = {
        "frontend": "psc-json-schema-to-lark",
        "parser": "lalr",
        "cache": "disabled",
        "timed_operation": "acceptance-dict-plus-bool-mask-plus-advance",
        "acceptance": "complete replay; EOS check outside timed token samples",
    }

    def __init__(self):
        super().__init__()
        self._monitor = None
        self._state = None
        self._accept = None
        self._mask = None

    def get_id(self):
        return "greatgramma"

    def get_name(self):
        return "GreatGramma"

    def get_module(self):
        return "alignment"

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
        self._monitor = None
        grammar = json_schema_to_lark(schema)
        self.compile_lark(grammar)

    # PSC's matching: the reference Lark grammar.
    grammar_format = "lark"

    def compile_text_grammar(self, grammar: str):
        self.compile_lark(grammar)

    def compile_lark(self, grammar: str):
        """Grammar-only entrypoint to isolate frontend and backend failures."""
        try:
            import torch
            from lark.exceptions import LarkError
            from alignment.monitor.grammar import CFGMonitor
        except Exception as exc:
            raise AdapterError("dependency", repr(exc)) from exc
        torch.set_num_threads(1)
        self.log_single(grammar)
        with StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            try:
                self._monitor = CFGMonitor.from_tokenizer(grammar, self.tokenizer)
            except LarkError as exc:
                raise AdapterError("grammar_parser", repr(exc), "unsupported") from exc
            except Exception as exc:
                raise AdapterError("preprocess", repr(exc)) from exc
        # Covers every id the tokenizer can emit, including added tokens.
        self._vocab_size = max(len(self.tokenizer), max(self._monitor.vocabulary) + 1)

    def reset(self):
        from alignment.monitor.grammar import CFGMonitorState

        init = self._monitor.initial_state
        self._state = CFGMonitorState(
            init.lexer_state, list(init.stack), init.lexer, init.parse_table
        )
        self._accept = None
        self._mask = None

    def compute_mask(self):
        import torch

        try:
            self._accept = self._state.acceptance
        except Exception as exc:
            raise AdapterError("mask", repr(exc)) from exc
        mask = torch.zeros(self._vocab_size, dtype=torch.bool)
        if self._accept:
            mask[list(self._accept)] = True
        self._mask = mask

    def commit_token(self, t: int) -> bool:
        if self._mask is None:
            raise AdapterError("advance", "compute_mask must precede commit_token")
        mask, self._mask = self._mask, None
        if t < 0 or t >= len(mask) or not mask[t]:
            return False
        try:
            self._state = self._state.feed_token(t)
        except Exception as exc:
            raise AdapterError("advance", repr(exc)) from exc
        return True

    def check_complete(self) -> bool:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise AdapterError("completion", "tokenizer has no EOS token")
        try:
            accept = self._state.acceptance
        except Exception as exc:
            raise AdapterError("completion", repr(exc)) from exc
        return eos in accept
