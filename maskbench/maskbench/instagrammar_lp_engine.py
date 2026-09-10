"""MaskBench engine adapter for InstaGrammar lexer-parser mode.

Benchmarks the SERVING path (identical to the vLLM backend): phases=4
compile (trace-signature table -> complete masks by pure retrieval) plus
a best-effort cache prewarm (cold fills move to load = GCT), a template
matcher cloned per test, and per-token fills through the C row path
(canonical fill key -> shared C mask cache; a state the prewarm missed
takes its one cold fill in the same C call). Masks are COMPLETE
(boundary-spanning), comparable to llguidance/xgrammar.

compute_mask() does the timed mask fill; commit_token() advances. The
runner times them together per token; TTFM is the phases=4 compile
including the prewarm.
"""

import collections
import hashlib
import json

import numpy as np

from .engine import Engine
from instagrammar.lexer_parser.compiler import LexerParserCompiler

_CACHE_MAX = 8


class InstaGrammarLPEngine(Engine):
    """MaskBench engine for InstaGrammar (Lexer-Parser, unified serving path)."""

    def __init__(self):
        super().__init__()
        self._compiler = None
        self._cache = collections.OrderedDict()
        self._matcher = None
        self._matcher_key = None
        self._row = None
        from instagrammar.lexer_parser._c_matcher import _available
        if not _available:
            raise RuntimeError(
                "lp_matcher.so not built (run `make build` in the "
                "InstaGrammar repo) -- the serving path is C-only")

    def get_id(self):
        return "instagrammar-lp"

    def get_name(self):
        return "InstaGrammar-LP"

    def get_module(self):
        return "instagrammar"

    def get_version(self):
        from instagrammar import __version__
        return __version__ + "-lp-serving"

    def init(self):
        pass

    def compile_grammar(self, schema: dict):
        from instagrammar.lexer_parser._c_matcher import (CLPMatcher,
                                                          _flatten_grammar)
        if self._compiler is None:
            self._compiler = LexerParserCompiler(self.tokenizer)
            self._row = np.zeros((self._compiler.vocab_size + 7) // 8,
                                 dtype=np.uint8)

        self._current_key = hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()).hexdigest()
        if self._current_key in self._cache:
            self._cache.move_to_end(self._current_key)
            self._compiled = self._cache[self._current_key]
            return

        # phases=4: trace-signature table (complete masks by retrieval) --
        # the exact compile the vLLM backend ships. TTFM == this call,
        # including the prewarm below.
        compiled = self._compiler.compile_schema(schema, verbose=0, phases=4)
        compiled._clp_flat = _flatten_grammar(compiled)
        compiled._clp_template = CLPMatcher(compiled)  # owns shared C cache
        # Pre-pay cold fills at load: BFS-warm the template's shared C mask
        # cache so serving is retrieval-only (states the budget misses still
        # take one cold fill in C at serving time). The C-side walk finishes
        # closure-complete grammars in well under the 1s cap; the cap exists
        # for data-shaped state spaces (e.g. free-text pattern terminals)
        # where more exploration cannot reach the serving states anyway.
        from instagrammar.lexer_parser._c_matcher import prewarm_cache
        prewarm_cache(CLPMatcher(compiled, mask_cache={},
                                 template=compiled._clp_template),
                      budget=10_000_000, time_limit_s=1.0)
        self._compiled = compiled
        self._cache[self._current_key] = compiled
        while len(self._cache) > _CACHE_MAX:
            evicted_key, _ = self._cache.popitem(last=False)
            if self._matcher_key == evicted_key:
                self._matcher = None
                self._matcher_key = None

    def reset(self):
        if self._matcher is not None and self._matcher_key == self._current_key:
            self._matcher.reset()
        else:
            from instagrammar.lexer_parser._c_matcher import CLPMatcher
            # clone = struct memcpy sharing the template's grammar + C mask
            # cache (the per-request path in the vLLM backend)
            self._matcher = CLPMatcher(self._compiled, mask_cache={},
                                       template=self._compiled._clp_template)
            self._matcher_key = self._current_key

    def compute_mask(self):
        m = self._matcher
        # ONE C call, warm AND cold: key -> shared C cache -> row memcpy;
        # a prewarm-missed state is assembled and inserted inside the call.
        st = m.fill_trace_row_into(self._row)
        if st > 0:
            return
        if st == -1:
            raise RuntimeError("matcher state truncated (config explosion)")
        # no C trace table attached: python retrieval fallback
        mask = m.fill_next_token_bitmask_retrieval()
        m.fill_row_insert_into(np.ascontiguousarray(mask), self._row)

    def commit_token(self, t: int) -> bool:
        return self._matcher.accept_token(t)
