"""MaskBench engine adapter for InstaGrammar lexer-parser mode.

Benchmarks the SERVING path (identical to the vLLM backend): phases=4
compile (trace-signature table -> complete masks by pure retrieval), an
optional cache prewarm, a template matcher cloned per test, and per-token
fills through the C row path (canonical fill key -> shared C mask cache;
a state the prewarm missed takes its one cold fill in the same C call).
Masks are COMPLETE (boundary-spanning), comparable to llguidance/xgrammar.

compute_mask() does the timed mask fill; commit_token() advances. The
runner times them together per token; TTFM is the whole compile_grammar
(or compile_text_grammar) call, including any synchronous prewarm.
INSTAGRAMMAR_PREWARM selects the prewarm (none / initial [default] /
capped / background, see instagrammar.lexer_parser._c_matcher.start_prewarm).
compile_extras() reports the compile / matcher-setup / prewarm split per
schema (compile_breakdown() is the same split under the audit's names);
replay_stats() tags every timed token as a warm hit (W), cold C fill (C)
or Python fallback (P), aligned with all_mask_us.
"""

import collections
import hashlib
import json
import time

import numpy as np

from .engine import Engine, StageError
from instagrammar.lexer_parser.compiler import LexerParserCompiler

# MaskBench visits each schema once (files are shuffled, never revisited),
# so a multi-entry LRU only multiplies peak memory: each compiled schema
# pins its phase-4 tables plus a ~131 MB C mask cache. Keep one.
_CACHE_MAX = 1


class InstaGrammarLPEngine(Engine):
    """MaskBench engine for InstaGrammar (Lexer-Parser, unified serving path)."""

    # Grammar workloads: InstaGrammar EBNF translated from PSC's GBNF
    # (experiments/workloads/gbnf.py), i.e. the same language XGrammar gets.
    grammar_format = "ig_ebnf"
    benchmark_contract = {
        "compile": "phases=4 + template + prewarm (INSTAGRAMMAR_PREWARM)",
        "timed_operation": "fill_trace_row_into + mask-bit check + accept_token",
        "acceptance": "token must be set in the complete mask; EOS bit "
                      "checked after the last token",
    }

    def __init__(self):
        super().__init__()
        self._compiler = None
        self._cache = collections.OrderedDict()
        self._matcher = None
        self._matcher_key = None
        self._row = None
        self._extras = {}
        self._prewarm = None
        self._path = []
        self._step = None
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

    def _ensure_compiler(self):
        if self._compiler is None:
            self._compiler = LexerParserCompiler(self.tokenizer)
            self._row = np.zeros((self._compiler.vocab_size + 7) // 8,
                                 dtype=np.uint8)

    def compile_grammar(self, schema: dict):
        self._ensure_compiler()
        key = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        self._load(key, lambda: self._compiler.compile_schema(schema, verbose=0, phases=4))

    def compile_text_grammar(self, grammar: str):
        self._ensure_compiler()
        key = hashlib.sha256(grammar.encode()).hexdigest()
        try:
            self._load(key, lambda: self._compiler.compile_ebnf(grammar, verbose=0, phases=4))
        except MemoryError:
            raise
        except ValueError as exc:
            raise StageError("grammar_parser", repr(exc)) from exc

    def _load(self, key, compile_fn):
        from instagrammar.lexer_parser._c_matcher import (
            CLPMatcher, _flatten_grammar, cancel_background_prewarms,
            prewarm_mode, start_prewarm)
        mode = prewarm_mode()
        # a previous schema's background walk must not compete with this
        # compile (or its timed fills) for CPU
        cancel_background_prewarms()
        self._current_key = key
        self._path = []
        self._prewarm = None
        if key in self._cache:
            self._cache.move_to_end(key)
            self._compiled = self._cache[key]
            self._extras = {"ig_prewarm_mode": mode, "ig_cache_hit": True,
                            "ig_compile_us": 0, "ig_matcher_init_us": 0,
                            "ig_prewarm_us": 0}
            return

        # phases=4: trace-signature table (complete masks by retrieval) --
        # the exact compile the vLLM backend ships. TTFM == this call,
        # including the prewarm below.
        t0 = time.perf_counter()
        compiled = compile_fn()
        t1 = time.perf_counter()
        compiled._clp_flat = _flatten_grammar(compiled)
        compiled._clp_template = CLPMatcher(compiled)  # owns shared C cache
        t2 = time.perf_counter()
        # Optional pre-pay of cold fills (INSTAGRAMMAR_PREWARM). A state the
        # prewarm misses takes one cold fill in C at serving time, so the
        # mode moves cost between TTFM and TBM without changing any mask.
        self._prewarm = start_prewarm(compiled, mode)
        t3 = time.perf_counter()
        self._extras = {
            "ig_prewarm_mode": mode, "ig_cache_hit": False,
            "ig_compile_us": int((t1 - t0) * 1e6),
            "ig_matcher_init_us": int((t2 - t1) * 1e6),
            # synchronous part only (thread launch for `background`)
            "ig_prewarm_us": int((t3 - t2) * 1e6),
        }
        self._compiled = compiled
        self._cache[key] = compiled
        while len(self._cache) > _CACHE_MAX:
            evicted_key, _ = self._cache.popitem(last=False)
            if self._matcher_key == evicted_key:
                self._matcher = None
                self._matcher_key = None
                self._step = None

    def compile_extras(self) -> dict:
        """Per-schema TTFM split for the runner's status record. Read after
        the tests, so a background walk reports its states and wall time
        if it has finished (None while still running)."""
        extras = dict(self._extras)
        extras["ig_binding"] = "direct" if self._step is not None else "ctypes"
        pw = self._prewarm
        if pw is not None:
            extras["ig_prewarm_states"] = pw.states
            if pw.mode == "background":
                extras["ig_prewarm_bg_us"] = (
                    int(pw.elapsed_s * 1e6) if pw.elapsed_s is not None
                    else None)
        extras.update(self._table_stats())
        return extras

    def _table_stats(self) -> dict:
        """Signature count and memory of the compiled grammar: the static
        serving tables (C trace table buffers plus the flattened matcher
        grammar) and the shared runtime caches after this schema's replay."""
        compiled = getattr(self, "_compiled", None)
        if compiled is None:
            return {}
        try:
            from instagrammar.lexer_parser._c_matcher import cache_mem_stats
            tr = compiled.trace_retrieval
            nbytes = lambda xs: sum(int(a.nbytes) for a in xs
                                    if isinstance(a, np.ndarray))
            flat = getattr(compiled, "_clp_flat", None) or {}
            stats = {
                "ig_n_signatures": int(tr.sigs.n_signatures),
                "ig_n_parser_states": int(compiled.parser_pda.n_states),
                "ig_trace_table_bytes": nbytes(getattr(tr, "_c_arrays", ()) or ()),
                "ig_matcher_table_bytes": nbytes(
                    flat.values() if isinstance(flat, dict) else ()),
            }
            if self._matcher is not None:
                stats["ig_cache_bytes"] = int(sum(
                    v for v in cache_mem_stats(self._matcher).values()))
            return stats
        except Exception as e:  # stats must never fail a benchmark row
            return {"ig_table_stats_error": repr(e)[:200]}

    def compile_breakdown(self):
        """The compile_extras() TTFM split under the audit runner's names."""
        e = self._extras
        if e.get("ig_cache_hit"):
            return {"cache_hit": True}
        pw = self._prewarm
        return {
            "compile_us": e.get("ig_compile_us"),
            "template_us": e.get("ig_matcher_init_us"),
            "prewarm_us": e.get("ig_prewarm_us"),
            "prewarm_states": int(pw.states or 0) if pw is not None else 0,
            "prewarm_mode": e.get("ig_prewarm_mode"),
        }

    def replay_stats(self):
        path = "".join(self._path)
        return {"mask_path": path, "warm": path.count("W"),
                "cold": path.count("C"), "python": path.count("P")}

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
            self._step = self._matcher.direct_step(self._row)

    def _fill(self):
        """Fill self._row with the complete mask; returns 'W', 'C' or 'P'."""
        m = self._matcher
        # ONE C call, warm AND cold: key -> shared C cache -> row memcpy;
        # a prewarm-missed state is assembled and inserted inside the call.
        st = self._step.fill() if self._step is not None else 0
        if st == 0:
            st = m.fill_trace_row_into(self._row)
        if st == 1:
            return "W"
        if st == 2:
            return "C"
        if st == -1:
            raise StageError("config_explosion",
                             "matcher state truncated (LP_MAX_CONFIGS/LP_MAX_ACTIVE)")
        if st == -2:
            raise MemoryError("matcher C allocation failed")
        # no C trace table attached: python retrieval fallback
        try:
            mask = m.fill_next_token_bitmask_retrieval()
        except RuntimeError as exc:
            if "truncated" in str(exc):
                raise StageError("config_explosion", str(exc)) from exc
            raise
        m.fill_row_insert_into(np.ascontiguousarray(mask), self._row)
        return "P"

    def compute_mask(self):
        self._path.append(self._fill())

    def check_complete(self) -> bool:
        self._fill()
        return self.eos_allowed(self._row)

    def commit_token(self, t: int) -> bool:
        # Judge the token by the MASK the engine just produced (as the
        # llguidance adapter does), not only by the acceptor: a token the
        # acceptor takes but the mask forbids is a mask-soundness error and
        # must surface as a validation error, not be silently accepted.
        if self._step is not None:
            result = self._step.accept(t)
            if result == 2:
                self._matcher._check_truncated()
            return result == 1
        if not (self._row[t >> 3] >> (t & 7)) & 1:
            return False
        return self._matcher.accept_token(t)
