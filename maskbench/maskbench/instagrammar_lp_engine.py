"""MaskBench engine adapter for InstaGrammar lexer-parser mode."""

import collections
import hashlib
import json

from .engine import Engine
from instagrammar.lexer_parser.compiler import LexerParserCompiler
from instagrammar.lexer_parser.matcher import LexerParserMatcher

_CACHE_MAX = 8


class InstaGrammarLPEngine(Engine):
    """MaskBench engine for InstaGrammar (Lexer-Parser separated mode).

    Pipeline: JSON schema → terminals + parser PDA → DFA bitmasks → online matcher.
    Uses C-accelerated matcher when available.
    """

    def __init__(self):
        super().__init__()
        self._compiler = None
        self._cache = collections.OrderedDict()
        self._use_c = False
        self._matcher = None
        self._matcher_key = None
        try:
            from instagrammar.lexer_parser._c_matcher import _available
            self._use_c = _available
        except ImportError:
            pass

    def get_id(self):
        return "instagrammar-lp"

    def get_name(self):
        return "InstaGrammar-LP"

    def get_module(self):
        return "instagrammar"

    def get_version(self):
        from instagrammar import __version__
        return __version__ + "-lp"

    def init(self):
        pass

    def compile_grammar(self, schema: dict):
        if self._compiler is None:
            self._compiler = LexerParserCompiler(self.tokenizer)

        self._current_key = hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()).hexdigest()
        if self._current_key in self._cache:
            self._cache.move_to_end(self._current_key)
            self._compiled = self._cache[self._current_key]
            return

        self._compiled = self._compiler.compile_schema(schema, verbose=0)
        self._cache[self._current_key] = self._compiled
        while len(self._cache) > _CACHE_MAX:
            evicted_key, _ = self._cache.popitem(last=False)
            if self._matcher_key == evicted_key:
                self._matcher = None
                self._matcher_key = None

    def reset(self):
        if self._use_c:
            if self._matcher is not None and self._matcher_key == self._current_key:
                self._matcher.reset()
            else:
                from instagrammar.lexer_parser._c_matcher import CLPMatcher
                self._matcher = CLPMatcher(self._compiled)
                self._matcher_key = self._current_key
        else:
            self._matcher = LexerParserMatcher(self._compiled)
            self._matcher_key = self._current_key

    def compute_mask(self):
        pass

    def commit_token(self, t: int) -> bool:
        _, ok = self._matcher.step_token(t)
        return ok
