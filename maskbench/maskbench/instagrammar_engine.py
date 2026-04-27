import collections
import hashlib
import json

from .engine import Engine
from instagrammar.json_schema_converter import json_schema_to_ebnf_v2
from instagrammar.sanitize import json_schema_to_ebnf as xgr_json_schema_to_ebnf, sanitize_ebnf

_CACHE_MAX = 1


class InstaGrammarEngine(Engine):
    """MaskBench engine for InstaGrammar (Stack Signature GCD).

    Pipeline: JSON schema → EBNF (via XGrammar) → sanitize → PDA → signatures → bitmasks.

    Uses a single C call per token: compute_mask() is a no-op and commit_token()
    calls step_token() which fills the bitmask and accepts the token in one shot.
    """

    def __init__(self):
        super().__init__()
        self._compiler = None
        self._ebnf_cache = collections.OrderedDict()
        self._matcher = None
        self._matcher_key = None

    def get_id(self):
        return "instagrammar"

    def get_name(self):
        return "InstaGrammar"

    def get_module(self):
        return "instagrammar"

    def get_version(self):
        from instagrammar import __version__
        return __version__

    def init(self):
        from instagrammar.engine import SignatureCompiler, create_chained_matcher
        self._SignatureCompiler = SignatureCompiler
        self._create_chained_matcher = create_chained_matcher

    def compile_grammar(self, schema: dict):
        if self._compiler is None:
            self._compiler = self._SignatureCompiler(self.tokenizer)

        # Try v2 converter first (better feature coverage), fall back to XGrammar
        try:
            ebnf = json_schema_to_ebnf_v2(schema)
        except Exception:
            ebnf = xgr_json_schema_to_ebnf(schema)
            ebnf = sanitize_ebnf(ebnf)

        # Cache compiled grammars by EBNF hash (many schemas produce identical EBNF)
        self._current_key = hashlib.sha256(ebnf.encode()).hexdigest()
        if self._current_key in self._ebnf_cache:
            self._ebnf_cache.move_to_end(self._current_key)
            self._compiled = self._ebnf_cache[self._current_key]
            return

        self._compiled = self._compiler.compile_ebnf(ebnf, verbose=0)
        self._ebnf_cache[self._current_key] = self._compiled
        while len(self._ebnf_cache) > _CACHE_MAX:
            evicted_key, _ = self._ebnf_cache.popitem(last=False)
            if self._matcher_key == evicted_key:
                self._matcher = None
                self._matcher_key = None

    def reset(self):
        if self._matcher is not None and self._matcher_key == self._current_key:
            self._matcher.reset()
        else:
            self._matcher = self._create_chained_matcher(self._compiled)
            self._matcher_key = self._current_key

    def compute_mask(self):
        # No-op: deferred to commit_token() for single C call via step_token()
        pass

    def commit_token(self, t: int) -> bool:
        _, ok = self._matcher.step_token(t)
        return ok
