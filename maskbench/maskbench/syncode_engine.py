from .engine import Engine
import json
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

    # Ensure the top-level package exists as an empty namespace so that
    # relative imports inside syncode submodules resolve correctly.
    if "syncode" not in sys.modules:
        pkg = types.ModuleType("syncode")
        pkg.__path__ = __import__("syncode", fromlist=["__path__"]).__path__ if False else []
        # Find the real package path from the installed location
        import importlib.util
        spec = importlib.util.find_spec("syncode")
        if spec and spec.submodule_search_locations:
            pkg.__path__ = list(spec.submodule_search_locations)
        pkg.__package__ = "syncode"
        sys.modules["syncode"] = pkg

    # Import only the submodules we need, which avoids the
    # syncode.infer -> syncode.dataset -> mxeval.data chain.
    importlib.import_module("syncode.common")
    importlib.import_module("syncode.parsers")
    importlib.import_module("syncode.parsers.grammars")
    importlib.import_module("syncode.mask_store")
    importlib.import_module("syncode.grammar_decoder")


class SyncodeEngine(Engine):
    def __init__(self):
        super().__init__()
        self._input_ids = None
        self._accept_mask = None
        self._processor = None

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

        grammar = Grammar("json")
        self._processor = SyncodeLogitsProcessor(
            grammar=grammar,
            tokenizer=self.tokenizer,
            use_cache=True,
            parse_output_only=False,
            num_samples=1,
            parser="lalr",
            mode="grammar_mask",
        )

    def compile_grammar(self, schema: dict):
        # Syncode only supports a generic JSON grammar, not per-schema
        # constraints. Nothing schema-specific to compile here.
        pass

    def reset(self):
        self._input_ids = torch.zeros((1, 0), dtype=torch.long)
        self._accept_mask = None
        self._processor.reset()

    def compute_mask(self):
        vocab_size = len(self.tokenizer)
        scores = torch.zeros((1, vocab_size), dtype=torch.float32)
        masked_scores = self._processor(self._input_ids, scores.clone())
        # Tokens not set to -inf are accepted
        self._accept_mask = masked_scores[0] > -float("inf")

    def commit_token(self, t: int) -> bool:
        # Always advance input_ids and return True.  Syncode decodes
        # input_ids to text and re-lexes (though parsing is incremental
        # via cached parser states).  Its DFA mask can miss valid multi-
        # terminal tokens (e.g. '{"'), so gating on the mask would abort
        # most test sequences on the first token.
        self._input_ids = torch.cat(
            [self._input_ids, torch.tensor([[t]], dtype=torch.long)], dim=1
        )
        return True
