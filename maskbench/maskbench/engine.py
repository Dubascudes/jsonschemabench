from transformers import PreTrainedTokenizer


class StageError(ValueError):
    """A failure attributed to one pipeline stage; the runner records
    ``stage`` and ``outcome`` ('unsupported' or 'error') in the result row."""

    def __init__(self, stage, message, outcome="error"):
        super().__init__(message)
        self.stage = stage
        self.outcome = outcome


class Engine:
    tokenizer: PreTrainedTokenizer
    tokenizer_model_id: str
    # Which of a grammar workload's matched formats this engine consumes
    # ('lark', 'lark_llg', 'gbnf', 'kbnf', 'ig_ebnf'), following PSC's
    # matching: Lark for PSC/GreatGramma/SynCode, priority-free Lark for
    # llguidance, GBNF for XGrammar, KBNF for Formatron.
    grammar_format = None

    def __init__(self):
        self.multi = False
        self.debug = False

    def get_id(self):
        raise NotImplementedError()

    def get_name(self):
        return self.get_id()

    def get_module(self):
        return self.get_id()

    def get_version(self):
        from importlib.metadata import version

        return version(self.get_module())

    def init(self):
        raise NotImplementedError()

    def compile_grammar(self, schema: dict):
        raise NotImplementedError()

    def compile_workload(self, grammars: dict):
        """Compile one grammar workload given all its matched formats."""
        fmt = self.grammar_format
        if fmt is None or not hasattr(self, "compile_text_grammar"):
            raise StageError("grammar_format", f"{self.get_id()} has no grammar entry point",
                             "unsupported")
        if fmt not in grammars:
            raise StageError("grammar_format", f"workload has no {fmt} grammar", "unsupported")
        self.compile_text_grammar(grammars[fmt])

    def eos_allowed(self, mask_bits) -> bool:
        """Helper for check_complete: is the EOS token set in a packed
        little-endian bitmask (uint8 or int32 words)?"""
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise StageError("completion", "tokenizer has no EOS token")
        width = mask_bits.dtype.itemsize * 8
        word = int(mask_bits.reshape(-1)[eos // width])
        return bool((word >> (eos % width)) & 1)

    def reset(self):
        raise NotImplementedError()

    def compute_mask(self):
        raise NotImplementedError()

    def commit_token(self, token: int) -> bool:
        raise NotImplementedError()

    def log_single(self, s: str):
        import sys

        if not self.multi:
            print(s, file=sys.stderr)
