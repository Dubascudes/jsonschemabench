"""Formatron / kbnf (Sun et al., ICML 2025) MaskBench adapter.

JSON Schema goes through Formatron's own frontend (``create_schema`` +
``FormatterBuilder``); grammar workloads use the KBNF version of each
language, as in PSC's Formatron harness (``kbnf.Engine(grammar, vocab)``).
Both paths drive the underlying kbnf Earley engine directly:
``compute_allowed_token_ids`` per step, the token is checked against that
computation, then accepted.

Completion: PSC's KBNF grammars end with a literal ``"$"`` end marker and
never admit the EOS token, so an EOS-bit check would reject every valid
program. ``check_complete`` therefore asks whether the engine accepts the end
marker and finishes (for the JSON path, whether the engine is finished or
accepts EOS).
"""
import typing as _stdlib_typing

try:  # Formatron still imports ``pydantic.typing`` (removed in pydantic 2).
    import pydantic.typing as _pyd_typing  # type: ignore
except ImportError:
    import sys as _sys
    _sys.modules["pydantic.typing"] = _stdlib_typing
    _pyd_typing = _stdlib_typing
for _name in dir(_stdlib_typing):
    if not _name.startswith("_"):
        try:
            setattr(_pyd_typing, _name, getattr(_stdlib_typing, _name))
        except (AttributeError, TypeError):
            pass

from .engine import Engine, StageError


class FormatronEngine(Engine):
    grammar_format = "kbnf"
    benchmark_contract = {
        "frontend": "formatron create_schema (JSON) / PSC KBNF (grammars)",
        "timed_operation": "compute_allowed_token_ids + check + try_accept_new_token",
        "acceptance": "KBNF: end marker '$' accepted and engine finished; "
                      "JSON: engine finished or EOS allowed",
    }

    def __init__(self):
        super().__init__()
        self._engine = None
        self._end_marker = None

    def get_id(self):
        return "formatron"

    def get_name(self):
        return "Formatron"

    def get_module(self):
        return "formatron"

    def init(self):
        from formatron.integrations.transformers import create_engine_vocabulary

        self._vocab = create_engine_vocabulary(self.tokenizer)

        def decode(token_ids):
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)

        self._decode = decode

    def compile_grammar(self, schema: dict):
        from formatron.formatter import FormatterBuilder
        from formatron.schemas.json_schema import create_schema

        if "$id" not in schema or "$schema" not in schema:
            schema = dict(schema)
            schema.setdefault("$id", "https://maskbench.local/schema")
            schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
        try:
            schema_obj = create_schema(schema)
        except Exception as exc:
            raise StageError("schema_conversion", repr(exc), "unsupported") from exc
        builder = FormatterBuilder()
        # append_str, not append_line: MaskBench instances carry no trailing
        # newline, so append_line would make every valid instance incomplete.
        builder.append_str(f"{builder.json(schema_obj, capture_name='output')}")
        formatter = builder.build(self._vocab, self._decode)
        self._engine = formatter._engine
        self._end_marker = None

    def compile_text_grammar(self, grammar: str):
        from kbnf import Engine as KbnfEngine

        try:
            self._engine = KbnfEngine(grammar, self._vocab)
        except Exception as exc:
            raise StageError("grammar_parser", repr(exc)) from exc
        self._end_marker = b"$"

    def reset(self):
        self._engine.reset()

    def compute_mask(self):
        self._engine.compute_allowed_token_ids()

    def commit_token(self, t: int) -> bool:
        if not self._engine.check_if_token_is_allowed(t):
            return False
        self._engine.try_accept_new_token(t)
        return True

    def check_complete(self) -> bool:
        import kbnf

        if self._end_marker is not None:
            try:
                result = self._engine.try_accept_new_bytes(self._end_marker)
            except ValueError:
                return False
            return result == kbnf.AcceptTokenResult.Finished or self._engine.is_finished()
        if self._engine.is_finished():
            return True
        self._engine.compute_allowed_token_ids()
        eos = self.tokenizer.eos_token_id
        return eos is not None and self._engine.check_if_token_is_allowed(eos)
