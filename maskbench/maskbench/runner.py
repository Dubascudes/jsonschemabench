#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import time
import resource
import argparse
import signal
import traceback

from .engine import Engine
from transformers import AutoTokenizer


warm_replay = False


def time_us(prev: float) -> int:
    return int((time.monotonic() - prev) * 1000000)


CHECKPOINT_S = 30.0


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _peak_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def process_file(engine: Engine, file: str):
    """Compile one workload file and replay its tests token by token.

    A file is either a MaskBench schema ({"schema", "tests": [{"data",
    "valid"}]}) or a grammar workload ({"grammars": {format: text}, "tests":
    [{"text", "valid"}]}). Per token the runner records wall time
    (all_mask_us) and this thread's CPU time (all_mask_cpu_us) of
    compute_mask + commit_token; a large wall-minus-CPU gap marks a stall
    (preemption, page reclaim) rather than algorithmic cost. After the last
    token, engines implementing check_complete() are asked whether EOS is
    allowed, so a valid-prefix-but-incomplete input counts as rejected.
    Partial results are checkpointed, so a timeout keeps finished tests."""
    id = os.path.basename(file)
    output_name = os.path.join(output_path, id)

    if engine.multi:
        if os.path.exists(output_name):
            return None

        try:
            with open(output_name, "x") as f:
                f.write(json.dumps({"pending_file": 1}, indent=2))
        except FileExistsError:
            return None

    print(file, file=sys.stderr)

    with open(file) as f:
        pos_data = json.loads(f.read())
    is_grammar = "grammars" in pos_data

    # Skip schemas with no test cases — write a no-tests marker so the
    # run script doesn't re-queue this file every round.
    if not pos_data.get("tests"):
        status = {"id": id, "no_tests": True, "num_tokens": 0, "num_tests": 0,
                  "ttfm_us": 0, "max_ttfm_us": 0, "masks_us": 0, "max_mask_us": 0,
                  "all_mask_us": [], "num_valid_tests": 0, "num_invalid_tests": 0}
        _write_json(output_name, status)
        return status

    all_mask_us = []
    all_mask_cpu_us = []
    status = {
        "id": id,
        "workload_kind": "grammar" if is_grammar else "json_schema",
        "complete": False,
        "ttfm_us": 0,
        "max_ttfm_us": 0,
        "masks_us": 0,
        "max_mask_us": 0,
        "num_tokens": 0,
        "num_tests": len(pos_data["tests"]),
        "all_mask_us": all_mask_us,
        "all_mask_cpu_us": all_mask_cpu_us,
        "num_valid_tests": 0,
        "num_invalid_tests": 0,
        "adapter": getattr(engine, "benchmark_contract", None),
        "test_results": [],
        "completion_us": 0,
    }

    try:
        t0 = time.monotonic()
        c0 = time.process_time()
        if is_grammar:
            engine.compile_workload(pos_data["grammars"])
        else:
            engine.compile_grammar(pos_data["schema"])
    except Exception as e:
        if not engine.multi:
            traceback.print_exc()
        status["compile_error"] = repr(e)
        status["ttfm_failed_after_us"] = time_us(t0)
        # Adapter exceptions may distinguish unsupported frontends from
        # parser/matcher errors without changing existing result fields.
        if hasattr(e, "outcome"):
            status["outcome"] = e.outcome
            status["error_stage"] = e.stage
        status["peak_rss_kb"] = _peak_rss_kb()
        status["complete"] = True
        _write_json(output_name, status)
        return status

    status["ttfm_us"] = time_us(t0)
    status["ttfm_cpu_us"] = int((time.process_time() - c0) * 1e6)
    status["max_ttfm_us"] = status["ttfm_us"]
    status["peak_rss_after_compile_kb"] = _peak_rss_kb()
    if hasattr(engine, "compile_breakdown"):
        status["compile_breakdown"] = engine.compile_breakdown()
    status["outcome"] = "supported"

    masks_us = 0
    max_mask_us = 0
    num_tokens = 0
    cold_accepted = []
    last_checkpoint = time.monotonic()

    for i, test in enumerate(pos_data["tests"]):
        accepted = True
        test_error = None
        eof_checked = False
        rejected_at = None
        tokens = []
        if "text" in test:
            instance = test["text"]
        else:
            instance = json.dumps(test["data"], indent=None, ensure_ascii=False)
        try:
            tokens = engine.tokenizer.encode(instance, add_special_tokens=False)
            engine.reset()
            for tidx, t in enumerate(tokens):
                t2 = time.monotonic()
                c2 = time.thread_time()
                engine.compute_mask()
                ok = engine.commit_token(t)
                mask_time = time_us(t2)
                all_mask_cpu_us.append(int((time.thread_time() - c2) * 1e6))
                if engine.debug:
                    engine.log_single(
                        f"Token {tidx} {repr(engine.tokenizer.decode([t]))}: {ok}"
                    )
                num_tokens += 1
                masks_us += mask_time
                all_mask_us.append(mask_time)
                if mask_time > max_mask_us:
                    max_mask_us = mask_time
                if not ok:
                    accepted = False
                    rejected_at = tidx
                    break

            if accepted and hasattr(engine, "check_complete"):
                t_complete = time.monotonic()
                accepted = engine.check_complete()
                eof_checked = True
                status["completion_us"] += time_us(t_complete)
                if not accepted:
                    rejected_at = len(tokens)

            shown = instance[:200]
            if accepted and not test["valid"]:
                # Log details for false accepts
                detail = f"test #{i}: should reject but didn't"
                detail += f" | data={json.dumps(shown)}"
                if not is_grammar:
                    # Identify which constraint was violated
                    schema_str = json.dumps(pos_data.get("schema", {}))
                    features = [kw for kw in ['pattern', 'enum', 'const', '$ref',
                        'oneOf', 'anyOf', 'required', 'additionalProperties',
                        'minLength', 'maxLength', 'minItems', 'maxItems']
                        if f'"{kw}"' in schema_str]
                    detail += f" | schema_features={features}"
                status["validation_error"] = detail
            elif not accepted and test["valid"]:
                # Log details for false rejects
                if rejected_at is not None and rejected_at < len(tokens):
                    rejected_token = engine.tokenizer.decode([tokens[rejected_at]])
                else:
                    rejected_token = "<EOS>"
                detail = f"test #{i}: should accept but didn't"
                detail += f" | rejected_at_token={rejected_at}/{len(tokens)}"
                detail += f" | rejected_token={repr(rejected_token)}"
                detail += f" | data={json.dumps(shown)}"
                status["validation_error"] = detail
            else:
                if test["valid"]:
                    status["num_valid_tests"] += 1
                else:
                    status["num_invalid_tests"] += 1

        except Exception as e:
            if not engine.multi:
                traceback.print_exc()
            status["validation_error"] = f"test #{i}: EXN {repr(e)}"
            accepted = False
            test_error = repr(e)
            status["outcome"] = "error"
            status["error_stage"] = getattr(e, "stage", "replay")

        status["test_results"].append({
            "index": i, "expected_valid": test["valid"],
            "kind": test.get("kind"),
            "accepted": accepted if test_error is None else None,
            "error": test_error,
            "eof_checked": eof_checked,
            "num_tokens": len(tokens),
            "rejected_at": rejected_at,
        })
        if test_error is None and accepted != test["valid"] and status["outcome"] != "error":
            status["outcome"] = "validation_error"

        if time.monotonic() - last_checkpoint > CHECKPOINT_S:
            status["masks_us"], status["max_mask_us"] = masks_us, max_mask_us
            status["num_tokens"] = num_tokens
            status["tests_done"] = i + 1
            _write_json(output_name, status)
            last_checkpoint = time.monotonic()
        cold_accepted.append(accepted)

    status["masks_us"] = masks_us
    status["max_mask_us"] = max_mask_us
    status["num_tokens"] = num_tokens
    status["tests_done"] = len(pos_data["tests"])
    # before the warm pass, so mask_path stays aligned with all_mask_us
    if hasattr(engine, "replay_stats"):
        status["replay_stats"] = engine.replay_stats()
    if warm_replay:
        _warm_pass(engine, pos_data["tests"], cold_accepted, status)
    status.update(engine.compile_extras())
    status["peak_rss_kb"] = _peak_rss_kb()
    status["complete"] = True

    engine.log_single(json.dumps({k: v for k, v in status.items()
                                  if k not in ("all_mask_us", "all_mask_cpu_us")}, indent=2))
    _write_json(output_name, status)

    return status


def _warm_pass(engine: Engine, tests, cold_accepted, status):
    """Replay every test again on the same compiled grammar.

    Models a schema reused across requests: engine-side caches filled by the
    first (cold) pass stay warm. Timings go to separate warm_* fields; the
    cold fields keep their meaning. warm_outcome_mismatches counts tests whose
    accept/reject result differs from the cold pass (expected: 0).
    """
    warm_all = []
    mismatches = 0
    for test, cold in zip(tests, cold_accepted):
        if "text" in test:
            instance = test["text"]
        else:
            instance = json.dumps(test["data"], indent=None, ensure_ascii=False)
        tokens = engine.tokenizer.encode(instance, add_special_tokens=False)
        accepted = True
        try:
            engine.reset()
            for t in tokens:
                t2 = time.monotonic()
                engine.compute_mask()
                ok = engine.commit_token(t)
                warm_all.append(time_us(t2))
                if not ok:
                    accepted = False
                    break
            # same EOS check as the cold pass, so outcomes are comparable
            if accepted and hasattr(engine, "check_complete"):
                accepted = engine.check_complete()
        except Exception:
            accepted = False
        mismatches += accepted != cold
    status["warm_all_mask_us"] = warm_all
    status["warm_masks_us"] = sum(warm_all)
    status["warm_num_tokens"] = len(warm_all)
    status["warm_outcome_mismatches"] = mismatches


class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_argument(self, action):
        # Suppress defaults for certain arguments
        if action.default is not None and not isinstance(
            action, argparse._StoreTrueAction
        ):
            super().add_argument(action)
        else:
            # Call the parent method without the default
            action.default = argparse.SUPPRESS
            super().add_argument(action)


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Run mask computation.",
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument("--xgr", action="store_true", help="Enable XGrammar")
    parser.add_argument(
        "--xgr-cpp",
        action="store_true",
        help="Enable XGrammar with JSON->ENBF from llama.cpp",
    )
    parser.add_argument(
        "--xgr-compliant",
        action="store_true",
        help="Enable XGrammar in compliant (non-strict, any whitespace) mode",
    )
    parser.add_argument("--llg", action="store_true", help="Enable LLGuidance")
    parser.add_argument("--outlines", action="store_true", help="Enable Outlines")
    parser.add_argument(
        "--llamacpp", action="store_true", help="Enable llama.cpp grammars"
    )
    parser.add_argument(
        "--instagrammar", action="store_true", help="Enable InstaGrammar"
    )
    parser.add_argument(
        "--instagrammar-lp", action="store_true", help="Enable InstaGrammar (lexer-parser mode)"
    )
    parser.add_argument(
        "--syncode", action="store_true", help="Enable Syncode"
    )
    parser.add_argument(
        "--formatron", action="store_true", help="Enable Formatron"
    )
    parser.add_argument(
        "--psc", action="store_true", help="Enable PSC (parser stack classification)"
    )
    parser.add_argument(
        "--greatgramma", action="store_true", help="Enable GreatGramma"
    )
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="Tokenizer model ID",
    )

    parser.add_argument(
        "--time-limit", type=int, default=900, help="Time limit in seconds"
    )
    parser.add_argument(
        "--mem-limit", type=int, default=40, help="Memory (RSS) limit in gigabytes"
    )

    defl_cpu = min(os.cpu_count(), 40)
    # Memory-aware default: heavy engines (instagrammar) use ~2-4GB per
    # subprocess during compilation.  Cap concurrency so peak usage stays
    # under 75% of system RAM.
    try:
        import shutil
        _total_gb = shutil.disk_usage("/").total  # fallback, overridden below
        with open("/proc/meminfo") as _f:
            for _line in _f:
                if _line.startswith("MemTotal:"):
                    _total_gb = int(_line.split()[1]) / (1024 * 1024)
                    break
        _mem_threads = max(1, int(_total_gb * 0.75 / 4))  # ~4GB per worker
        defl_cpu = min(defl_cpu, _mem_threads)
    except Exception:
        pass
    parser.add_argument(
        "--num-threads", type=int, default=defl_cpu, help="Number of threads to run"
    )

    parser.add_argument(
        "--multi", action="store_true", help="Enable running from run_maskbench.py"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--warm-replay", action="store_true",
        help="After the timed replay, replay all tests again on the same compiled "
             "grammar and record warm_* timings")

    parser.add_argument(
        "files", metavar="file", type=str, nargs="+", help="List of files to process"
    )

    return parser


def get_engine(args) -> Engine:
    engine: Engine | None = None

    if args.xgr or args.xgr_compliant or args.xgr_cpp:
        from .xgr_engine import XgrEngine

        assert not engine, "Multiple engines specified"
        engine = XgrEngine()
        engine.compliant = args.xgr_compliant
        engine.llama_cpp = args.xgr_cpp

    if args.llg:
        from .llg_engine import LlgEngine

        assert not engine, "Multiple engines specified"
        engine = LlgEngine()

    if args.outlines:
        from .outlines_engine import OutlinesEngine

        assert not engine, "Multiple engines specified"
        engine = OutlinesEngine()

    if args.llamacpp:
        from .llamacpp_engine import LlamaCppEngine

        assert not engine, "Multiple engines specified"
        engine = LlamaCppEngine()

    if args.instagrammar:
        from .instagrammar_engine import InstaGrammarEngine

        assert not engine, "Multiple engines specified"
        engine = InstaGrammarEngine()

    if getattr(args, 'instagrammar_lp', False):
        from .instagrammar_lp_engine import InstaGrammarLPEngine

        assert not engine, "Multiple engines specified"
        engine = InstaGrammarLPEngine()

    if args.syncode:
        from .syncode_engine import SyncodeEngine

        assert not engine, "Multiple engines specified"
        engine = SyncodeEngine()

    if getattr(args, 'formatron', False):
        from .formatron_engine import FormatronEngine

        assert not engine, "Multiple engines specified"
        engine = FormatronEngine()

    if getattr(args, 'psc', False):
        from .psc_engine import PscEngine

        assert not engine, "Multiple engines specified"
        engine = PscEngine()

    if getattr(args, 'greatgramma', False):
        from .greatgramma_engine import GreatGrammaEngine

        assert not engine, "Multiple engines specified"
        engine = GreatGrammaEngine()

    if not engine:
        raise Exception("No grammar engine specified")

    engine.multi = args.multi
    engine.tokenizer_model_id = args.tokenizer
    engine.debug = args.debug

    return engine


def get_output(args):
    if args.output:
        return args.output
    else:
        id = get_engine(args).get_id()
        return f"tmp/out--{id}"


def get_files(args):
    files = []
    for arg in args.files:
        if arg.endswith(".json"):
            files.append(arg)
        else:
            files.extend(glob.glob(arg + "/*.json"))
    return files


def main():
    global output_path, warm_replay

    parser = setup_argparse()

    args = parser.parse_args()

    if sys.platform.startswith("linux"):
        limit_gb = args.mem_limit
        limit_bytes = limit_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    time_limit_s = args.time_limit

    engine = get_engine(args)
    output_path = get_output(args)
    warm_replay = args.warm_replay

    engine.tokenizer = AutoTokenizer.from_pretrained(
        engine.tokenizer_model_id
    )  # type: ignore

    engine.init()

    files = get_files(args)
    print(f"{len(files)} files, timeout {time_limit_s}s", file=sys.stderr)
    random.shuffle(files)

    os.makedirs(output_path, exist_ok=True)

    # rely on default SIGALRM handler (terminates the process)
    # def timeout_handler(signum, frame):
    #     print(f"Timeout ({time_limit_s}s). Terminating...", file=sys.stderr)
    #     os.kill(os.getpid(), signal.SIGKILL)

    # signal.signal(signal.SIGALRM, timeout_handler)

    for f in files:
        signal.alarm(time_limit_s)
        process_file(engine, f)
        signal.alarm(0)


if __name__ == "__main__":
    main()
