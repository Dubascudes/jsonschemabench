#!/usr/bin/env python3
"""
Re-run MaskBench trials that resulted in "zero tokens" or "empty/corrupt" for
InstaGrammar.  Uses subprocess isolation so a crash in one schema can't take
down others.

Workflow:
  1. Scan existing output for failures (zero-token, pending, empty, corrupt).
  2. Filter to schemas that have at least one test case.
  3. Delete stale output files.
  4. Run N schemas concurrently via subprocesses (default: 8).
     Each subprocess invokes `python3 -m maskbench.runner --multi <schema>`.
  5. After each subprocess, validate the output and clean up pending files
     from crashes.

Usage (from the maskbench/ directory):
  python3 scripts/rerun_failures.py --instagrammar data/
  python3 scripts/rerun_failures.py --instagrammar --num-threads 4 data/
  python3 scripts/rerun_failures.py --instagrammar --dry-run data/

Optional flags (same as run_maskbench.py):
  --time-limit 900      per-schema timeout in seconds (default: 900)
  --mem-limit  40       RSS limit in GB (default: 40)
  --num-threads 8       concurrent schemas (default: 8)
  --tokenizer  <id>     HuggingFace tokenizer (default: Meta-Llama-3.1-8B-Instruct)
  --output     <path>   override output directory
  --dry-run             list files that would be re-run without actually running
"""

import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from maskbench.runner import setup_argparse, get_engine, get_output, get_files


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def classify_existing_result(path):
    """Return a failure category or None if the result is fine."""
    try:
        sz = os.path.getsize(path)
    except OSError:
        return None

    if sz == 0:
        return "empty"

    try:
        with open(path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return "corrupt"

    if data.get("pending_file"):
        return "pending"

    ce = data.get("compile_error", "")
    if ce:
        # Crash stubs from previous subprocess failures should be re-run,
        # UNLESS they indicate a deterministic failure (timeout, segfault).
        # Descriptive errors from this script are real results, not retryable
        if ce.startswith(("Timeout", "Segfault", "Killed", "Aborted")):
            return None
        if ce.startswith("subprocess exit"):
            # Parse the exit code
            try:
                code = int(ce.split()[-1])
            except (ValueError, IndexError):
                return "crash_stub"
            # -14 = SIGALRM (timeout), -11 = SIGSEGV, -6 = SIGABRT (OOM)
            # These are deterministic and won't improve on retry
            if code in (-14, -11, -9, -6):
                return None
            return "crash_stub"
        return None

    if data.get("num_tokens", 0) == 0:
        return "zero_tokens"

    return None


def _has_tests(data_path):
    """Check if the source data file has at least one test case."""
    try:
        with open(data_path) as fh:
            data = json.load(fh)
        return len(data.get("tests", [])) > 0
    except (json.JSONDecodeError, OSError):
        return False


def find_failures(output_dir, data_files):
    """Find data files whose existing results are failures worth re-running."""
    basename_to_path = {os.path.basename(f): f for f in data_files}
    failures = {}
    skipped_no_tests = 0

    for basename, data_path in basename_to_path.items():
        if not _has_tests(data_path):
            skipped_no_tests += 1
            continue

        result_path = os.path.join(output_dir, basename)
        if not os.path.exists(result_path):
            failures[basename] = (data_path, "missing")
            continue

        category = classify_existing_result(result_path)
        if category:
            failures[basename] = (data_path, category)

    if skipped_no_tests:
        print(f"Skipped {skipped_no_tests} schemas with no test cases.")

    return failures


# ---------------------------------------------------------------------------
# Per-schema subprocess runner
# ---------------------------------------------------------------------------

def validate_output(output_path):
    """Check if an output file is a valid, complete result. Returns status string."""
    if not os.path.exists(output_path):
        return "missing"
    try:
        sz = os.path.getsize(output_path)
        if sz == 0:
            return "empty"
        with open(output_path) as f:
            data = json.load(f)
        if data.get("pending_file"):
            return "pending"
        if data.get("compile_error"):
            return f"compile_error"
        nt = data.get("num_tokens", 0)
        nv = data.get("num_valid_tests", 0)
        ni = data.get("num_invalid_tests", 0)
        ve = data.get("validation_error", "")
        if nt == 0:
            return "zero_tokens"
        tag = "success" if not ve else f"validation_error"
        return f"ok: {nt} tokens, {nv}v {ni}i ({tag})"
    except (json.JSONDecodeError, OSError):
        return "corrupt"


def run_one_schema(data_path, runner_cmd, output_dir, time_limit):
    """Run a single schema in a subprocess. Returns (basename, status_string)."""
    basename = os.path.basename(data_path)
    output_path = os.path.join(output_dir, basename)

    cmd = runner_cmd + [data_path]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=time_limit + 30,  # extra grace beyond SIGALRM
        )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = -1

    # Validate the output regardless of exit code
    status = validate_output(output_path)

    # Clean up pending files from crashes
    if status in ("pending", "empty", "corrupt", "missing"):
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        # Write an error result so we don't retry endlessly on persistent crashers
        if exit_code != 0:
            err_status = {
                "id": basename,
                "compile_error": f"subprocess exit {exit_code}",
                "ttfm_us": 0, "max_ttfm_us": 0,
                "masks_us": 0, "max_mask_us": 0,
                "num_tokens": 0, "num_tests": 0,
                "all_mask_us": [],
                "num_valid_tests": 0, "num_invalid_tests": 0,
            }
            try:
                with open(output_path, "w") as f:
                    json.dump(err_status, f, indent=2)
            except OSError:
                pass
            status = f"crash (exit {exit_code})"

    return basename, status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # chdir to maskbench root so relative paths (e.g. tmp/out--instagrammar)
    # resolve correctly regardless of where this script is invoked from
    maskbench_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(maskbench_root)

    parser = setup_argparse()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List failures without re-running",
    )
    args = parser.parse_args()

    engine = get_engine(args)
    output_dir = get_output(args)
    data_files = get_files(args)

    if not data_files:
        print("Error: no data files found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Find failures
    failures = find_failures(output_dir, data_files)
    if not failures:
        print("No failures found — nothing to re-run.")
        return

    cats = Counter(cat for _, cat in failures.values())
    print(f"Found {len(failures)} schemas to re-run:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    print()

    if args.dry_run:
        for basename in sorted(failures):
            _, cat = failures[basename]
            print(f"  [{cat}] {basename}")
        return

    # Delete stale output files so the runner doesn't skip them
    deleted = 0
    for basename in failures:
        result_path = os.path.join(output_dir, basename)
        if os.path.exists(result_path):
            os.remove(result_path)
            deleted += 1
    print(f"Deleted {deleted} stale output files.")

    # Build the subprocess command (same args minus our extras and positional
    # file args, plus --multi).  Each subprocess gets exactly ONE schema file
    # so that crashes are correctly attributed and progress is visible.
    # Flags whose values should be passed through to the runner subprocess.
    # Their next arg (the value) must also be kept.
    VALUE_FLAGS = {"--mem-limit", "--time-limit", "--tokenizer", "--output"}
    # Flags to strip entirely (flag + value).
    STRIP_FLAGS = {"--num-threads"}

    passthrough = []
    skip_next = False
    keep_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if keep_next:
            passthrough.append(a)
            keep_next = False
            continue
        if a == "--dry-run":
            continue
        if a in STRIP_FLAGS:
            skip_next = True
            continue
        if a in VALUE_FLAGS:
            passthrough.append(a)
            keep_next = True
            continue
        # Strip positional file/directory args (they don't start with --)
        if not a.startswith("--"):
            continue
        passthrough.append(a)

    runner_cmd = [
        sys.executable, "-m", "maskbench.runner",
        "--multi",
    ] + passthrough

    n_threads = args.num_threads
    # Divide cores across workers: OMP_NUM_THREADS = total_cores / n_threads
    total_cores = os.cpu_count() or 4
    omp_per_worker = max(1, total_cores // n_threads)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_per_worker)

    print(f"\nRunning {len(failures)} schemas, {n_threads} concurrent workers, "
          f"{omp_per_worker} OMP threads each")
    print(f"Command template: {' '.join(runner_cmd)} <schema>")
    print(f"Output: {output_dir}")
    print()

    # Override runner_cmd to include env
    def run_with_env(data_path):
        basename = os.path.basename(data_path)
        output_path = os.path.join(output_dir, basename)
        cmd = runner_cmd + [data_path]
        stderr_tail = ""
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=args.time_limit + 30, env=env,
            )
            exit_code = result.returncode
            stderr_tail = (result.stderr or "")[-500:]
        except subprocess.TimeoutExpired:
            exit_code = -1

        status = validate_output(output_path)

        # Clean up pending/corrupt from crashes
        if status in ("pending", "empty", "corrupt", "missing"):
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            if exit_code != 0:
                # Map signal-based exit codes to descriptive errors
                SIGNAL_NAMES = {
                    -14: "Timeout (SIGALRM)",
                    -11: "Segfault (SIGSEGV)",
                    -9:  "Killed (SIGKILL)",
                    -6:  "MemoryError (SIGABRT)",
                    -1:  "Timeout (subprocess)",
                }
                error_desc = SIGNAL_NAMES.get(
                    exit_code, f"subprocess exit {exit_code}"
                )
                err_status = {
                    "id": basename,
                    "compile_error": error_desc,
                    "stderr_tail": stderr_tail,
                    "ttfm_us": 0, "max_ttfm_us": 0,
                    "masks_us": 0, "max_mask_us": 0,
                    "num_tokens": 0, "num_tests": 0,
                    "all_mask_us": [],
                    "num_valid_tests": 0, "num_invalid_tests": 0,
                }
                try:
                    with open(output_path, "w") as f:
                        json.dump(err_status, f, indent=2)
                except OSError:
                    pass
                status = f"crash ({error_desc})"

        return basename, status

    # Run with thread pool
    total = len(failures)
    succeeded = 0
    errored = 0
    still_zero = 0
    completed = 0
    t_start = time.monotonic()
    print_lock = Lock()

    sorted_items = sorted(failures.items())

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        future_to_basename = {}
        for basename, (data_path, old_cat) in sorted_items:
            fut = pool.submit(run_with_env, data_path)
            future_to_basename[fut] = (basename, old_cat)

        for fut in as_completed(future_to_basename):
            basename, old_cat = future_to_basename[fut]
            completed += 1
            elapsed = time.monotonic() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0

            try:
                _, status = fut.result()
            except Exception as e:
                status = f"future error: {e}"

            if status.startswith("ok:"):
                succeeded += 1
            elif "zero_tokens" in status:
                still_zero += 1
            else:
                errored += 1

            with print_lock:
                print(
                    f"[{completed}/{total}] {basename}  (was: {old_cat})  "
                    f"-> {status}  "
                    f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left]",
                    flush=True
                )

    elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Succeeded (got tokens): {succeeded}")
    print(f"  Still zero tokens:      {still_zero}")
    print(f"  Errors/crashes:         {errored}")
    print(f"  Total re-run:           {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
