"""Score saved raw completions only inside constrained local Docker containers.

Original responses stay unchanged. Raw scoring is primary; a deterministic
top-level-dedent stop is reported separately as a collection-format diagnostic.
Neither result is a comparable public leaderboard run or an automatic gate pass.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import subprocess
import threading
import time
import tokenize
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/optimization-48h/coding-review"
DOCKER = r"docker"
IMAGE = "python@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016"
LIMITS = ["--network", "none", "--read-only", "--cap-drop", "ALL",
          "--security-opt", "no-new-privileges", "--user", "65534:65534",
          "--cpus", "0.5", "--memory", "256m", "--memory-swap", "256m",
          "--pids-limit", "32", "--tmpfs", "/tmp:rw,size=32m,mode=1777,noexec,nosuid,nodev",
          "--workdir", "/tmp", "--log-driver", "none", "--ulimit", "nofile=64:64",
          "--entrypoint", "python"]
RUNNER = r'''
import contextlib, json, os, resource, signal, sys
payload = json.loads(sys.stdin.buffer.read(1048576))
resource.setrlimit(resource.RLIMIT_AS, (192*1024*1024, 192*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024*1024, 1024*1024))
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
class Sink:
    def write(self, text): return len(text)
    def flush(self): pass
    def read(self, *args): raise IOError('benchmark input disabled')
    def readline(self, *args): raise IOError('benchmark input disabled')
class TestTimeout(Exception): pass
def expire(*args): raise TestTimeout()
signal.signal(signal.SIGALRM, expire)
outcome = 'failed'
kind = None
try:
    signal.setitimer(signal.ITIMER_REAL, 3)
    with contextlib.redirect_stdout(Sink()), contextlib.redirect_stderr(Sink()):
        sys.stdin = Sink()
        exec(compile(payload['code'], '<humaneval>', 'exec'), {})
    outcome = 'passed'
except TestTimeout: outcome = 'timed-out'
except BaseException as exc: kind = type(exc).__name__
finally: signal.setitimer(signal.ITIMER_REAL, 0)
print(json.dumps({'outcome': outcome, 'exception_type': kind}), flush=True)
'''


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def deadline() -> datetime:
    budget = json.loads((ROOT / "data/optimization-48h/budget.json").read_text(encoding="utf-8-sig"))
    return datetime.fromisoformat(budget["deadline_utc"].replace("Z", "+00:00"))


def check_time() -> None:
    if datetime.now(timezone.utc) >= deadline():
        raise RuntimeError("Original 48-hour deadline reached")


def cli(args: list[str], timeout: int = 30) -> str:
    result = subprocess.run([DOCKER, *args], capture_output=True, text=True,
                            encoding="utf-8", timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"Docker operation failed: {result.stderr[:1000]}")
    return result.stdout


def stopped_completion(prompt: str, response: str) -> str:
    """Stop at the first dedent out of the prompt's function; never repair code."""
    combined = prompt + response
    lines = combined.splitlines(keepends=True)
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(combined).readline):
            if token.type == tokenize.INDENT:
                depth += 1
            elif token.type == tokenize.DEDENT:
                depth -= 1
                offset = sum(map(len, lines[:token.start[0] - 1])) - len(prompt)
                if depth == 0 and offset >= 0:
                    return response[:offset]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return response


def program(problem: dict, completion: str) -> str:
    return problem["prompt"] + completion + "\n" + problem["test"] + "\ncheck(" + problem["entry_point"] + ")\n"


def run_case(code: str) -> dict:
    check_time()
    name = "flash-next-he-" + uuid.uuid4().hex
    args = [DOCKER, "run", "--rm", "-i", "--name", name, *LIMITS,
            IMAGE, "-I", "-B", "-c", RUNNER]
    started = time.monotonic()
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outputs = [bytearray(), bytearray()]
    oversized = threading.Event()

    def read(pipe, output):
        while block := pipe.read(4096):
            if len(output) + len(block) > 65536:
                oversized.set()
            elif not oversized.is_set():
                output.extend(block)

    threads = [threading.Thread(target=read, args=(pipe, output), daemon=True)
               for pipe, output in zip((process.stdout, process.stderr), outputs)]
    for thread in threads:
        thread.start()
    try:
        process.stdin.write(json.dumps({"code": code}).encode())
        process.stdin.close()
        while process.poll() is None:
            check_time()
            if oversized.is_set() or time.monotonic() - started > 30:
                raise RuntimeError("Container output or wall-time limit reached")
            time.sleep(0.05)
        for thread in threads:
            thread.join(timeout=2)
        if oversized.is_set():
            raise RuntimeError("Container output limit reached")
        if process.returncode:
            # Engine/launch errors must not become failed model answers.
            raise RuntimeError(f"Container exited {process.returncode}: {bytes(outputs[1]).decode(errors='replace')[:500]}")
        result = json.loads(bytes(outputs[0]))
        if result.get("outcome") not in ("passed", "failed", "timed-out"):
            raise RuntimeError("Invalid container result")
        return {**result, "seconds": round(time.monotonic() - started, 3), "program_sha256": sha(code.encode())}
    finally:
        if process.poll() is None:
            subprocess.run([DOCKER, "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            process.kill()
            process.wait(timeout=5)
        subprocess.run([DOCKER, "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)


def freeze() -> tuple[list[dict], dict[str, list[dict]], dict]:
    dataset = ROOT / "data/eval/selected/humaneval.jsonl"
    selected_manifest = json.loads((dataset.parent / "manifest.json").read_text(encoding="utf-8-sig"))
    assert sha(dataset.read_bytes()) == selected_manifest["sets"]["humaneval"]["sha256"]
    problems = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    assert len(problems) == 164 and len({item["task_id"] for item in problems}) == 164
    samples, fingerprints = {}, {}
    for label in ("ista", "control", "original-iq4"):
        source = ROOT / f"data/eval/results/review-{label}-code/humaneval.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 164 and sorted(row["index"] for row in rows) == list(range(164))
        for row in rows:
            assert row["correct"] is None and row["set"] == "humaneval"
            assert row["row_sha256"] == sha(json.dumps(problems[row["index"]], sort_keys=True).encode())
        samples[label] = sorted(rows, key=lambda row: row["index"])
        fingerprints[label] = {"path": str(source), "sha256": sha(source.read_bytes())}
    manifest = {"script_sha256": sha(Path(__file__).read_bytes()), "image": IMAGE,
                "limits": LIMITS, "runner_sha256": sha(RUNNER.encode()), "sources": fingerprints,
                "dataset_sha256": sha(dataset.read_bytes()), "count_per_model": 164,
                "generation": "Raw completion256 tokens, temperature0, seed1729, no stop strings. Legacy result protocol field incorrectly says chat-512-v1; originals preserved.",
                "primary": "Unmodified raw completions against original tests",
                "secondary": "First top-level dedent stop, identical tokenizer rule for all models. No syntax/function repairs; may remove required helpers. Diagnostic only.",
                "reference": "https://github.com/openai/human-eval/blob/master/human_eval/execution.py",
                "scope": "Local deterministic pass@1 over164 tasks; collection-format limitations, not public-leaderboard parity or an automatic release gate"}
    path = FOLDER / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert {key: saved[key] for key in manifest} == manifest, "Frozen execution protocol changed"
        manifest = saved
    else:
        manifest["frozen_utc"] = utc()
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return problems, samples, manifest


def verify_sandbox(problem: dict) -> dict:
    name = "flash-next-he-preflight-" + uuid.uuid4().hex
    try:
        cli(["create", "-i", "--name", name, *LIMITS, IMAGE, "-I", "-B", "-c", RUNNER])
        config = json.loads(cli(["inspect", name]))[0]
        host = config["HostConfig"]
        assert config["Config"]["User"] == "65534:65534"
        assert host["NetworkMode"] == "none" and host["ReadonlyRootfs"]
        assert host["CapDrop"] == ["ALL"] and "no-new-privileges" in host["SecurityOpt"]
        assert host["Memory"] == host["MemorySwap"] == 256 * 1024 * 1024
        assert host["NanoCpus"] == 500000000 and host["PidsLimit"] == 32
        assert not host.get("Binds") and not host.get("Mounts")
        assert set(host["Tmpfs"]) == {"/tmp"} and not config.get("Mounts")
    finally:
        cli(["rm", "-f", name])
    passed = run_case(program(problem, problem["canonical_solution"]))
    failed = run_case(program(problem, "    return False\n"))
    timeout = run_case("while True: pass\n")
    assert passed["outcome"] == "passed" and failed["outcome"] == "failed"
    assert timeout["outcome"] == "timed-out"
    isolation = run_case("import os,socket\nassert os.getuid()==65534\nassert not os.path.exists('/var/run/docker.sock')\ntry:\n open('/isolation-probe','w')\nexcept PermissionError: pass\nexcept OSError: pass\nelse: raise AssertionError('root writable')\ns=socket.socket();s.settimeout(.2)\ntry:\n s.connect(('1.1.1.1',80))\nexcept OSError: pass\nelse: raise AssertionError('network reachable')\n")
    assert isolation["outcome"] == "passed"
    return {"utc": utc(), "host_config": host, "container_user": config["Config"]["User"],
            "cases": {"canonical": passed, "incorrect": failed, "timeout": timeout, "isolation": isolation}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    check_time()
    problems, samples, manifest = freeze()
    verification = verify_sandbox(problems[0])
    (FOLDER / "executor-verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    if args.preflight_only:
        print("Isolated executor verification passed", flush=True)
        return
    results_path = FOLDER / "results.jsonl"
    saved = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()] if results_path.exists() else []
    done = {(row["model"], row["index"]): row for row in saved}
    assert len(done) == len(saved)
    frozen_hash = sha(json.dumps(manifest, sort_keys=True).encode())
    for row in saved:
        assert row["manifest_sha256"] == frozen_hash
    with results_path.open("a", encoding="utf-8", newline="\n") as stream:
        for label, rows in samples.items():
            for response in rows:
                index = response["index"]
                if (label, index) in done:
                    continue
                check_time()
                problem = problems[index]
                raw = run_case(program(problem, response["response"]))
                stopped = stopped_completion(problem["prompt"], response["response"])
                trimmed = run_case(program(problem, stopped)) if stopped != response["response"] else {**raw, "reused_identical_program": True}
                item = {"model": label, "index": index, "task_id": problem["task_id"],
                        "raw": raw, "dedent_stop_diagnostic": trimmed,
                        "trimmed_characters": len(response["response"]) - len(stopped),
                        "manifest_sha256": frozen_hash, "utc": utc()}
                stream.write(json.dumps(item, sort_keys=True) + "\n")
                stream.flush()
                done[label, index] = item
                if (index + 1) % 16 == 0 or index == 163:
                    print(f"{label}: {index+1}/164 scored in isolation", flush=True)
    summary = {"utc": utc(), "manifest_sha256": frozen_hash, "models": {}, "scope": manifest["scope"]}
    for label in samples:
        rows = [done[label, index] for index in range(164)]
        summary["models"][label] = {"samples": len(rows), "raw_passed": sum(row["raw"]["outcome"] == "passed" for row in rows),
                                    "dedent_stop_passed": sum(row["dedent_stop_diagnostic"]["outcome"] == "passed" for row in rows),
                                    "stopped_responses": sum(row["trimmed_characters"] > 0 for row in rows)}
    from tools.evaluate_local import wilson
    summary["paired_comparisons"] = {}
    for candidate, reference in (("control", "ista"), ("original-iq4", "ista"), ("original-iq4", "control")):
        summary["paired_comparisons"][candidate + "-vs-" + reference] = {}
        for mode in ("raw", "dedent_stop_diagnostic"):
            delta = [int(done[candidate, index][mode]["outcome"] == "passed") -
                     int(done[reference, index][mode]["outcome"] == "passed") for index in range(164)]
            rng = random.Random(20261002)
            draws = sorted(sum(rng.choices(delta, k=164)) * 100 / 164 for _ in range(5000))
            summary["paired_comparisons"][candidate + "-vs-" + reference][mode] = {
                "difference_pp": sum(delta) * 100 / 164,
                "bootstrap_95_pp": [draws[124], draws[4874]], "resamples": 5000,
                "scope": "Paired task bootstrap; no multiple-comparison adjustment, not general coding confidence"}
    for model in summary["models"].values():
        for mode in ("raw", "dedent_stop"):
            correct = model[mode + "_passed"]
            model[mode + "_pass_at_1"] = correct / 164
            model[mode + "_wilson_95"] = wilson(correct, 164)
    (FOLDER / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
