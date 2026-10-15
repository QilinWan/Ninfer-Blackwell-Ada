#!/usr/bin/env python3
"""Linux launcher for the agentic A/B suite: two ninfer-serve builds, each under a systemd unit.

Replaces runner.py's Windows orchestration (launch bat, control calibration, host-cache
translation, taskkill) and reuses its closed-loop client unchanged. Both arms get the same
artifact and the same flag profile; only the executable differs. Per workload seed, each arm:

  1. waits until the GPUs are free (no compute process, no busy unit such as `tune-*`);
  2. starts `<exe> <artifact> --host H --port P --model-id ID <profile> --request-log-jsonl ...`
     as the transient user unit `<prefix>-<arm>` (systemd-run --user --collect) and waits for
     /health;
  3. runs runner.ArmClient over the seed's plan (one priming request first, excluded);
  4. stops the unit, and writes arm.json (wall time, failures, load time, build identity,
     any foreign GPU process seen while it ran).

Arm order alternates by seed (ABBA): the first seed runs A then B, the second B then A, and so
on, so a drift over the session (clocks, temperature) does not favour one arm. After each seed
analyze.py writes report.md / summary.json into <out>/seed-<n>; with several seeds it adds a
combined report in <out>.

More than two arms, or arms that differ in their flags or in the client's max_tokens, come from
--arms-file: a JSON list of {"name", "exe", "label", "flags" (full flag string; default the
profile plus --extra-flags), "agent_max_tokens" (default --agent-max-tokens)}, baseline first.
The order then alternates forward / reversed over the seeds, or follows --arm-order.

Usage: python3 linux.py --arm-a BIN --arm-b BIN --model ARTIFACT [--profile NAME | --flags STR]
                        [--seeds 42,43,44] [--scale 1.0] [--out DIR] [--dry-run] [...]
       python3 linux.py --arms-file ARMS.json --model ARTIFACT [--arm-order x,y] [...]
run_linux.sh is a thin wrapper that checks the corpus and calls this file.

Written for ValerioDolci/ninfer-tp2 around the suite from Wallawalla47/ninfer-custom (commit
535fe587); Apache-2.0 like the rest of the tree.
"""
import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import threading
import time
import urllib.request

import analyze
import runner
import workload

# The production launch line (llama-swap `qwen27b`) plus --lm-head-draft, as the benchmarks run it.
PROD = ("--tp 2 --devices 0,1 --kv-dtype int8 --max-context 196608 --kv-capacity 196608 "
        "--device-state-slots 4 --max-concurrency 1 --spec mtp --draft-tokens 3 --vision "
        "--vision-device 0 --max-vision-tokens 4096 --lm-head-draft")
PROFILES = {
    # One lane, as production: every request queues behind the one decoding.
    "prod": PROD,
    # Four lanes and eight Device checkpoint slots, so subagents and sessions can decode together
    # when their reservations fit the shared KV pool.
    "agentic": PROD.replace("--device-state-slots 4", "--device-state-slots 8")
                   .replace("--max-concurrency 1", "--max-concurrency 4"),
}
# Always added unless the profile sets it: the default 30 s admission timeout would reject the
# requests that queue behind a long decode, and the workload keeps up to eight in flight.
HARNESS_FLAGS = [("--pending-timeout-ms", "3600000")]


def parse_flags(text):
    """[(flag, value|None), ...] from a flag string, as runner.parse_bat reads a bat line."""
    toks = shlex.split(text)
    flags, i = [], 0
    while i < len(toks):
        name, val = toks[i], None
        if not name.startswith("--"):
            raise SystemExit("flag profile: expected a --flag, got %r" % name)
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            val = toks[i + 1]
            i += 1
        flags.append((name, val))
        i += 1
    return flags


def flat(flags):
    out = []
    for n, v in flags:
        out += [n] if v is None else [n, v]
    return out


def sh(args, timeout=60):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------------------
# GPU gate
# ---------------------------------------------------------------------------------------

def gpu_apps():
    """[(pid, name)] of the CUDA compute processes on every GPU, or None if nvidia-smi fails."""
    try:
        out = sh(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], 30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    apps = []
    for ln in out.stdout.splitlines():
        if ln.strip():
            pid, _, name = ln.partition(",")
            apps.append((pid.strip(), name.strip()))
    return apps


def busy_units(patterns):
    if not patterns:
        return []
    out = sh(["systemctl", "--user", "list-units", "--type=service", "--no-legend", "--plain",
              "--state=active,activating,reloading"])
    return [ln.split()[0] for ln in out.stdout.splitlines()
            if ln.strip() and any(p in ln.split()[0] for p in patterns)]


def gpu_state():
    try:
        out = sh(["nvidia-smi", "--query-gpu=index,memory.used,clocks.sm,clocks.max.sm,"
                  "temperature.gpu,power.draw", "--format=csv,noheader"], 30)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return None


def wait_gpu_free(patterns, max_wait_s, poll_s=120):
    """Blocks until no CUDA process runs and no unit matching `patterns` is loaded."""
    t0 = time.time()
    while True:
        apps, units = gpu_apps(), busy_units(patterns)
        if apps == [] and not units:
            if time.time() - t0 > 1:
                runner.log("GPU free after %.0f min of waiting" % ((time.time() - t0) / 60))
            return
        if time.time() - t0 > max_wait_s:
            raise SystemExit("GPU still busy after %.0f min: apps %s, units %s"
                             % (max_wait_s / 60, apps, units))
        runner.log("GPU busy (apps %s, units %s); waiting %d s" % (apps, units, poll_s))
        time.sleep(poll_s)


class ForeignGpuMonitor:
    """Samples the compute processes while an arm runs and keeps any that is not its serve."""

    def __init__(self, own_pid, period_s=30):
        self.own_pid, self.period_s = str(own_pid), period_s
        self.seen = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.period_s):
            for pid, name in gpu_apps() or []:
                if pid != self.own_pid and pid not in self.seen:
                    self.seen[pid] = {"pid": pid, "name": name, "first_seen": time.time()}
                    runner.log("WARNING: foreign GPU process %s %s during the arm" % (pid, name))

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        return list(self.seen.values())


# ---------------------------------------------------------------------------------------
# Serve under systemd
# ---------------------------------------------------------------------------------------

def build_identity(exe):
    """sha256 of the executable plus `git describe` of the checkout it was built in."""
    h = hashlib.sha256()
    with open(exe, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    ident = {"exe": exe, "sha256": h.hexdigest(), "mtime": os.path.getmtime(exe)}
    d = os.path.dirname(os.path.realpath(exe))
    for _ in range(4):
        d = os.path.dirname(d)
        r = sh(["git", "-C", d, "describe", "--tags", "--always", "--dirty"])
        if r.returncode == 0:
            ident["source"] = d
            ident["describe"] = r.stdout.strip()
            ident["commit"] = sh(["git", "-C", d, "rev-parse", "--short=8", "HEAD"]).stdout.strip()
            break
    return ident


class SystemdServe:
    def __init__(self, unit, exe, model, flags, arm_dir, host, port, model_id, env):
        self.unit, self.exe, self.model, self.flags = unit, exe, model, flags
        self.arm_dir, self.host, self.port, self.model_id_arg = arm_dir, host, port, model_id
        self.env = env
        self.request_log = os.path.join(arm_dir, "request_log.jsonl")
        self.serve_log = os.path.join(arm_dir, "serve.log")
        self.model_id = model_id
        self.pid = None
        self.load_seconds = None

    def command(self):
        return ([self.exe, self.model, "--host", self.host, "--port", str(self.port),
                 "--model-id", self.model_id_arg] + flat(self.flags) +
                ["--request-log-jsonl", self.request_log])

    def active(self):
        return sh(["systemctl", "--user", "is-active", "--quiet", self.unit]).returncode == 0

    def port_open(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False

    def tail(self, n=5):
        try:
            with open(self.serve_log, encoding="utf-8", errors="replace") as f:
                return " | ".join(ln.strip() for ln in f.readlines()[-n:])
        except OSError:
            return "(no serve log)"

    def start(self, load_timeout):
        if self.active():
            raise SystemExit("unit %s is already active; stop it first" % self.unit)
        if self.port_open():
            raise SystemExit("port %d is already in use; stop the other server first" % self.port)
        for p in (self.request_log, self.serve_log):
            if os.path.exists(p):
                os.remove(p)
        args = ["systemd-run", "--user", "--unit=" + self.unit, "--collect",
                "-p", "WorkingDirectory=" + self.arm_dir,
                "-p", "StandardOutput=append:" + self.serve_log,
                "-p", "StandardError=append:" + self.serve_log]
        for k, v in self.env:
            args += ["-E", "%s=%s" % (k, v)]
        args += self.command()
        runner.log("launch %s: %s" % (self.unit, " ".join(shlex.quote(a) for a in self.command())))
        t0 = time.time()
        r = sh(args)
        if r.returncode != 0:
            raise RuntimeError("systemd-run failed: %s" % (r.stderr or r.stdout).strip())
        url = "http://%s:%d" % (self.host, self.port)
        deadline = t0 + load_timeout
        while time.time() < deadline:
            if not self.active():
                raise RuntimeError("serve exited during startup; %s: %s"
                                   % (self.serve_log, self.tail()))
            try:
                with urllib.request.urlopen(url + "/health", timeout=3) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            self.stop()
            raise RuntimeError("no /health within %d s; %s: %s"
                               % (load_timeout, self.serve_log, self.tail()))
        self.load_seconds = time.time() - t0
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self.model_id = data["data"][0]["id"]
        except Exception:
            pass
        self.pid = sh(["systemctl", "--user", "show", "-p", "MainPID", "--value",
                       self.unit]).stdout.strip()
        runner.log("serve ready in %.1f s (unit %s, pid %s, model %s)"
                   % (self.load_seconds, self.unit, self.pid, self.model_id))
        return self

    def stop(self):
        sh(["systemctl", "--user", "stop", self.unit], timeout=180)
        for _ in range(120):
            if not self.active():
                break
            time.sleep(1)
        sh(["systemctl", "--user", "reset-failed", self.unit])
        # The CUDA contexts go away a moment after the process: wait for the serve's pid.
        for _ in range(60):
            apps = gpu_apps() or []
            if not any(pid == self.pid for pid, _ in apps):
                break
            time.sleep(1)
        runner.log("serve stopped (unit %s)" % self.unit)


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------

def arm_complete(arm_dir):
    p = os.path.join(arm_dir, "arm.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        meta = json.load(f)
    return meta if meta.get("complete") else None


def run_arm(arm, a, plan, run_dir, opts):
    arm_dir = os.path.join(run_dir, arm)
    os.makedirs(arm_dir, exist_ok=True)
    if opts.resume:
        meta = arm_complete(arm_dir)
        if meta:
            runner.log("=== arm %s (%s): already complete, skipped (--resume)" % (arm, a["label"]))
            return meta["failures"]
    for name in ("client.jsonl", "arm.json"):
        p = os.path.join(arm_dir, name)
        if os.path.exists(p):
            os.remove(p)
    if not opts.skip_gpu_check:
        wait_gpu_free(opts.wait_units, opts.gpu_wait_s)
    runner.log("=== arm %s (%s): %s, max-context %d, agent max_tokens %d"
               % (arm, a["label"], a["exe"], a["ctx"], a["agent_max_tokens"]))
    # The client reads the module-level cap on every agent turn; arms run one at a time.
    runner.AGENT_MAX_TOKENS = a["agent_max_tokens"]
    serve = SystemdServe("%s-%s" % (opts.unit_prefix, arm), a["exe"], opts.model, a["flags"],
                         arm_dir, opts.host, opts.port, opts.model_id, opts.env)
    meta = {"arm": arm, "label": a["label"], "exe": a["exe"], "build": a["build"],
            "flags": a["flags"], "max_context": a["ctx"],
            "agent_max_tokens": a["agent_max_tokens"], "unit": serve.unit,
            "gpu_before": gpu_state(),
            "started": time.time(), "complete": False}
    wall, client, foreign = None, None, []
    serve.start(opts.load_timeout)
    monitor = ForeignGpuMonitor(serve.pid).start()
    try:
        client = runner.ArmClient(arm, plan, serve.model_id, arm_dir, a["ctx"])
        # One trivial request primes the arm (CUDA graphs, allocator); it carries no workload
        # seed, so the analysis never joins it.
        client.post([{"role": "user", "content": "Reply with the single word: ok"}], [], 8, 1)
        wall = client.run()
        runner.log("=== arm %s: workload finished in %.1f min (%d client failures)"
                   % (arm, wall / 60, len(client.failures)))
    finally:
        foreign = monitor.stop()
        meta["gpu_after"] = gpu_state()
        meta["serve_alive_at_end"] = serve.active()
        if not meta["serve_alive_at_end"]:
            runner.log("WARNING: the serve of arm %s was no longer running; %s: %s"
                       % (arm, serve.serve_log, serve.tail()))
        serve.stop()
        meta.update({"wall_seconds": wall, "load_seconds": serve.load_seconds,
                     "model_id": serve.model_id, "finished": time.time(),
                     "failures": client.failures if client else [("arm", "did not run")],
                     "foreign_gpu_processes": foreign, "complete": wall is not None})
        with open(os.path.join(arm_dir, "arm.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    return meta["failures"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm-a", help="ninfer-serve of arm A (the baseline)")
    ap.add_argument("--arm-b", help="ninfer-serve of arm B (compared with A)")
    ap.add_argument("--arms-file",
                    help="JSON list of arms, baseline first: {name, exe, label, flags, "
                         "agent_max_tokens}; replaces --arm-a/--arm-b")
    ap.add_argument("--arm-order",
                    help="comma-separated arm names to run, in this order, on every seed "
                         "(default: all arms, order per --order)")
    ap.add_argument("--agent-max-tokens", type=int, default=runner.AGENT_MAX_TOKENS,
                    help="max_tokens the client sends on agent turns (default %d, or "
                         "AB_AGENT_MAX_TOKENS)" % runner.AGENT_MAX_TOKENS)
    ap.add_argument("--keep-going", action="store_true",
                    help="an arm that fails to start or run is recorded in its arm.json and the "
                         "run goes on with the next arm")
    ap.add_argument("--label-a", help="arm A's name in the report (default: git describe)")
    ap.add_argument("--label-b", help="arm B's name in the report (default: git describe)")
    ap.add_argument("--model", required=True, help=".ninfer artifact served by both arms")
    ap.add_argument("--profile", default="agentic", choices=sorted(PROFILES),
                    help="named flag profile (default: agentic)")
    ap.add_argument("--flags", help="explicit flag string for both arms (replaces --profile)")
    ap.add_argument("--extra-flags", default="", help="flags appended to the profile")
    ap.add_argument("--seeds", default="42,43,44", help="comma-separated workload seeds")
    ap.add_argument("--order", default="abba", choices=("abba", "ab"),
                    help="abba: arms forward on even seed positions, reversed on odd (default); "
                         "ab: always forward")
    ap.add_argument("--scale", type=float, default=1.0, help="stretch the session loop lengths")
    ap.add_argument("--out", help="output directory (default: ./agab-<timestamp>)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--model-id", default="qwen27b")
    ap.add_argument("--unit-prefix", default="agab", help="units are <prefix>-<arm name>")
    ap.add_argument("--wait-units", default="tune-",
                    help="comma-separated user-unit name fragments that mean the GPU is taken")
    ap.add_argument("--gpu-wait-s", type=int, default=3 * 3600,
                    help="longest wait for a free GPU before an arm (default 3 h)")
    ap.add_argument("--load-timeout", type=int, default=900, help="seconds to wait for /health")
    ap.add_argument("--request-timeout", type=int, default=3600,
                    help="client socket timeout per request in seconds (default 3600)")
    ap.add_argument("--skip-gpu-check", action="store_true",
                    help="start without waiting for free GPUs (only for a dry run with a mock serve)")
    ap.add_argument("--resume", action="store_true",
                    help="skip arms whose arm.json is complete (continue an interrupted run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plans, flags, order and commands; start nothing")
    opts = ap.parse_args(argv)

    seeds = [int(s) for s in opts.seeds.split(",") if s.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds needs distinct integers")
    if opts.arms_file:
        with open(opts.arms_file, encoding="utf-8") as f:
            specs = json.load(f)
    elif opts.arm_a and opts.arm_b:
        specs = [{"name": "a", "exe": opts.arm_a, "label": opts.label_a},
                 {"name": "b", "exe": opts.arm_b, "label": opts.label_b}]
    else:
        raise SystemExit("give --arm-a and --arm-b, or --arms-file")
    names = [sp["name"] for sp in specs]
    if len(names) < 2 or len(set(names)) != len(names):
        raise SystemExit("need at least two arms with distinct names")
    for sp in specs:
        if not os.access(sp["exe"], os.X_OK):
            raise SystemExit("not an executable: %s" % sp["exe"])
    if not os.path.exists(opts.model):
        raise SystemExit("missing artifact: %s" % opts.model)
    opts.wait_units = [p for p in opts.wait_units.split(",") if p]
    cuda_home = os.environ.get("CUDA_HOME") or ("/usr/local/cuda-13.1"
                                                if os.path.isdir("/usr/local/cuda-13.1") else None)
    opts.env = [("CUDA_HOME", cuda_home)] if cuda_home else []

    def arm_flags(text):
        flags = parse_flags(text)
        for n, v in HARNESS_FLAGS:
            if n not in dict(flags):
                flags.append((n, v))
        for n in ("--host", "--port", "--model-id", "--request-log-jsonl"):
            if n in dict(flags):
                raise SystemExit("%s is set by the launcher; remove it from the flags" % n)
        if not int(dict(flags).get("--max-context") or 0):
            raise SystemExit("every arm's flags need an explicit --max-context")
        return flags

    default_flags = (opts.flags if opts.flags else PROFILES[opts.profile]) + " " + opts.extra_flags

    out_dir = os.path.abspath(opts.out or "agab-%s" % time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    runner._log_path = os.path.join(out_dir, "runner.log")
    runner.HOST, runner.PORT = opts.host, opts.port
    # A request queued behind a long decode receives nothing until it is admitted; wait for it
    # as long as the serve's own admission timeout (HARNESS_FLAGS) instead of runner.py's 20 min.
    runner.REQUEST_TIMEOUT_S = opts.request_timeout

    arms = {}
    for sp in specs:
        exe = sp["exe"]
        build = build_identity(exe)
        flags = arm_flags(sp.get("flags") or default_flags)
        arms[sp["name"]] = {
            "exe": exe, "build": build, "flags": flags,
            "ctx": int(dict(flags)["--max-context"]),
            "agent_max_tokens": int(sp.get("agent_max_tokens") or opts.agent_max_tokens),
            "label": sp.get("label") or build.get("describe") or os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(exe))))}
    if opts.arm_order:
        fixed = [n for n in opts.arm_order.split(",") if n]
        unknown = [n for n in fixed if n not in arms]
        if unknown or len(set(fixed)) != len(fixed):
            raise SystemExit("--arm-order: unknown or repeated arms %s" % (unknown or fixed))
        orders = [list(fixed) for _ in seeds]
    else:
        orders = [list(names) if (opts.order == "ab" or i % 2 == 0) else list(reversed(names))
                  for i in range(len(seeds))]
    base = names[0]

    plans = {seed: workload.build_plan(seed=seed, scale=opts.scale) for seed in seeds}
    runner.log("model: %s" % opts.model)
    for arm in names:
        a = arms[arm]
        runner.log("arm %s = %s: %s (sha256 %s), max-context %d, agent max_tokens %d"
                   % (arm, a["label"], a["exe"], a["build"]["sha256"][:12], a["ctx"],
                      a["agent_max_tokens"]))
        runner.log("flags (%s): %s" % (arm, runner.flag_str(a["flags"])))
    runner.log("seeds %s, order %s, scale %.2f, port %d"
               % (seeds, " / ".join(",".join(o) for o in orders), opts.scale, opts.port))
    if opts.dry_run:
        for seed in seeds:
            print("seed %d:\n%s" % (seed, workload.summarize(plans[seed])))
        for arm in names:
            s = SystemdServe("%s-%s" % (opts.unit_prefix, arm), arms[arm]["exe"], opts.model,
                             arms[arm]["flags"], os.path.join(out_dir, "seed-%d" % seeds[0], arm),
                             opts.host, opts.port, opts.model_id, opts.env)
            print("command (arm %s, seed %d, agent max_tokens %d): %s"
                  % (arm, seeds[0], arms[arm]["agent_max_tokens"],
                     " ".join(map(shlex.quote, s.command()))))
        print("GPU now: apps %s, busy units %s" % (gpu_apps(), busy_units(opts.wait_units)))
        return 0

    two = not opts.arms_file
    config = {"harness": "linux", "arms": list(names),
              "labels": {k: ("%s: %s" % (k.upper(), v["label"]) if two else v["label"])
                         for k, v in arms.items()},
              "short": {k: (k.upper() if two else k) for k in arms},
              "exes": {k: v["exe"] for k, v in arms.items()},
              "builds": {k: v["build"] for k, v in arms.items()},
              "model": opts.model,
              "profile": None if (opts.flags or opts.arms_file) else opts.profile,
              # flags / max_context / agent_max_tokens: the baseline's, as the analyzer's
              # header reads them; the *_by_arm and <arm>_flags keys hold every arm's.
              "flags": arms[base]["flags"], "max_context": arms[base]["ctx"],
              "agent_max_tokens": arms[base]["agent_max_tokens"],
              "max_context_by_arm": {k: v["ctx"] for k, v in arms.items()},
              "agent_max_tokens_by_arm": {k: v["agent_max_tokens"] for k, v in arms.items()},
              "scale": opts.scale, "sampling": runner.SAMPLING, "port": opts.port,
              "seeds": seeds, "order": opts.arm_order or opts.order}
    for k, v in arms.items():
        config[k + "_flags"] = v["flags"]
    t0 = time.time()
    failures, run_dirs = [], []
    for seed, order in zip(seeds, orders):
        run_dir = os.path.join(out_dir, "seed-%d" % seed)
        os.makedirs(run_dir, exist_ok=True)
        plan = plans[seed]
        with open(os.path.join(run_dir, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f)
        runner.log("=== seed %d (%s): %s" % (seed, ",".join(order),
                                              workload.summarize(plan).splitlines()[-1]))
        cfg = dict(config, seed=seed, corpus_commit=plan["corpus_commit"], arm_order=order)
        ts = time.time()
        for arm in order:
            if not opts.keep_going:
                failures += run_arm(arm, arms[arm], plan, run_dir, opts)
                continue
            try:
                failures += run_arm(arm, arms[arm], plan, run_dir, opts)
            except (Exception, SystemExit) as e:
                runner.log("=== arm %s FAILED: %s: %s" % (arm, type(e).__name__, e))
                failures.append((arm, "arm failed: %s" % e))
                p = os.path.join(run_dir, arm, "arm.json")
                if not os.path.exists(p):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump({"arm": arm, "label": arms[arm]["label"], "complete": False,
                                   "error": "%s: %s" % (type(e).__name__, e),
                                   "flags": arms[arm]["flags"], "finished": time.time()},
                                  f, indent=1)
        cfg["total_seconds"] = time.time() - ts
        # A seed run in several invocations (--arm-order, --resume) keeps the arms and times of
        # the earlier ones.
        cfg_path = os.path.join(run_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("arms") == cfg["arms"]:
                cfg["arm_order"] = ([a for a in old.get("arm_order") or [] if a not in order]
                                    + order)
                cfg["total_seconds"] += old.get("total_seconds") or 0
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        runner.log("=== seed %d: arms %s finished in %.1f min"
                   % (seed, ",".join(order), (time.time() - ts) / 60))
        done = [a for a in names if arm_complete(os.path.join(run_dir, a))]
        if len(done) >= 2 and done[0] == base:
            try:
                analyze.main([run_dir])
            except Exception as e:
                runner.log("analysis of %s failed: %s: %s" % (run_dir, type(e).__name__, e))
        else:
            runner.log("seed %d: analysis waits for the baseline and one more arm (have %s)"
                       % (seed, done))
        run_dirs.append(run_dir)
    runner.log("all seeds finished in %.1f min" % ((time.time() - t0) / 60))
    if len(run_dirs) > 1:
        try:
            analyze.aggregate(out_dir, run_dirs)
        except (Exception, SystemExit) as e:
            runner.log("aggregate report failed: %s: %s" % (type(e).__name__, e))
    if failures:
        runner.log("RUN INVALID: %d client request failure(s): %s" % (len(failures), failures[:5]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
