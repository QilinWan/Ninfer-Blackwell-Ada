#!/usr/bin/env python3
"""check_shard_sources.py [SRC_ROOT] — guards the shards compiled from upstream's own sources.

src/ops/*/tp2/*_{shard,half}_*.{cu,cpp} include an upstream launcher or plan source after a names
header of #defines (docs/maintainer/upstream-merge.md §2.4). Two things would turn that into a
silent bug after an upstream change, and this script fails on both:

1. A template, inline, constexpr, static or kernel definition OUTSIDE an anonymous namespace in an
   included upstream source: it would be compiled with different bodies in the parent's and the
   shard's translation units under the same name, and the linker would keep only one.
2. A renamed identifier appearing in a header that the prelude does not include first: that header
   would be processed with the renames in the shard's translation unit and differ from the parent's.

Exit 0 when clean, 1 with a report otherwise."""
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..")
src = os.path.join(root, "src")
include_re = re.compile(r'^#include "([^"]+)"')
define_re = re.compile(r"^#define (\w+) (\w+)")
problems = []


def included(path):
    out = []
    for line in open(path, encoding="utf-8"):
        m = include_re.match(line.strip())
        if m:
            out.append(m.group(1))
    return out


def resolve(inc):
    for base in (src, os.path.join(root, "include")):
        p = os.path.join(base, inc)
        if os.path.exists(p):
            return p
    return None


def transitive(path, seen):
    for inc in included(path):
        p = resolve(inc)
        if p and p not in seen:
            seen.add(p)
            transitive(p, seen)
    return seen


units = []
for dirpath, _, files in os.walk(src):
    if os.path.basename(dirpath) != "tp2":
        continue
    for name in files:
        if re.search(r"_(shard|half)_\w+\.(cu|cpp)$", name):
            units.append(os.path.join(dirpath, name))

for unit in sorted(units):
    incs = included(unit)
    upstream = [resolve(i) for i in incs if re.search(r"\.(cu|cpp)$", i)]
    names_hdr = [resolve(i) for i in incs if i.endswith("_names.h")]
    if not upstream or not names_hdr:
        problems.append(f"{unit}: no upstream source or names header included")
        continue
    renamed = set()
    for h in names_hdr:
        for line in open(h, encoding="utf-8"):
            m = define_re.match(line.strip())
            if m:
                renamed.add(m.group(1))
    for line in open(unit, encoding="utf-8"):
        m = define_re.match(line.strip())
        if m:
            renamed.add(m.group(1))
    # headers processed before the renames: everything the unit includes before its names header
    pre = set()
    for inc in incs:
        if inc.endswith("_names.h"):
            break
        p = resolve(inc)
        if p:
            pre.add(p)
            transitive(p, pre)
    for up in upstream:
        lines = open(up, encoding="utf-8").read().split("\n")
        anon = 0
        for n, line in enumerate(lines, 1):
            s = line.strip()
            if s == "namespace {":
                anon += 1
            elif s == "} // namespace" and anon:
                anon -= 1
            elif anon == 0 and re.match(r"^(template\s*<|inline\b|constexpr\b|static\b|__global__|__device__)", line):
                problems.append(f"{up}:{n}: definition outside an anonymous namespace: {s[:90]}")
        for hdr in sorted(transitive(up, set()) - pre):
            text = open(hdr, encoding="utf-8").read()
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # comments are not renamed
            text = re.sub(r"//[^\n]*", "", text)
            hits = sorted(r for r in renamed if re.search(r"\b%s\b" % re.escape(r), text))
            if hits:
                problems.append(f"{unit}: {hdr} names {', '.join(hits)} but is first included after the renames")

if problems:
    print("shard sources: FAIL")
    for p in problems:
        print("  " + p)
    sys.exit(1)
print(f"shard sources: OK ({len(units)} translation units)")
