#!/usr/bin/env python3
"""Extract the D2 capacity surface from a serve/perplexity log plus its request log.

Usage: python3 d2_extract.py <serve.log> [req.jsonl]
Prints: per-rank weights/sharded/replicated/local, the graph allowance line per rank, the capacity
line, and (from the first server_start record) the memory block in MiB.
"""
import json, re, sys

def mib(x):
    return f'{x / (1024 * 1024):.1f} MiB'

def main(log_path, req_path=None):
    text = open(log_path, encoding='utf-8', errors='replace').read()
    print('== rank lines')
    for m in re.finditer(r'^.*rank (\d) \| device (\d+) \| weights ([^|]+) \| sharded ([^|]+) '
                         r'\| replicated ([^|]+) \| local ([^|]+)', text, re.M):
        print(f'  rank {m.group(1)} dev {m.group(2)}: weights {m.group(3).strip()} '
              f'sharded {m.group(4).strip()} replicated {m.group(5).strip()} local {m.group(6).strip()}')
    print('== cuda graphs')
    for m in re.finditer(r'cuda graphs \| rank (\d+): observed ([0-9.]+) MiB, allowance ([0-9.]+) MiB', text):
        print(f'  rank {m.group(1)}: observed {m.group(2)} MiB allowance {m.group(3)} MiB')
    print('== capacity')
    for m in re.finditer(r'^.*capacity \| (.*)$', text, re.M):
        print('  ' + m.group(1).strip())
    print('== tensor parallel')
    for m in re.finditer(r'^.*tensor parallel \| (.*)$', text, re.M):
        print('  ' + m.group(1).strip()[:200])
    if req_path:
        recs = [json.loads(l) for l in open(req_path, encoding='utf-8') if l.strip()]
        starts = [r for r in recs if r.get('event') == 'server_start']
        if starts:
            mem = starts[0].get('memory') or {}
            print('== memory block (rank 0 plan)')
            for key in ('weights', 'sequence', 'workspace'):
                d = mem.get(key) or {}
                if d:
                    print(f'  {key}: capacity {mib(d.get("capacity_bytes", 0))} '
                          f'used {mib(d.get("used_bytes", 0))} peak {mib(d.get("peak_bytes", 0))}')
            for key in ('available_after_startup_bytes', 'runtime_reservation_bytes',
                        'cuda_graph_allowance_bytes', 'kv_payload_bytes',
                        'kv_capacity_headroom_bytes', 'planned_slack_bytes'):
                if key in mem:
                    print(f'  {key}: {mib(mem[key])}')
        done = [r for r in recs if r.get('event') == 'request_done']
        print(f'== requests: {len(done)}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
