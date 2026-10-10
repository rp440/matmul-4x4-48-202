#!/usr/bin/env python3
"""
Check that 4x4x4_48_204-16ALT_P_84ops.slp computes the same 16x48 linear map
as the published PLinOpt file data/4x4x4_48_204-16ALT_P.slp, and count operations.

Usage:
    python3 check_alt_core.py                 # download the published file
    python3 check_alt_core.py PATH_TO_SLP     # use a local copy instead

Standard library only (Python 3.8+). All arithmetic is exact (fractions.Fraction).

Cost model: each binary +/- is 1, each multiplication by a constant other than
+1/-1 is 1, a unary leading minus is 1, plain copies (o3:=v7;) are free.
"""
import os
import re
import sys
import urllib.request
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = os.path.join(HERE, "4x4x4_48_204-16ALT_P_84ops.slp")
REPO = "jgdumas/plinopt"
SHORT = "52b9617"
PATH = "data/4x4x4_48_204-16ALT_P.slp"
RAW = "https://raw.githubusercontent.com/%s/%s/%s"

TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([A-Za-z_]\w*)|(.))")


def tokenize(s):
    out, pos = [], 0
    s = s.strip()
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            break
        pos = m.end()
        if m.group(1):
            out.append(("num", Fraction(m.group(1))))
        elif m.group(2):
            out.append(("name", m.group(2)))
        elif m.group(3).strip():
            out.append(("op", m.group(3)))
    return out


class Parser:
    """Evaluates one expression into either a constant or a linear form.
    A value is ('c', Fraction) or ('l', {input_name: Fraction}).
    self.ops accumulates the operation count."""

    def __init__(self, tokens, env):
        self.t, self.i, self.env, self.ops = tokens, 0, env, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self):
        v = self.expr()
        if self.i != len(self.t):
            raise ValueError("trailing tokens: %r" % (self.t[self.i:],))
        return v

    def expr(self):
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            w = self.term()
            v = self.addsub(v, w, 1 if op == "+" else -1)
        return v

    def term(self):
        v = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            w = self.unary()
            if op == "/":
                if w[0] != "c":
                    raise ValueError("division by non-constant")
                w = ("c", 1 / w[1])
            v = self.mul(v, w)
        return v

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            v = self.unary()
            if v[0] == "l":
                self.ops += 1  # unary minus on a linear form costs 1
            return self.scale(v, -1)
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.atom()

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return ("c", val)
        if kind == "name":
            if val not in self.env:
                raise ValueError("undefined name %s" % val)
            return ("l", self.env[val])
        if (kind, val) == ("op", "("):
            v = self.expr()
            if self.take() != ("op", ")"):
                raise ValueError("missing )")
            return v
        raise ValueError("unexpected token %r" % ((kind, val),))

    @staticmethod
    def scale(v, c):
        if v[0] == "c":
            return ("c", v[1] * c)
        return ("l", {k: x * c for k, x in v[1].items() if x * c != 0})

    def mul(self, v, w):
        if v[0] == "c" and w[0] == "c":
            return ("c", v[1] * w[1])
        if v[0] == "l" and w[0] == "l":
            raise ValueError("product of two linear forms")
        c, lin = (v, w) if v[0] == "c" else (w, v)
        if c[1] not in (1, -1):
            self.ops += 1
        return self.scale(lin, c[1])

    def addsub(self, v, w, sign):
        if v[0] != "l" or w[0] != "l":
            raise ValueError("adding a constant")
        self.ops += 1
        r = dict(v[1])
        for k, x in w[1].items():
            r[k] = r.get(k, 0) + sign * x
            if r[k] == 0:
                del r[k]
        return ("l", r)


def parse_slp(text):
    """Return (outputs, ops): outputs maps 'oN' -> {'iM': coeff}; ops is the cost."""
    env, total = {}, 0
    n_inputs = None
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*:=\s*(.*?);?\s*$", line)
        if not m:
            raise ValueError("cannot parse line: %r" % line)
        name, rhs = m.groups()
        p = Parser(tokenize(rhs), _Env(env))
        kind, val = p.parse()
        if kind != "l":
            raise ValueError("line %r is a constant" % line)
        env[name] = val
        total += p.ops
    outs = {k: v for k, v in env.items() if re.fullmatch(r"o\d+", k)}
    return outs, total


class _Env(dict):
    """Environment that treats unknown iN names as fresh inputs."""

    def __init__(self, env):
        super().__init__()
        self.env = env

    def __contains__(self, k):
        return k in self.env or re.fullmatch(r"i\d+", k) is not None

    def __getitem__(self, k):
        if k in self.env:
            return self.env[k]
        return {k: Fraction(1)}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "check_alt_core.py"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def download():
    url = RAW % (REPO, SHORT, PATH)
    try:
        text = fetch(url)
        print("downloaded", url)
        return text
    except Exception as e:
        print("short-hash URL failed (%s); resolving full hash via GitHub API" % e)
    import json
    api = "https://api.github.com/repos/%s/commits/%s" % (REPO, SHORT)
    full = json.loads(fetch(api))["sha"]
    url = RAW % (REPO, full, PATH)
    text = fetch(url)
    print("downloaded", url)
    return text


def main():
    ref_text = open(sys.argv[1]).read() if len(sys.argv) > 1 else download()
    ref_out, ref_ops = parse_slp(ref_text)
    our_out, our_ops = parse_slp(open(OURS).read())

    assert len(ref_out) == len(our_out) == 16, (len(ref_out), len(our_out))
    assert set(ref_out) == set(our_out) == {"o%d" % k for k in range(16)}
    for k in sorted(ref_out, key=lambda s: int(s[1:])):
        assert ref_out[k] == our_out[k], "output %s differs" % k
    used = set()
    for v in our_out.values():
        used.update(v)
    assert all(int(u[1:]) < 48 for u in used)
    print("OK: identical 16x48 map; published %d ops, this file %d ops" % (ref_ops, our_ops))


if __name__ == "__main__":
    main()
