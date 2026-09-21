#!/usr/bin/env python3
"""Evaluate and choose AMD() signature templates against labelled calls.

Replays each call the way apps/app_amd.c runs it: the signature matcher of
apps/amd/signature.c and AMD's own state machine, taken from amd_replay.py, see
the same 20ms frames, and the matcher goes first on each frame. A call is
therefore SCREENED only if a template matches before AMD reaches a verdict of
its own, and the detections, false positives and lead over AMD's verdict are
what AMD() would report on the same audio.

The matcher is mirrored in full: onset, template length, a match window tried
at every offset step within the template, the per window energy gate, the
silence threshold over the window being scored, rounding of the score to a
percentage, and the two consecutive hops a match must hold for.

Apple changes these prompts, and a stale reference set fails silently as a
MACHINE verdict, so this is worth re-running as calls are labelled.

The filterbank here is an FFT convolution with each biquad's impulse response,
which is far faster over hundreds of files than the sample-by-sample recursion
in apps/amd/signature.c but numerically identical to it, and the scorer is a
matrix form of the per hop loop. --self-test asserts both, so this script
cannot silently drift from the module.

A reference cut from a call of the corpus matches that call perfectly, which
says nothing about the calls it has not heard. Such a reference is recognised,
the whole of it reappearing in the recording, and not scored on that call.

A corpus is a directory of <call>.wav, 8 kHz mono 16-bit, each with a
<call>.json sidecar carrying a "label". Requires numpy.

Usage:
  amd_signature_sweep.py --corpus DIR --templates ref1.wav ref2.wav ...
  amd_signature_sweep.py --corpus DIR               # choose a set from the corpus
  amd_signature_sweep.py --corpus DIR --config amd.conf --thresholds 94,95,96
  amd_signature_sweep.py --self-test
"""

import argparse
import collections
import glob
import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amd_replay  # noqa: E402

# Must track the SIG_* constants in apps/amd/signature.c.
FS = 8000
HOP = 80
HOP_MS = 10
NBANDS = 12
Q = 2.5
FREQ_LOW = 200.0
FREQ_HIGH = 3500.0
FLOOR_REL = 0.10
GATE = 10 ** -2.5
ONSET = 1e-3
SMOOTH = 3
DEBOUNCE = 2
NIMP = 1024

# The [signature] defaults, SIG_DEF_* in apps/amd/signature.c.
SIG_DEFAULTS = {
    "threshold": 95,
    "template_length": 3000,
    "match_window": 1200,
    "offset_step": 100,
    "silence_threshold": 256,
}

# A frame is two hops.
HOPS_PER_FRAME = amd_replay.FRAME_MS // HOP_MS

# Mean similarity over a whole reference above which it is taken to be cut
# from the recording itself. The same rendition from another call stays under 0.98.
SELF_CUT = 0.995


def biquads():
    """RBJ constant skirt bandpass sections, as sig_init_coeffs() builds them."""
    out = []
    for f0 in np.geomspace(FREQ_LOW, FREQ_HIGH, NBANDS):
        w0 = 2 * np.pi * f0 / FS
        alpha = np.sin(w0) / (2 * Q)
        a0 = 1 + alpha
        out.append((alpha / a0, -alpha / a0, -2 * np.cos(w0) / a0, (1 - alpha) / a0))
    return out


COEFFS = biquads()


def impulse_responses():
    imp = np.zeros((NBANDS, NIMP))
    for j, (b0, b2, a1, a2) in enumerate(COEFFS):
        x1 = x2 = y1 = y2 = 0.0
        for i in range(NIMP):
            x = 1.0 if i == 0 else 0.0
            y = b0 * x + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, x, y1, y
            imp[j, i] = y
    return imp


IMP = impulse_responses()
_IMPF = {}


def band_powers_exact(x):
    """The recursion in sig_fe_feed(), for --self-test only."""
    nframes = len(x) // HOP
    p = np.zeros((nframes, NBANDS))
    for j, (b0, b2, a1, a2) in enumerate(COEFFS):
        x1 = x2 = y1 = y2 = 0.0
        acc = 0.0
        k = 0
        for i in range(len(x)):
            xi = x[i]
            y = b0 * xi + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, xi, y1, y
            acc += y * y
            if (i + 1) % HOP == 0:
                if k < nframes:
                    p[k, j] = acc / HOP
                acc = 0.0
                k += 1
    return p


def band_powers(x):
    n = len(x)
    nframes = n // HOP
    if nframes < 1:
        return None
    size = 1 << int(np.ceil(np.log2(n + NIMP)))
    if size not in _IMPF:
        _IMPF[size] = np.fft.rfft(IMP, size, axis=1)
    y = np.fft.irfft(np.fft.rfft(x, size)[None, :] * _IMPF[size], size, axis=1)[:, :n]
    return (y * y)[:, : nframes * HOP].reshape(NBANDS, nframes, HOP).mean(2).T


def features(x):
    """Per hop: the unit spectral shape and total power sig_fe_emit() produces,
    and the mean absolute sample value the silence threshold is tested on."""
    p = band_powers(x)
    if p is None:
        return None, None, None
    cs = np.cumsum(np.vstack([np.zeros(NBANDS), p]), 0)
    smoothed = np.empty_like(p)
    for k in range(len(p)):
        lo = max(0, k - SMOOTH + 1)
        smoothed[k] = (cs[k + 1] - cs[lo]) / (k + 1 - lo)
    total = smoothed.sum(1, keepdims=True)
    v = np.log(smoothed + FLOOR_REL * total + 1e-9)
    v -= v.mean(1, keepdims=True)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1
    level = np.abs(x[: len(p) * HOP]).reshape(len(p), HOP).mean(1)
    return v / norm, total[:, 0], level


class Template:
    """A reference cut into windows, as sig_template_build() does it."""

    def __init__(self, name, x, sig):
        self.name = name
        v, total, _ = features(x)
        if v is None:
            raise ValueError("%s: no audio" % name)
        self.onset = int(np.argmax(total >= total.max() * ONSET))
        self.win = sig["match_window"] // HOP_MS
        step = max(1, sig["offset_step"] // HOP_MS)
        length = min(sig["template_length"] // HOP_MS, len(v) - self.onset)
        if length < self.win:
            raise ValueError("%s: %dms of audio after silence, need %dms"
                             % (name, length * HOP_MS, sig["match_window"]))
        self.frames = v[self.onset: self.onset + length]
        energy = total[self.onset: self.onset + length]
        self.windows = []                   # (offset, gated frames, active count)
        for base in range(0, length - self.win + 1, step):
            seg = energy[base: base + self.win]
            act = seg >= seg.max() * GATE
            if act.any():
                self.windows.append((base, act, int(act.sum())))


def to_percent(s):
    """(int) (100.0f * sum / active + 0.5f), clamped to 0..100."""
    return np.clip(np.trunc(100.0 * s + 0.5), 0, 100).astype(int)


def score_hops(tmpl, v):
    """The score sig_detector_frame() computes for this template at each hop.

    Entry s is the hop at which the ring holds frames s .. s + win - 1, so the
    hop count is s + win. The best window wins. The sum over a window of
    dot(call frame s + k, template frame offset + k) is a diagonal of the
    matrix of every call frame against every template frame, read through a
    strided view rather than recomputed per offset.
    """
    win = tmpl.win
    n = len(v) - win + 1
    if n <= 0:
        return np.zeros(0, dtype=int)
    g = np.ascontiguousarray(v @ tmpl.frames.T)
    s0, s1 = g.strides
    best = np.full(n, -1.0)
    for base, act, count in tmpl.windows:
        diag = np.lib.stride_tricks.as_strided(g[:, base:], shape=(n, win), strides=(s0, s0 + s1))
        best = np.maximum(best, diag @ act / count)
    return to_percent(best)


def window_levels(level, win):
    """Mean level over the frames in the ring at each hop, as in sig_detector_frame()."""
    if len(level) < win:
        return np.zeros(0)
    return np.convolve(level, np.ones(win) / win, "valid")


def first_hit(scores, loud, threshold):
    """Index of the first hop a template matches at, or None.

    A hop above the threshold counts only if the window is loud enough; a quiet
    one resets the count, as does one below the threshold.
    """
    ok = (scores >= threshold) & loud
    run = np.convolve(ok.astype(int), np.ones(DEBOUNCE, dtype=int), "full")[: len(ok)]
    idx = np.flatnonzero(run >= DEBOUNCE)
    return int(idx[0]) if len(idx) else None


def read_signature_config(path):
    """The [signature] keys of an amd.conf, other than templates."""
    sig = dict(SIG_DEFAULTS)
    if not path or not os.path.exists(path):
        return sig
    section = None
    for line in open(path):
        line = line.split(";")[0].strip()
        if line.startswith("["):
            section = line[1:line.find("]")].strip().lower()
            continue
        if section != "signature" or "=" not in line:
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        if key in sig:
            try:
                sig[key] = int(value)
            except ValueError:
                pass
    return sig


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getframerate() != FS or w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit("%s: need %d Hz mono 16-bit, got %d Hz %dch %d-bit"
                     % (path, FS, w.getframerate(), w.getnchannels(), w.getsampwidth() * 8))
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)


def load_corpus(dirname, cfg, sig):
    """Read a corpus and replay AMD over each call.

    Only the audio AMD would still be listening to is kept for matching: a
    template matching after AMD has decided never reaches the dialplan.
    """
    win = sig["match_window"] // HOP_MS
    calls = []
    for sidecar in sorted(glob.glob(os.path.join(dirname, "*.json"))):
        wav = sidecar[:-5] + ".wav"
        if not os.path.exists(wav):
            continue
        with open(sidecar) as fh:
            label = json.load(fh).get("label")
        x = read_wav(wav)
        status, cause, ms = amd_replay.amd(amd_replay.frame_levels(x), cfg)
        frames = ms // amd_replay.FRAME_MS
        # AMD tests total_analysis_time before anything else in a frame, so on a
        # TOOLONG the matcher never sees the frame that timed out. On any other
        # verdict, and when the recording simply ends, it has seen that frame.
        if cause.startswith("TOOLONG") and ms >= cfg["total_analysis_time"]:
            frames -= 1
        x = x[: frames * amd_replay.FRAME_MS * FS // 1000]
        v, _, level = features(x) if len(x) >= HOP else (None, None, None)
        calls.append({
            "name": os.path.basename(wav)[:-4], "label": label, "samples": x,
            "features": v, "loud": None if v is None else window_levels(level, win) >= sig["silence_threshold"],
            "amd": (status, cause.split("-")[0], ms),
        })
    return calls


def full_features(call):
    """Features of the whole recording, not only what AMD listened to."""
    if "full" not in call:
        call["full"] = features(read_wav(call["path"]))
    return call["full"]


def self_cuts(templates, calls, positive):
    """{template index: call index} for each template cut from a positive call.

    Such a template reappears in that recording frame for frame, which a
    different rendition of the prompt never does.
    """
    found = {}
    for k, call in enumerate(calls):
        if call["label"] != positive:
            continue
        v = full_features(call)[0]
        for i, tmpl in enumerate(templates):
            L = len(tmpl.frames)
            if v is None or len(v) < L:
                continue
            g = np.ascontiguousarray(v @ tmpl.frames.T)
            s0, s1 = g.strides
            diag = np.lib.stride_tricks.as_strided(g, shape=(len(v) - L + 1, L), strides=(s0, s0 + s1))
            if diag.mean(1).max() >= SELF_CUT:
                found[i] = k
    return found


def hit_matrix(templates, calls, thresholds, mask=None):
    """For each threshold, the hop count each template first matches each call at.

    np.inf where it never does before AMD's own verdict, or where mask, a
    template by call array, is False.
    """
    out = {t: np.full((len(templates), len(calls)), np.inf) for t in thresholds}
    best = np.zeros((len(templates), len(calls)), dtype=int)
    for i, tmpl in enumerate(templates):
        for k, call in enumerate(calls):
            if call["features"] is None or (mask is not None and not mask[i, k]):
                continue
            scores = score_hops(tmpl, call["features"])
            if not len(scores):
                continue
            loud = call["loud"][: len(scores)]
            best[i, k] = scores[loud].max() if loud.any() else 0
            for t in thresholds:
                hit = first_hit(scores, loud, t)
                if hit is not None:
                    out[t][i, k] = hit + tmpl.win
    return out, best


def verdict(hops, call):
    """What AMD() returns once the matcher is folded in, and when."""
    if np.isfinite(hops):
        frame = -(-int(hops) // HOPS_PER_FRAME)
        return "SCREENED", frame * amd_replay.FRAME_MS
    status, _, ms = call["amd"]
    return status, ms


def report(templates, calls, hits, best, thresholds, report_at, positive):
    if not templates:
        print("no templates to report on")
        return
    labels = sorted({c["label"] for c in calls}, key=lambda l: (l != positive, str(l)))
    pos = [k for k, c in enumerate(calls) if c["label"] == positive]
    neg = [k for k, c in enumerate(calls) if c["label"] != positive]

    print("%8s %14s %16s %s" % ("thresh", "detected", "false positives", "latency / lead over AMD, ms"))
    for t in thresholds:
        first = hits[t].min(0)
        det = [k for k in pos if np.isfinite(first[k])]
        fp = [k for k in neg if np.isfinite(first[k])]
        lat = [verdict(first[k], calls[k])[1] for k in det]
        lead = [calls[k]["amd"][2] - verdict(first[k], calls[k])[1] for k in det]
        print("%8d %8d/%-5d %8d/%-7d %s%s" % (
            t, len(det), len(pos), len(fp), len(neg),
            "median %d, p90 %d; lead median %d, min %d" % (
                np.median(lat), np.percentile(lat, 90), np.median(lead), min(lead)) if det else "",
            "   " + ", ".join("%s %d" % kv for kv in collections.Counter(calls[k]["label"] for k in fp).items())
            if fp else ""))

    first = hits[report_at].min(0)
    who = hits[report_at].argmin(0)
    print("\n=== at threshold %d: label (row) against AMD() verdict (column) ===" % report_at)
    cols = ["SCREENED", "MACHINE", "HUMAN", "NOTSURE"]
    print("%-10s" % "label" + "".join("%10s" % c for c in cols))
    for label in labels:
        seen = collections.Counter(verdict(first[k], c)[0] for k, c in enumerate(calls) if c["label"] == label)
        print("%-10s" % label + "".join("%10d" % seen[c] for c in cols))

    print("\ntemplates, by the %s calls each matches first / at all:" % positive)
    for i, tmpl in enumerate(templates):
        firsts = sum(1 for k in pos if np.isfinite(first[k]) and who[k] == i)
        anyhit = sum(1 for k in pos if np.isfinite(hits[report_at][i, k]))
        print("  %-44s onset %4dms %3d / %3d%s" % (
            tmpl.name, tmpl.onset * HOP_MS, firsts, anyhit, "   <- never first" if not firsts else ""))

    missed = [k for k in pos if not np.isfinite(first[k])]
    if missed:
        print("\nmissed %s calls (best loud score, AMD verdict):" % positive)
        for k in missed:
            status, cause, ms = calls[k]["amd"]
            print("  %-44s %3d   %s/%s at %dms" % (calls[k]["name"], best[:, k].max(), status, cause, ms))

    if pos:
        lead = sorted((calls[k]["amd"][2] - verdict(first[k], calls[k])[1], k)
                      for k in pos if np.isfinite(first[k]))
        if lead:
            print("\nsmallest leads over AMD's own verdict (ms):")
            for ms, k in lead[:3]:
                print("  %+5d  %-44s AMD %s/%s" % (ms, calls[k]["name"], *calls[k]["amd"][:2]))

    closest = sorted(neg, key=lambda k: -best[:, k].max())[:5]
    print("\nhighest loud scores among the other calls (check these for mislabels):")
    for k in closest:
        print("  %3d  %-9s %s" % (best[:, k].max(), calls[k]["label"], calls[k]["name"]))


def greedy_cover(detects, members):
    """Fewest templates whose detections cover every positive call.

    Set cover is NP-hard, so this is the standard greedy approximation. For the
    sizes involved -- tens of calls -- it is within a template of optimal and the
    result is only a starting point for a human to trim anyway.
    """
    covered = set()
    chosen = []
    while True:
        best = (0, None, None)
        for i in members:
            reach = {k for k in members if detects[i, k]}
            gain = len(reach - covered)
            if gain > best[0]:
                best = (gain, i, reach)
        if best[1] is None:
            break
        chosen.append(best[1])
        covered |= best[2]
        if len(covered) >= len(members):
            break
    return chosen, covered


def cmd_select(calls, sig, thresholds, report_at, positive):
    """Choose a set from the corpus' own positives, and cross-validate the choice."""
    pos = [k for k, c in enumerate(calls) if c["label"] == positive]
    neg = [k for k, c in enumerate(calls) if c["label"] != positive]
    candidates, owner = [], []
    for k in pos:
        try:
            candidates.append(Template(calls[k]["name"], read_wav(calls[k]["path"]), sig))
            owner.append(k)
        except ValueError as exc:
            print("  skipped: %s" % exc)
    hits, best = hit_matrix(candidates, calls, thresholds)
    n = len(candidates)
    # detects[i, j]: candidate i catches the positive candidate j was cut from
    col = {k: j for j, k in enumerate(owner)}

    print("%8s %10s %10s %24s %s" % ("thresh", "templates", "coverage", "held-out detection", "false positives"))
    chosen_at = {}
    for t in thresholds:
        detects = np.isfinite(hits[t][:, owner])
        chosen, covered = greedy_cover(detects, list(range(n)))
        chosen_at[t] = chosen
        # Leave-one-out: choose again without the call being scored, so a call is
        # never detected by a template cut from itself.
        held = 0
        for j in range(n):
            subset, _ = greedy_cover(detects, [i for i in range(n) if i != j])
            if any(detects[i, j] for i in subset):
                held += 1
        fp = sum(1 for k in neg if any(np.isfinite(hits[t][i, k]) for i in chosen))
        print("%8d %10d %6d/%-3d %14d/%-3d (%5.1f%%) %6d/%-4d" % (
            t, len(chosen), len(covered), n, held, n, 100.0 * held / max(1, n), fp, len(neg)))

    chosen = chosen_at[report_at]
    print("\n=== chosen at threshold %d ===" % report_at)
    for i in chosen:
        reach = int(np.isfinite(hits[report_at][i, owner]).sum())
        print("  %s.wav  onset %dms  covers %d%s" % (
            candidates[i].name, candidates[i].onset * HOP_MS, reach,
            "  <- matches only itself; no held-out evidence for it" if reach <= 1 else ""))
    uncovered = [calls[k]["name"] for k in pos if k in col
                 and not any(np.isfinite(hits[report_at][i, k]) for i in chosen)]
    if uncovered:
        print("  not covered: %s" % ", ".join(uncovered))
    print()
    report([candidates[i] for i in chosen], calls,
           {t: hits[t][chosen] for t in thresholds}, best[chosen], thresholds, report_at, positive)


def cmd_templates(templates, calls, thresholds, report_at, positive):
    """Evaluate a reference set, never scoring a reference on its own call."""
    mask = np.ones((len(templates), len(calls)), dtype=bool)
    for i, k in self_cuts(templates, calls, positive).items():
        print("%s is cut from %s, so it is not scored on that call" % (templates[i].name, calls[k]["name"]))
        mask[i, k] = False
    hits, best = hit_matrix(templates, calls, thresholds, mask)
    report(templates, calls, hits, best, thresholds, report_at, positive)


def self_test():
    rng = np.random.RandomState(0)
    x = rng.randn(8000) * 3000
    exact = band_powers_exact(x)
    fast = band_powers(x)
    err = np.abs(fast - exact).max() / exact.max()
    print("FFT front end vs the recursion in apps/amd/signature.c: max relative error %.2e" % err)
    if err > 1e-9:
        sys.exit("front ends disagree; this script no longer matches the module")

    # A reference which, like speech, changes smoothly from hop to hop: harmonics
    # on a gliding pitch with a moving tilt, broken into syllables. White noise
    # would not do, because it only matches at exact alignments and so never
    # holds a match for two hops.
    tt = np.arange(24000) / FS
    f0 = 180 + 60 * np.sin(2 * np.pi * 0.8 * tt) + 40 * np.sin(2 * np.pi * 2.3 * tt)
    phase = 2 * np.pi * np.cumsum(f0) / FS
    tilt = 0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * tt)
    ref = sum(np.sin(h * phase) * (tilt if h % 2 else 1 - tilt) / h for h in range(1, 16)) * 6000
    ref *= np.sin(2 * np.pi * 3 * tt) > -0.6
    # A call which plays part of it too quietly to count, then late and in noise,
    # then something else: every branch of the per hop loop.
    call = np.concatenate([rng.randn(4000) * 20, ref[6000:18000] * 0.03 + rng.randn(12000),
                           rng.randn(8000) * 3, ref[6000:20000] * 0.3 + rng.randn(14000) * 50,
                           rng.randn(8000) * 3, rng.randn(8000) * 2000])
    sig = dict(SIG_DEFAULTS)
    tmpl = Template("ref", ref, sig)
    v, _, level = features(call)
    fast = score_hops(tmpl, v)
    loud = window_levels(level, tmpl.win) >= sig["silence_threshold"]

    # sig_detector_frame(), statement for statement
    win, direct, hits, matched = tmpl.win, [], 0, None
    for count in range(1, len(v) + 1):
        if count < win:
            continue
        ring = v[count - win: count]
        lvl = level[count - win: count].mean()
        score_best = 0
        for base, act, active in tmpl.windows:
            total = 0.0
            for k in range(win):
                if act[k]:
                    total += float(ring[k] @ tmpl.frames[base + k])
            score_best = max(score_best, int(to_percent(np.array([total / active]))[0]))
        direct.append(score_best)
        hit = score_best >= sig["threshold"]
        if lvl < sig["silence_threshold"]:
            hits = 0
            continue
        if hit:
            hits += 1
            if hits >= DEBOUNCE and matched is None:
                matched = count
        else:
            hits = 0
    delta = np.abs(fast - np.array(direct)).max()
    first = first_hit(fast, loud, sig["threshold"])
    first = None if first is None else first + win
    quiet = int(((fast >= sig["threshold"]) & ~loud).sum())
    print("matrix scorer vs the per hop loop: max score difference %d, first match at hop %s vs %s, "
          "%d quiet hops over the threshold ignored" % (delta, first, matched, quiet))
    if delta or first != matched or matched is None or not quiet:
        sys.exit("scorers disagree, or the test no longer reaches every branch")
    print("OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", metavar="DIR", help="directory of <call>.wav plus <call>.json")
    ap.add_argument("--templates", nargs="+", metavar="WAV",
                    help="reference set to evaluate; without it, one is chosen from the corpus")
    ap.add_argument("--config", default="amd.conf",
                    help="amd.conf whose [general] and [signature] to replay (default amd.conf); "
                         "the templates key is ignored, give --templates instead")
    ap.add_argument("--thresholds", help="thresholds to evaluate (default: the configured one "
                                         "and two either side)")
    ap.add_argument("--positive-label", default="SCREENED",
                    help="sidecar label marking a screened call (default SCREENED)")
    ap.add_argument("--self-test", action="store_true",
                    help="check the front end and scorer against the module's and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.corpus:
        ap.error("give --corpus, or --self-test")

    cfg = amd_replay.read_config(args.config)
    sig = read_signature_config(args.config)
    report_at = sig["threshold"]
    thresholds = sorted({int(t) for t in args.thresholds.split(",")} | {report_at}) \
        if args.thresholds else [report_at + d for d in (-2, -1, 0, 1, 2) if 0 <= report_at + d <= 100]
    print("config: %s%s" % (args.config, "" if os.path.exists(args.config) else " (not found, defaults)"))
    print("signature: %s" % " ".join("%s=%d" % kv for kv in sig.items()))

    calls = load_corpus(args.corpus, cfg, sig)
    if not calls:
        sys.exit("no <call>.wav / <call>.json pairs under %s" % args.corpus)
    for c in calls:
        c["path"] = os.path.join(args.corpus, c["name"] + ".wav")
    print("corpus: %d calls   %s\n" % (len(calls), dict(collections.Counter(c["label"] for c in calls))))

    if args.templates:
        templates = [Template(os.path.basename(p)[:-4] if p.endswith(".wav") else os.path.basename(p),
                              read_wav(p), sig) for p in args.templates]
        cmd_templates(templates, calls, thresholds, report_at, args.positive_label)
    else:
        cmd_select(calls, sig, thresholds, report_at, args.positive_label)


if __name__ == "__main__":
    main()
