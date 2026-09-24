#!/usr/bin/env python3
"""Make, evaluate and choose AMD() signature references against labelled calls.

Replays each call the way apps/app_amd.c runs it: the signature matcher of
apps/amd/signature.c and AMD's own state machine, taken from amd_replay.py, see
the same 20ms frames, and the matcher goes first on each frame. A call is
therefore SCREENED only if a reference matches before AMD reaches a verdict of
its own, and the detections, false positives and lead over AMD's verdict are
what AMD() would report on the same audio.

The matcher is mirrored in full: onset, template length, the lifter, the
activity gate, the open-begin alignment and its scores, the silence threshold
over the last match window, and the two consecutive hops a match must hold
for. doc/amd-signature.txt describes the matcher and why it is built that way.

Apple changes these prompts, and a stale reference set fails silently as a
MACHINE verdict, so this is worth re-running as calls are labelled.

The filterbank here is an FFT convolution with each biquad's impulse response,
which is far faster over hundreds of files than the sample-by-sample recursion
in apps/amd/signature.c but numerically identical to it, and the scorer
computes a whole column of the alignment for every call at once rather than
cell by cell. --self-test asserts both, so this script cannot silently drift
from the module.

A reference cut from a call of the corpus matches that call perfectly, which
says nothing about the calls it has not heard. Such a reference is recognised,
the whole of it reappearing in the recording, and not scored on that call; a
.sig reference is not scored on the calls it was averaged from.

--average replaces each reference, the seed of a voice, by the average of
every screened call of that voice. Each such call is assigned the seed it
aligns with best, and its prompt is cut out and averaged with the seed by DTW
barycenter averaging. A screened call is only scored by an average built
without it, so the figures are those of calls the set has not heard.
--write-averages then writes each voice's average, built from every call, as
the <seed>.sig file AMD() loads from /etc/asterisk/amd/<language>/.

A corpus is a directory of <call>.wav, 8 kHz mono 16-bit, each with a
<call>.json sidecar carrying a "label". Requires numpy.

Usage:
  amd_signature_sweep.py --corpus DIR --templates ref1.wav ref2.sig ...
  amd_signature_sweep.py --corpus DIR               # choose a set from the corpus
  amd_signature_sweep.py --corpus DIR --config amd.conf --thresholds 94,95,96
  amd_signature_sweep.py --corpus DIR --templates seed1.wav ... --average --write-averages OUT
  amd_signature_sweep.py --self-test
"""

import argparse
import collections
import copy
import datetime
import glob
import json
import os
import shutil
import sys
import tempfile
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
    "silence_threshold": 256,
}

# A frame is two hops.
HOPS_PER_FRAME = amd_replay.FRAME_MS // HOP_MS

# Mean similarity over a whole reference above which it is taken to be cut
# from the recording itself. The same rendition from another call stays under 0.98.
SELF_CUT = 0.995

# Must track SIG_THETA, SIG_WARP_PENALTY and SIG_NCEPS in apps/amd/signature.c:
# the similarity a step must beat to extend a path, the cost of a step off the
# diagonal per active reference frame, and the cepstral coefficients the
# lifter keeps, c0 being zero after mean removal. A path counts once it spans
# a match window, at least half of it active.
THETA = 0.90
WARP_PENALTY = 0.20
NCEPS = 6
LIFTER = np.cos(np.pi * np.arange(1, NCEPS + 1)[:, None] * (np.arange(NBANDS) + 0.5) / NBANDS)
DBA_ITERATIONS = 10

# The .sig format, as sig_load_sig() in apps/amd/signature.c reads it.
SIG_FORMAT = "amd-signature 1"
SIG_FRONTEND = "bands=%d hop=%d q=%g low=%g high=%g floor=%.2f smooth=%d" % (
    NBANDS, HOP_MS, Q, FREQ_LOW, FREQ_HIGH, FLOOR_REL, SMOOTH)


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


def unit(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return v / norm


def lift(v):
    """Spectral shapes reduced to their first cepstral coefficients, unit length."""
    return unit(v @ LIFTER.T)


def log_power(total):
    return np.log(total + 1e-9)


class Template:
    """A reference, as sig_template_from_frames() builds it."""

    def __init__(self, name, x, sig):
        v, total, _ = features(x)
        if v is None:
            raise ValueError("%s: no audio" % name)
        self._build(name, v, total, sig)

    @classmethod
    def from_features(cls, name, v, total, sig):
        tmpl = cls.__new__(cls)
        tmpl._build(name, v, total, sig)
        return tmpl

    def _build(self, name, v, total, sig):
        self.name = name
        self.sources = []                   # corpus calls a .sig was averaged from
        self.onset = int(np.argmax(total >= total.max() * ONSET))
        self.win = sig["match_window"] // HOP_MS
        length = min(sig["template_length"] // HOP_MS, len(v) - self.onset)
        if length < self.win:
            raise ValueError("%s: %dms of audio after silence, need %dms"
                             % (name, length * HOP_MS, sig["match_window"]))
        self.frames = v[self.onset: self.onset + length]
        energy = total[self.onset: self.onset + length]
        # Frames more than 25dB below the loudest within half a window either
        # side carry no reliable shape, and are not scored.
        self.total = energy
        self.lifted = lift(self.frames)
        h = self.win // 2
        self.act = np.array([energy[i] >= energy[max(0, i - h): i + h].max() * GATE
                             for i in range(length)], dtype=float)


def write_sig(path, v, total, seed, calls, sources):
    """Write a reference as the .sig file sig_load_sig() reads.

    sources are (kind, name) pairs, kind "hand" for a recording and "call" for
    a corpus call: names of files only, never anything from a sidecar.
    """
    with open(path, "w") as fh:
        fh.write("; AMD() signature reference, made by amd_signature_sweep.py\n")
        fh.write("format: %s\n" % SIG_FORMAT)
        fh.write("frontend: %s\n" % SIG_FRONTEND)
        fh.write("seed: %s\n" % seed)
        fh.write("calls: %d\n" % calls)
        fh.write("sources: %d\n" % len(sources))
        for kind, name in sources:
            fh.write("source: %s %s\n" % (kind, name))
        fh.write("built: %s\n" % datetime.date.today().isoformat())
        fh.write("frames: %d\n" % len(v))
        for lp, row in zip(log_power(total), v):
            fh.write(" ".join("%.5g" % x for x in (lp, *row)) + "\n")


def read_sig(path):
    """(v, total, header) from a .sig file, rejecting one made for another front end."""
    header, rows = {"source": []}, []
    for lineno, line in enumerate(open(path), 1):
        line = line.split(";")[0].strip()
        if not line:
            continue
        if ":" in line:
            key, value = (s.strip() for s in line.split(":", 1))
            if key == "source":
                header["source"].append(tuple(value.split(None, 1)))
            else:
                header[key] = value
            continue
        try:
            row = [float(x) for x in line.split()]
        except ValueError:
            raise ValueError("%s:%d: not a frame" % (path, lineno))
        if len(row) != NBANDS + 1:
            raise ValueError("%s:%d: %d values, need %d" % (path, lineno, len(row), NBANDS + 1))
        rows.append(row)
    if header.get("format") != SIG_FORMAT:
        raise ValueError("%s: format '%s', need '%s'" % (path, header.get("format"), SIG_FORMAT))
    if header.get("frontend") != SIG_FRONTEND:
        raise ValueError("%s: made for front end '%s', this one is '%s'"
                         % (path, header.get("frontend"), SIG_FRONTEND))
    if not rows or ("frames" in header and int(header["frames"]) != len(rows)):
        raise ValueError("%s: %d frames, header says %s" % (path, len(rows), header.get("frames")))
    rows = np.array(rows)
    # The shapes were written to five digits: make them unit length again, as
    # the module does.
    return unit(rows[:, 1:]), np.exp(rows[:, 0]), header


def load_template(path, sig):
    """A reference from a recording or a .sig file."""
    name = os.path.splitext(os.path.basename(path))[0]
    if path.endswith(".sig"):
        v, total, header = read_sig(path)
        tmpl = Template.from_features(name, v, total, sig)
        tmpl.sources = [n for kind, n in header["source"] if kind == "call"]
        return tmpl
    return Template(name, read_wav(path), sig)


def score_local(tmpl, frames, theta=THETA, penalty=WARP_PENALTY):
    """The score sig_detector_frame() computes for this reference at each hop
    of each call in frames.

    Entry s is the hop at which the last match window of the call holds hops
    s .. s + win - 1, so the hop count is s + win: nothing is scored before a
    window of audio has been heard, although the alignment runs from the
    first hop.

    A local alignment on similarity: stepping onto reference frame i at call
    hop j gains act_i * (t_i . v_j - theta), less penalty * act_i for a step
    off the diagonal, and a path restarts wherever every way into it is
    negative. Steps are (1,1), (1,2), which skips a call hop, and (2,1), which
    covers two reference frames in one hop. Each cell carries where its path
    started in the reference and the active frames it covered, A, so its
    score theta + H / A is the mean similarity over them less the penalties.
    A hop scores the best path spanning at least a match window of which at
    least half is active: without the second condition, a reference which is
    mostly pauses matches the noise floor of any call on the few frames it
    has.

    Every way into a cell comes from an earlier hop, so a whole column is
    computed at once, for all calls together. Calls are taken longest first so
    that the ones still running are a prefix.
    """
    T, act, span = tmpl.lifted, tmpl.act, tmpl.win
    W = len(T)
    lens = np.array([len(f) for f in frames])
    order = np.argsort(-lens, kind="stable")
    C = np.zeros((len(frames), lens.max(), T.shape[1]))
    for row, k in enumerate(order):
        C[row, :lens[k]] = frames[k]
    pen = penalty * act
    idx = np.arange(W)
    shape = (len(frames), W)
    H1, H2 = np.full(shape, -np.inf), np.full(shape, -np.inf)
    A1, A2 = np.zeros(shape), np.zeros(shape)
    S1, S2 = np.zeros(shape, dtype=int), np.zeros(shape, dtype=int)
    out = np.zeros((len(frames), lens.max()))
    for j in range(lens.max()):
        m = int((lens > j).sum())
        g = act * (C[:m, j] @ T.T - theta)
        H = np.zeros((m, W))
        A = np.zeros((m, W))
        S = np.broadcast_to(idx, (m, W)).copy()
        for h, a, s, i0 in ((H1[:m, :-1], A1[:m, :-1], S1[:m, :-1], 1),
                            (H2[:m, :-1] - pen[1:], A2[:m, :-1], S2[:m, :-1], 1),
                            (H1[:m, :-2] + g[:, 1:-1] - pen[2:], A1[:m, :-2] + act[1:-1], S1[:m, :-2], 2)):
            better = h > H[:, i0:]
            H[:, i0:] = np.where(better, h, H[:, i0:])
            A[:, i0:] = np.where(better, a, A[:, i0:])
            S[:, i0:] = np.where(better, s, S[:, i0:])
        H += g
        A += act
        spans = (idx - S + 1 >= span) & (A >= 0.5 * span)
        out[:m, j] = np.where(spans, 100.0 * (theta + H / np.maximum(A, 1)), 0).max(1)
        H2[:m], A2[:m], S2[:m] = H1[:m], A1[:m], S1[:m]
        H1[:m], A1[:m], S1[:m] = H, A, S
    scores = [None] * len(frames)
    for row, k in enumerate(order):
        scores[k] = np.maximum(out[row, span - 1: lens[k]], 0)
    return scores


def _dtw(c, penalty, free_start):
    """Cumulative cost over c (reference frames by call hops) with the steps of
    score_local(), each reference frame costed once, and the step taken into
    each cell: 0 for (1,1), 1 for (1,2), 2 for (2,1)."""
    W, N = c.shape
    D = np.full((W, N), np.inf)
    step = np.zeros((W, N), dtype=np.int8)
    if free_start:
        D[0] = c[0]
    else:
        D[0, 0] = c[0, 0]
    for i in range(1, W):
        best = np.full(N, np.inf)
        best[1:] = D[i - 1, :-1]
        cand = np.full(N, np.inf)
        cand[2:] = D[i - 1, :-2] + penalty
        better = cand < best
        best[better] = cand[better]
        step[i, better] = 1
        if i >= 2:
            cand = np.full(N, np.inf)
            cand[1:] = D[i - 2, :-1] + c[i - 1, 1:] + penalty
            better = cand < best
            best[better] = cand[better]
            step[i, better] = 2
        D[i] = best + c[i]
    return D, step


def _backtrack(step, i, j):
    """Reference and call indices along the path ending at (i, j), every call
    hop and reference frame it passes through included."""
    pairs = [(i, j)]
    while i > 0:
        kind = step[i, j]
        if kind == 0:
            i, j = i - 1, j - 1
        elif kind == 1:
            pairs.append((i, j - 1))
            i, j = i - 1, j - 2
        else:
            pairs.append((i - 1, j))
            i, j = i - 2, j - 1
        pairs.append((i, j))
    pairs = np.array(pairs)
    return pairs[:, 0], pairs[:, 1]


def locate(ref, x, penalty=WARP_PENALTY):
    """Where lifted reference frames ref best fit within lifted frames x:
    (mean cost per reference frame, first hop, last hop)."""
    D, step = _dtw(1 - ref @ x.T, penalty, free_start=True)
    end = int(np.argmin(D[-1]))
    _, hops = _backtrack(step, len(ref) - 1, end)
    return D[-1, end] / len(ref), int(hops.min()), end


def average(init, members, penalty=WARP_PENALTY, iterations=DBA_ITERATIONS):
    """DTW barycenter averaging of (v, log total) sequences on init's time axis.

    Each pass aligns every member to the current average end to end and
    replaces each frame of the average by the mean of the member frames
    aligned to it: spectral shapes renormalised as features() leaves them, log
    power averaged. Alignment is on the liftered shapes.
    """
    v, logt = init
    for _ in range(iterations):
        acc_v = np.zeros_like(v)
        acc_l = np.zeros(len(v))
        count = np.zeros(len(v))
        ref = lift(v)
        for mv, ml in members:
            D, step = _dtw(1 - ref @ lift(mv).T, penalty, free_start=False)
            if not np.isfinite(D[-1, -1]):
                raise ValueError("cannot align sequences of %d and %d hops" % (len(v), len(mv)))
            i, j = _backtrack(step, len(v) - 1, len(mv) - 1)
            np.add.at(acc_v, i, mv[j])
            np.add.at(acc_l, i, ml[j])
            np.add.at(count, i, 1)
        acc_v /= count[:, None]
        v = unit(acc_v - acc_v.mean(1, keepdims=True))
        logt = acc_l / count
    return v, logt


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
    """The [signature] keys of an amd.conf that tune the matcher."""
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
                sig[key] = float(value) if key == "threshold" else int(value)
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


def call_scores(tmpl, calls):
    """Per call, the score at each hop, or None for a call too short to score."""
    live = [k for k, c in enumerate(calls) if c["features"] is not None]
    out = [None] * len(calls)
    for k, s in zip(live, score_local(tmpl, [lift(calls[k]["features"]) for k in live])):
        out[k] = s
    return out


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
    best = np.zeros((len(templates), len(calls)))
    for i, tmpl in enumerate(templates):
        per_call = call_scores(tmpl, calls)
        for k, call in enumerate(calls):
            scores = per_call[k]
            if scores is None or not len(scores) or (mask is not None and not mask[i, k]):
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
        print("%8g %8d/%-5d %8d/%-7d %s%s" % (
            t, len(det), len(pos), len(fp), len(neg),
            "median %d, p90 %d; lead median %d, min %d" % (
                np.median(lat), np.percentile(lat, 90), np.median(lead), min(lead)) if det else "",
            "   " + ", ".join("%s %d" % kv for kv in collections.Counter(calls[k]["label"] for k in fp).items())
            if fp else ""))
    if pos and neg:
        weakest = min(best[:, k].max() for k in pos)
        strongest = max(best[:, k].max() for k in neg)
        print("\nmargin %+.1f: weakest %s call scores %.1f, strongest other call %.1f" % (
            weakest - strongest, positive, weakest, strongest))

    first = hits[report_at].min(0)
    who = hits[report_at].argmin(0)
    print("\n=== at threshold %g: label (row) against AMD() verdict (column) ===" % report_at)
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
            print("  %-44s %5.1f   %s/%s at %dms" % (calls[k]["name"], best[:, k].max(), status, cause, ms))

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
        print("  %5.1f  %-9s %s" % (best[:, k].max(), calls[k]["label"], calls[k]["name"]))


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
        print("%8g %10d %6d/%-3d %14d/%-3d (%5.1f%%) %6d/%-4d" % (
            t, len(chosen), len(covered), n, held, n, 100.0 * held / max(1, n), fp, len(neg)))

    chosen = chosen_at[report_at]
    print("\n=== chosen at threshold %g ===" % report_at)
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
    """Evaluate a reference set, never scoring a reference on its own calls."""
    mask = np.ones((len(templates), len(calls)), dtype=bool)
    for i, k in self_cuts(templates, calls, positive).items():
        print("%s is cut from %s, so it is not scored on that call" % (templates[i].name, calls[k]["name"]))
        mask[i, k] = False
    index = {c["name"]: k for k, c in enumerate(calls)}
    for i, tmpl in enumerate(templates):
        heard = [index[n] for n in tmpl.sources if n in index]
        if heard:
            print("%s is averaged from %d calls of this corpus, so it is not scored on them"
                  % (tmpl.name, len(heard)))
            mask[i, heard] = False
    hits, best = hit_matrix(templates, calls, thresholds, mask)
    report(templates, calls, hits, best, thresholds, report_at, positive)


def cmd_average(refs, calls, sig, thresholds, report_at, positive, write_dir=None):
    """Average each reference with the positive calls of its voice, evaluate
    the averages leave-one-out, and write the full averages to write_dir."""
    pos = [k for k, c in enumerate(calls) if c["label"] == positive]
    cut = self_cuts(refs, calls, positive)

    # Each positive goes to the reference it aligns with best, and the stretch
    # of its recording that reference aligns to is its instance of the prompt.
    voice, inst = {}, {}
    for k in pos:
        v, total, _ = full_features(calls[k])
        if v is None:
            continue
        lv = lift(v)
        fits = [locate(r.lifted, lv) for r in refs]
        i = int(np.argmin([f[0] for f in fits]))
        _, first, last = fits[i]
        voice[k] = i
        inst[k] = (v[first:last + 1], log_power(total[first:last + 1]))

    def build(i, without=None):
        # A reference cut from a call is that call's instance already, and
        # must not be the starting point of an average built without it.
        ref = (refs[i].frames, log_power(refs[i].total))
        members = [inst[k] for k in pos if voice.get(k) == i and k != without]
        if i not in cut:
            members.append(ref)
        if not members:
            return None
        init = members[0] if without is not None and cut.get(i) == without else ref
        v, logt = average(init, members)
        return Template.from_features(refs[i].name, v, np.exp(logt), sig)

    print("averaging %d positive calls onto %d references:" % (len(voice), len(refs)))
    for i, r in enumerate(refs):
        print("  %-30s %3d calls%s" % (r.name, sum(1 for k in voice if voice[k] == i),
                                       "   (cut from %s)" % calls[cut[i]]["name"] if i in cut else ""))
    full = [build(i) for i in range(len(refs))]
    loo = {k: build(voice[k], without=k) for k in voice}

    # A positive is scored by the averages of the other voices and by its own
    # voice's built without it; every other call by all of them.
    templates, owner = [], []
    for i, t in enumerate(full):
        if t is not None:
            templates.append(t)
            owner.append(i)
    nfull = len(templates)
    held = {}
    for k, t in loo.items():
        if t is not None:
            held[k] = len(templates)
            templates.append(t)
            owner.append(voice[k])
    mask = np.ones((len(templates), len(calls)), dtype=bool)
    for k in pos:
        mask[:, k] = False
        mask[[j for j in range(nfull) if owner[j] != voice.get(k)], k] = True
        if k in held:
            mask[held[k], k] = True
    print("scoring %d averages (%d leave-one-out)\n" % (len(templates), len(held)))
    hits, best = hit_matrix(templates, calls, thresholds, mask)

    if write_dir:
        os.makedirs(write_dir, exist_ok=True)
        written = set()
        print("writing to %s:" % write_dir)
        for i, t in enumerate(full):
            if t is None:
                continue
            members = [k for k in pos if voice.get(k) == i]
            sources = ([] if i in cut else [("hand", refs[i].name)]) + [("call", calls[k]["name"]) for k in members]
            path = os.path.join(write_dir, refs[i].name + ".sig")
            write_sig(path, t.frames, t.total, refs[i].name, len(members), sources)
            written.add(os.path.basename(path))
            print("  %-30s average of %d calls%s" % (os.path.basename(path), len(members),
                                                     "" if i in cut else " and the seed"))
        stale = sorted(f for f in os.listdir(write_dir) if f.endswith(".sig") and f not in written)
        for f in stale:
            print("  %-30s not refreshed by this run: remove it if that voice is retired" % f)
        print()

    # Report one row per voice: the first match among its averages.
    rows = sorted(set(owner))
    hits = {t: np.vstack([h[[j for j in range(len(templates)) if owner[j] == i]].min(0) for i in rows])
            for t, h in hits.items()}
    best = np.vstack([best[[j for j in range(len(templates)) if owner[j] == i]].max(0) for i in rows])
    shown = []
    for i in rows:
        shown.append(copy.copy(full[i] if full[i] is not None else refs[i]))
        shown[-1].name = "average of " + refs[i].name
    report(shown, calls, hits, best, thresholds, report_at, positive)


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
    loud = window_levels(level, tmpl.win) >= sig["silence_threshold"]

    # sig_detector_frame(), cell by cell and hop by hop, against the scorer
    # which computes a column at a time, on the call and, to exercise the
    # batching of calls of unequal length, on the call and a prefix of it.
    lv = lift(v)
    T, act, win = tmpl.lifted, tmpl.act, tmpl.win
    prev1 = prev2 = [(-np.inf, 0.0, 0)] * len(T)
    direct, hits, matched = [], 0, None
    for j in range(len(lv)):
        g = act * (T @ lv[j] - THETA)
        cur, score_best = [], 0.0
        for i in range(len(T)):
            h, a, start = 0.0, 0.0, i
            ways = []
            if i >= 1:
                ways.append(prev1[i - 1])
                ways.append((prev2[i - 1][0] - WARP_PENALTY * act[i], prev2[i - 1][1], prev2[i - 1][2]))
            if i >= 2:
                ways.append((prev1[i - 2][0] + g[i - 1] - WARP_PENALTY * act[i],
                             prev1[i - 2][1] + act[i - 1], prev1[i - 2][2]))
            for way in ways:
                if way[0] > h:
                    h, a, start = way
            cur.append((h + g[i], a + act[i], start))
            if i - start + 1 >= win and 2 * (a + act[i]) >= win:
                score_best = max(score_best, 100.0 * (THETA + (h + g[i]) / (a + act[i])))
        prev1, prev2 = cur, prev1
        count = j + 1
        if count < win:
            continue
        direct.append(score_best)
        if level[count - win: count].mean() < sig["silence_threshold"]:
            hits = 0
        elif score_best >= sig["threshold"]:
            hits += 1
            if hits >= DEBOUNCE and matched is None:
                matched = count
        else:
            hits = 0
    direct = np.array(direct)
    whole, part = score_local(tmpl, [lv, lv[:len(lv) // 2]])
    delta = max(np.abs(whole - direct).max(), np.abs(part - whole[:len(part)]).max())
    first = first_hit(whole, loud, sig["threshold"])
    first = None if first is None else first + win
    quiet = int(((whole >= sig["threshold"]) & ~loud).sum())
    print("column scorer vs the cell by cell recursion: max score difference %.1e, first match at hop "
          "%s vs %s, %d quiet hops over the threshold ignored" % (delta, first, matched, quiet))
    if delta > 1e-9 or first != matched or matched is None or not quiet:
        sys.exit("scorers disagree, or the test no longer reaches every branch")

    # A reference written as a .sig and read back scores as the one it came
    # from, to the five digits written, and a .sig made for another front
    # end is refused.
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "ref.sig")
    write_sig(path, tmpl.frames, tmpl.total, "ref", 0, [("hand", "ref")])
    back = load_template(path, sig)
    again = score_local(back, [lv])[0]
    drift = np.abs(again - whole).max()
    with open(path) as fh:
        text = fh.read().replace("bands=%d" % NBANDS, "bands=%d" % (NBANDS + 1))
    with open(path, "w") as fh:
        fh.write(text)
    try:
        read_sig(path)
        refused = False
    except ValueError:
        refused = True
    shutil.rmtree(tmpdir)
    print(".sig round trip: max score difference %.1e; other front end refused: %s" % (drift, refused))
    if drift > 0.01 or not refused:
        sys.exit(".sig files do not round trip")

    # --average: the reference placed among unrelated frames is found where it
    # was put, and averaging it with itself, and with a copy played slower,
    # gives back the reference.
    # Digital silence between the syllables leaves frames with no shape at
    # all, which match nothing, themselves included.
    shaped = np.linalg.norm(tmpl.lifted, axis=1) > 0
    noise = unit(rng.randn(200, NCEPS))
    cost, begin, end = locate(tmpl.lifted, np.vstack([noise[:80], tmpl.lifted, noise[80:]]))
    ref = (tmpl.frames, log_power(tmpl.total))
    same = average(ref, [ref, ref])
    slow = np.repeat(np.arange(len(tmpl.frames)), [2 if i % 4 == 0 else 1 for i in range(len(tmpl.frames))])
    stretched = average(ref, [ref, (ref[0][slow], ref[1][slow])])
    drift = np.abs(same[0] - tmpl.frames).max()
    kept = (stretched[0] * tmpl.frames).sum(1)[shaped].mean()
    print("located at hops %d-%d (expected 80-%d), cost over shaped frames %.1e; averaged with itself: "
          "max change %.1e; with a slower copy: mean frame similarity %.5f" % (
              begin, end, 80 + len(tmpl.lifted) - 1, cost - (~shaped).mean(), drift, kept))
    if (begin, end) != (80, 80 + len(tmpl.lifted) - 1) or cost - (~shaped).mean() > 1e-9 \
            or drift > 1e-9 or kept < 0.999:
        sys.exit("alignment or averaging is broken")
    print("OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", metavar="DIR", help="directory of <call>.wav plus <call>.json")
    ap.add_argument("--templates", nargs="+", metavar="FILE",
                    help="reference set to evaluate, recordings (.wav) or .sig files; without it, "
                         "one is chosen from the corpus")
    ap.add_argument("--config", default="amd.conf",
                    help="amd.conf whose [general] and [signature] to replay (default amd.conf)")
    ap.add_argument("--thresholds", help="thresholds to evaluate, decimals allowed (default: the "
                                         "configured one and two points either side)")
    ap.add_argument("--average", action="store_true",
                    help="replace each of --templates, one seed recording per voice, by the "
                         "average of its voice's positive calls, evaluated leave-one-out")
    ap.add_argument("--write-averages", metavar="DIR",
                    help="with --average, write each voice's average as DIR/<seed>.sig")
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
    if args.average and not args.templates:
        ap.error("--average needs --templates, one per voice")
    if args.write_averages and not args.average:
        ap.error("--write-averages needs --average")

    cfg = amd_replay.read_config(args.config)
    sig = read_signature_config(args.config)
    report_at = sig["threshold"]
    thresholds = sorted({float(t) for t in args.thresholds.split(",")} | {report_at}) \
        if args.thresholds else [report_at + d for d in (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)
                                 if 0 <= report_at + d <= 100]
    print("config: %s%s" % (args.config, "" if os.path.exists(args.config) else " (not found, defaults)"))
    print("signature: %s" % " ".join("%s=%g" % kv for kv in sig.items()))

    calls = load_corpus(args.corpus, cfg, sig)
    if not calls:
        sys.exit("no <call>.wav / <call>.json pairs under %s" % args.corpus)
    for c in calls:
        c["path"] = os.path.join(args.corpus, c["name"] + ".wav")
    print("corpus: %d calls   %s\n" % (len(calls), dict(collections.Counter(c["label"] for c in calls))))

    if args.templates:
        try:
            templates = [load_template(p, sig) for p in args.templates]
        except ValueError as exc:
            sys.exit(str(exc))
        if args.average:
            if any(p.endswith(".sig") for p in args.templates):
                sys.exit("--average needs seed recordings, not .sig files")
            cmd_average(templates, calls, sig, thresholds, report_at, args.positive_label, args.write_averages)
        else:
            cmd_templates(templates, calls, thresholds, report_at, args.positive_label)
    else:
        cmd_select(calls, sig, thresholds, report_at, args.positive_label)


if __name__ == "__main__":
    main()
