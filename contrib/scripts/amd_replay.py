#!/usr/bin/env python3
"""Replay AMD() over recorded calls, to measure and tune it offline.

Mirrors isAnsweringMachine() in apps/app_amd.c statement for statement, so a
verdict here is the verdict Asterisk would reach on the same audio. That makes
it possible to measure what amd.conf is actually doing on real traffic, and to
try changes without placing a call.

The order of the checks matters and is preserved: MAXWORDS is tested before
LONGGREETING, so a greeting crossing both in the same frame reports MAXWORDS.
Silence detection reproduces __ast_dsp_silence_noise() in main/dsp.c: the mean
absolute sample value over a frame, compared with the threshold, accumulated
across frames and reset by the first frame that is not silent.

Audio must be 8 kHz mono 16-bit WAV. A corpus is a directory of <call>.wav with
a <call>.json sidecar carrying a "label" of MACHINE, HUMAN or anything else;
labels AMD has no verdict for are reported separately rather than scored.

Usage:
  amd_replay.py --corpus DIR [--config amd.conf]
  amd_replay.py --corpus DIR --sweep          # grid search, cross-validated
  amd_replay.py --call FILE.wav [--config amd.conf]
"""

import argparse
import collections
import glob
import json
import os
import sys
import wave

import numpy as np

FRAME_MS = 20
RATE = 8000
STATE_IN_WORD, STATE_IN_SILENCE = 1, 2

# apps/app_amd.c, the dflt* globals
DEFAULTS = {
    "initial_silence": 2500,
    "greeting": 1500,
    "after_greeting_silence": 800,
    "total_analysis_time": 5000,
    "min_word_length": 100,
    "between_words_silence": 50,
    "maximum_number_of_words": 2,
    "maximum_word_length": 5000,
    "silence_threshold": 256,
}


def read_config(path):
    """Read the [general] keys of an amd.conf. Unknown keys are ignored."""
    cfg = dict(DEFAULTS)
    if not path or not os.path.exists(path):
        return cfg
    section = None
    for line in open(path):
        line = line.split(";")[0].strip()
        if line.startswith("["):
            section = line[1:line.find("]")].strip().lower()
            continue
        # Other sections reuse key names: [signature] has its own silence_threshold.
        if section != "general" or "=" not in line:
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        if key in cfg:
            try:
                cfg[key] = int(value)
            except ValueError:
                pass
    return cfg


def frame_levels(samples):
    """Mean absolute sample value per 20ms frame: what ast_dsp_silence() tests.

    The integer division matches the C, which truncates.
    """
    n = FRAME_MS * RATE // 1000
    count = len(samples) // n
    if not count:
        return np.zeros(0, dtype=np.int64)
    return np.abs(samples[:count * n].astype(np.int64)).reshape(count, n).sum(1) // n


def amd(levels, cfg):
    """Return (status, cause, ms_to_decision) for one call."""
    total_time = words = 0
    silence = voice = consecutive_voice = 0
    in_initial_silence, in_greeting = 1, 0
    state = STATE_IN_WORD          # app_amd.c initialises to IN_WORD, not IN_SILENCE
    dsp_silence = 0

    for level in levels:
        total_time += FRAME_MS
        if total_time >= cfg["total_analysis_time"]:
            return "NOTSURE", "TOOLONG-%d" % total_time, total_time

        if level < cfg["silence_threshold"]:
            dsp_silence += FRAME_MS
        else:
            dsp_silence = 0

        if dsp_silence > 0:
            silence = dsp_silence
            if silence >= cfg["between_words_silence"]:
                state = STATE_IN_SILENCE
                consecutive_voice = 0
            if in_initial_silence == 1 and silence >= cfg["initial_silence"]:
                return ("MACHINE", "INITIALSILENCE-%d-%d" % (silence, cfg["initial_silence"]),
                        total_time)
            if silence >= cfg["after_greeting_silence"] and in_greeting == 1:
                return ("HUMAN", "HUMAN-%d-%d" % (silence, cfg["after_greeting_silence"]),
                        total_time)
        else:
            consecutive_voice += FRAME_MS
            voice += FRAME_MS
            if consecutive_voice >= cfg["min_word_length"] and state == STATE_IN_SILENCE:
                words += 1
                state = STATE_IN_WORD
            if consecutive_voice >= cfg["maximum_word_length"]:
                return "MACHINE", "MAXWORDLENGTH-%d" % consecutive_voice, total_time
            if words > cfg["maximum_number_of_words"]:
                return ("MACHINE", "MAXWORDS-%d-%d" % (words, cfg["maximum_number_of_words"]),
                        total_time)
            if in_greeting == 1 and voice >= cfg["greeting"]:
                return ("MACHINE", "LONGGREETING-%d-%d" % (voice, cfg["greeting"]), total_time)
            if voice >= cfg["min_word_length"]:
                silence = 0
            if consecutive_voice >= cfg["min_word_length"] and in_greeting == 0:
                in_initial_silence = 0
                in_greeting = 1

    # Ran out of audio. A live channel would keep waiting; a recording cannot.
    return "NOTSURE", "TOOLONG-%d" % total_time, total_time


def read_wav(path):
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            sys.exit("%s: need %d Hz mono 16-bit" % (path, RATE))
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def load_corpus(dirname):
    out = []
    for sidecar in sorted(glob.glob(os.path.join(dirname, "*.json"))):
        wav = sidecar[:-5] + ".wav"
        if not os.path.exists(wav):
            continue
        with open(sidecar) as fh:
            label = json.load(fh).get("label")
        out.append((label, os.path.basename(wav), frame_levels(read_wav(wav))))
    return out


def report(calls, cfg):
    results = [(label, name) + amd(levels, cfg) for label, name, levels in calls]
    scored = [r for r in results if r[0] in ("MACHINE", "HUMAN")]
    other = [r for r in results if r[0] not in ("MACHINE", "HUMAN")]

    print("calls: %d\n" % len(results))
    print("=== label (row) against AMD verdict (column) ===")
    print("%-10s %9s %9s %9s %9s" % ("label", "MACHINE", "HUMAN", "NOTSURE", "total"))
    for label in sorted({r[0] for r in results}, key=lambda x: (x != "MACHINE", x != "HUMAN", x)):
        rows = [r for r in results if r[0] == label]
        seen = collections.Counter(r[2] for r in rows)
        print("%-10s %9d %9d %9d %9d" % (
            label, seen["MACHINE"], seen["HUMAN"], seen["NOTSURE"], len(rows)))

    if scored:
        correct = sum(1 for r in scored if r[0] == r[2])
        print("\naccuracy on MACHINE/HUMAN: %d/%d (%.1f%%)" % (
            correct, len(scored), 100.0 * correct / len(scored)))
        for label in ("MACHINE", "HUMAN"):
            rows = [r for r in scored if r[0] == label]
            if rows:
                print("  %-8s recall    %3d/%3d (%.1f%%)" % (
                    label, sum(1 for r in rows if r[2] == label), len(rows),
                    100.0 * sum(1 for r in rows if r[2] == label) / len(rows)))

        # Two precisions, because they answer different questions. The first
        # asks how often a verdict is right among the calls AMD is meant to
        # judge. The second counts every labelled call, so a screened call
        # reported as MACHINE is the error it actually is on live traffic.
        for label in ("MACHINE", "HUMAN"):
            got = [r for r in scored if r[2] == label]
            every = [r for r in results if r[2] == label]
            if got:
                print("  %-8s precision %3d/%3d (%.1f%%) among MACHINE/HUMAN calls,"
                      " %3d/%3d (%.1f%%) over every label" % (
                          label,
                          sum(1 for r in got if r[0] == label), len(got),
                          100.0 * sum(1 for r in got if r[0] == label) / len(got),
                          sum(1 for r in every if r[0] == label), len(every),
                          100.0 * sum(1 for r in every if r[0] == label) / len(every)))

    if other:
        print("\n=== labels AMD has no verdict for ===")
        for label in sorted({r[0] for r in other}):
            rows = [r for r in other if r[0] == label]
            print("  %-10s n=%-3d  %s" % (
                label, len(rows), dict(collections.Counter(r[2] for r in rows))))

    print("\n=== AMDCAUSE by label ===")
    for label in sorted({r[0] for r in results}):
        rows = [r for r in results if r[0] == label]
        seen = collections.Counter(r[3].split("-")[0] for r in rows)
        print("  %-10s %s" % (label, ", ".join("%s %d" % kv for kv in seen.most_common())))

    print("\n=== time to decision, ms ===")
    for label in sorted({r[0] for r in results}):
        times = np.array([r[4] for r in results if r[0] == label])
        print("  %-10s median %5d  p90 %5d  max %5d" % (
            label, np.median(times), np.percentile(times, 90), times.max()))


SWEEP = {
    "after_greeting_silence": (800, 1000, 1200, 1500, 2000, 2500),
    "greeting": (1000, 1500, 2000, 2500),
    "maximum_number_of_words": (2, 3, 4),
    "total_analysis_time": (5000, 6000, 8000),
}


def sweep(calls, cfg, folds=5):
    """Grid search, then cross-validate the search itself.

    Picking the best of a few hundred configs on the same calls you score it on
    will flatter any of them, so the headline number is the held-out one: the
    config is chosen on four folds and measured on the fifth.
    """
    scored = [(label, levels) for label, _, levels in calls if label in ("MACHINE", "HUMAN")]
    if not scored:
        sys.exit("no MACHINE or HUMAN labels to score against")
    truth = np.array([label for label, _ in scored])

    grid = []
    keys = sorted(SWEEP)
    def build(i, acc):
        if i == len(keys):
            grid.append(dict(cfg, **acc))
            return
        for v in SWEEP[keys[i]]:
            build(i + 1, dict(acc, **{keys[i]: v}))
    build(0, {})

    print("evaluating %d configs over %d MACHINE/HUMAN calls" % (len(grid), len(scored)))
    verdicts = np.empty((len(grid), len(scored)), dtype="U8")
    for i, candidate in enumerate(grid):
        for k, (_, levels) in enumerate(scored):
            verdicts[i, k] = amd(levels, candidate)[0]

    def describe(row):
        false_human = int(((truth == "MACHINE") & (row == "HUMAN")).sum())
        false_machine = int(((truth == "HUMAN") & (row == "MACHINE")).sum())
        return false_human, false_machine, int((row == "NOTSURE").sum())

    baseline = [i for i, c in enumerate(grid) if all(c[k] == cfg[k] for k in keys)]
    if baseline:
        row = verdicts[baseline[0]]
        fh, fm, ns = describe(row)
        print("\nbaseline (the given config): %.1f%%   machine->HUMAN %d, human->MACHINE %d, NOTSURE %d"
              % (100.0 * (row == truth).mean(), fh, fm, ns))

    accuracy = (verdicts == truth).mean(axis=1)
    print("\n=== best configs, fitted on every call (optimistic) ===")
    print("%9s %7s %7s %8s   %s" % ("accuracy", "m->H", "h->M", "NOTSURE", "config"))
    for i in np.argsort(accuracy)[::-1][:8]:
        fh, fm, ns = describe(verdicts[i])
        print("%8.1f%% %7d %7d %8d   %s" % (
            100 * accuracy[i], fh, fm, ns,
            " ".join("%s=%d" % (k, grid[i][k]) for k in keys)))

    rng = np.random.RandomState(0)
    order = rng.permutation(len(scored))
    parts = np.array_split(order, folds)
    print("\n=== %d-fold cross-validation of the search ===" % folds)
    total = 0.0
    picked = []
    for f in range(folds):
        test = parts[f]
        train = np.concatenate([parts[g] for g in range(folds) if g != f])
        best = int((verdicts[:, train] == truth[train]).mean(axis=1).argmax())
        got = (verdicts[best, test] == truth[test]).mean()
        total += got * len(test)
        picked.append(tuple(grid[best][k] for k in keys))
        print("  fold %d: %s -> test %.1f%%" % (
            f + 1, " ".join("%s=%d" % (k, grid[best][k]) for k in keys), 100 * got))
    print("  held-out accuracy: %.1f%%" % (100.0 * total / len(scored)))
    print("\n  configs chosen per fold:")
    for combo, count in collections.Counter(picked).most_common():
        print("    x%d  %s" % (count, " ".join("%s=%d" % (k, v) for k, v in zip(keys, combo))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="amd.conf", help="amd.conf to replay (default amd.conf)")
    ap.add_argument("--corpus", metavar="DIR", help="directory of <call>.wav plus <call>.json")
    ap.add_argument("--call", metavar="FILE", help="replay a single recording")
    ap.add_argument("--sweep", action="store_true",
                    help="grid search amd.conf parameters and cross-validate the search")
    args = ap.parse_args()

    cfg = read_config(args.config)
    changed = [k for k in DEFAULTS if cfg[k] != DEFAULTS[k]]
    print("config: %s%s\n" % (
        args.config if os.path.exists(args.config) else "compiled defaults",
        ("   (differs from defaults: %s)" % ", ".join(
            "%s=%d" % (k, cfg[k]) for k in sorted(changed))) if changed else ""))

    if args.call:
        status, cause, ms = amd(frame_levels(read_wav(args.call)), cfg)
        print("%s  AMDSTATUS=%s  AMDCAUSE=%s  after %dms" % (args.call, status, cause, ms))
        return
    if not args.corpus:
        ap.error("give --corpus or --call")

    calls = load_corpus(args.corpus)
    if not calls:
        sys.exit("no <call>.wav / <call>.json pairs under %s" % args.corpus)
    if args.sweep:
        sweep(calls, cfg)
    else:
        report(calls, cfg)


if __name__ == "__main__":
    main()
