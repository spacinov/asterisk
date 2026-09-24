/*
 * Asterisk -- An open source telephony toolkit.
 *
 * Copyright (C) 2026, Jeremy Lainé
 *
 * Jeremy Lainé <jeremy.laine@m4x.org>
 *
 * See http://www.asterisk.org for more information about
 * the Asterisk project. Please do not directly contact
 * any of the maintainers of this project for assistance;
 * the project provides a web site, mailing lists and IRC
 * channels for your use.
 *
 * This program is free software, distributed under the terms of
 * the GNU General Public License Version 2. See the LICENSE file
 * at the top of the source tree.
 */

/*! \file
 *
 * \brief Recognise a known recorded prompt for AMD()
 *
 * Matches the inbound audio against one or more references of a known prompt.
 * This is aimed at call screening services such as the one recent iOS versions
 * place in front of a call: they answer, play a fixed synthetic prompt, and only
 * connect a human once the caller has said something. To AMD() that prompt is
 * indistinguishable from an answering machine greeting, because on energy and
 * timing features it is one. The prompt is however always the same words in a
 * handful of voices, so it can be recognised directly, and AMD() reports it as
 * SCREENED.
 *
 * The detector runs a 12 band filterbank over 10ms hops, converts each hop to a
 * gain invariant log spectral shape, reduces that to its envelope, and aligns
 * the inbound audio with each reference by dynamic time warping, so that a
 * rendition spoken faster, slower or with another intonation still matches. A
 * verdict is available roughly one second after the prompt starts, earlier than
 * AMD() would reach MACHINE on the same audio, which leaves the dialplan time to
 * respond while the prompt is still playing.
 *
 * References are .sig files, one per voice, in the amd/<language>/ directory
 * under the configuration directory. doc/amd-signature.txt describes the
 * matcher, the file format, and how contrib/scripts/amd_signature_sweep.py
 * makes references from labelled calls.
 *
 * \author Jeremy Lainé <jeremy.laine@m4x.org>
 */

#include "asterisk.h"

#include <dirent.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

#include "asterisk/astobj2.h"
#include "asterisk/channel.h"
#include "asterisk/cli.h"
#include "asterisk/config.h"
#include "asterisk/file.h"
#include "asterisk/format_cache.h"
#include "asterisk/frame.h"
#include "asterisk/lock.h"
#include "asterisk/options.h"
#include "asterisk/paths.h"
#include "asterisk/test.h"
#include "asterisk/translate.h"
#include "asterisk/utils.h"

#include "signature.h"

/*! Number of filterbank bands. */
#define SIG_NBANDS		12
/*! Analysis hop, in samples at 8kHz. 80 samples is 10ms. */
#define SIG_HOP			80
/*! Milliseconds per hop. */
#define SIG_HOP_MS		10
/*! Length of the causal moving average applied to the band powers, in hops. */
#define SIG_SMOOTH		3
/*! Lowest band centre frequency, in Hz. */
#define SIG_FREQ_LOW		200.0
/*! Highest band centre frequency, in Hz. */
#define SIG_FREQ_HIGH		3500.0
/*! Filterbank Q. */
#define SIG_Q			2.5
/*!
 * Log floor, as a fraction of the frame's own total power. A floor relative to
 * the frame rather than an absolute one is what keeps broadband noise from
 * dominating the quiet bands once the per frame mean is removed.
 */
#define SIG_FLOOR_REL		0.10
/*!
 * Reference frames this far below the loudest reference frame within half a
 * match window either side are not scored: they carry no reliable spectral
 * shape and are the first thing noise corrupts. -25dB.
 */
#define SIG_GATE		0.0031622777
/*! The reference starts at the first frame this far below the loudest one. -30dB. */
#define SIG_ONSET		0.001
/*!
 * Cepstral coefficients kept by the lifter. The 12 bands are narrow enough to
 * resolve the harmonics of a voice, which move with its pitch; the first few
 * cepstral coefficients keep the spectral envelope, which carries the words,
 * and drop the harmonics, so that another intonation costs no similarity. c0
 * is zero once the per frame mean is removed.
 */
#define SIG_NCEPS		6
/*!
 * The alignment. A step onto a reference frame gains its similarity to the
 * incoming frame less SIG_THETA, so a path only grows while it matches better
 * than that, and a step off the diagonal costs SIG_WARP_PENALTY more. These,
 * SIG_NCEPS and the front end above must track contrib/scripts/amd_signature_sweep.py,
 * which chose them and writes the references.
 */
#define SIG_THETA		0.90f
#define SIG_WARP_PENALTY	0.20f

#define SIG_MAX_TEMPLATES	16
#define SIG_MIN_TEMPLATE_MS	200
#define SIG_MAX_TEMPLATE_MS	8000
#define SIG_MIN_WINDOW_MS	200
/*! The format a .sig file must declare. */
#define SIG_FORMAT		"amd-signature 1"
/*! Where references are found, under the configuration directory. */
#define SIG_DIR			"amd"

/*!
 * The reference is aligned over template_length of it, from its onset, and a
 * path counts once it spans match_window of the reference. A shorter window
 * gives an earlier verdict, but 1200ms is the shortest that keeps the threshold
 * forgiving, and 3000ms of reference covers a call answered up to 1800ms into
 * the prompt.
 */
#define SIG_DEF_TEMPLATE_MS	3000
#define SIG_DEF_WINDOW_MS	1200
#define SIG_DEF_THRESHOLD	95.0f
/*!
 * Minimum mean absolute sample value, over the last match window, for a match
 * to count. Removing the per frame mean and normalising discards absolute level
 * by design, so that a quiet prompt matches a loud reference; the cost is that
 * near silence still produces a shape, and a steady noise floor can sit within
 * a few points of a reference. This is the floor that rules those out. It is
 * the same measure and default as AMD's own silenceThreshold.
 */
#define SIG_DEF_SILENCE		256
/*!
 * Consecutive hops a reference must stay above the threshold before it counts.
 * Two is enough to rule out a single anomalous frame, and three is too many: a
 * good match can peak sharply.
 */
#define SIG_DEF_DEBOUNCE	2

/*! Defaults read from the [signature] section of amd.conf. */
AST_MUTEX_DEFINE_STATIC(config_lock);
static int dfltEnabled;
static float dfltThreshold = SIG_DEF_THRESHOLD;
static int dfltTemplateLength = SIG_DEF_TEMPLATE_MS;
static int dfltWindow = SIG_DEF_WINDOW_MS;
static int dfltSilence = SIG_DEF_SILENCE;

/*! Extensions probed when "amd signature test" is given a recording. */
static const char * const sig_exts[] = { "sln", "wav", "WAV", "ulaw", "alaw", "g722", "sln16", "gsm" };

/*! One biquad section. The bandpass form has b1 == 0, so it is not stored. */
struct sig_biquad {
	float b0;
	float b2;
	float a1;
	float a2;
};

static struct sig_biquad sig_bq[SIG_NBANDS];
static float sig_lifter[SIG_NCEPS][SIG_NBANDS];
/*! The front end a .sig file must have been made for, "bands=12 hop=10 ...". */
static char sig_frontend[128];

/*! One 10ms analysis frame: a unit length spectral shape, its total power, and its level. */
struct sig_frame {
	float v[SIG_NBANDS];
	float ptot;
	float level;		/*!< mean absolute sample value over the hop */
};

/*! Filterbank state. Identical for a reference and for the live audio. */
struct sig_fe {
	float x1[SIG_NBANDS], x2[SIG_NBANDS];
	float y1[SIG_NBANDS], y2[SIG_NBANDS];
	double acc[SIG_NBANDS];
	double lacc;
	int nacc;
	float hist[SIG_SMOOTH][SIG_NBANDS];
	int hpos;
	int hcount;
};

/*! A loaded reference. Immutable once built, and shared between calls. */
struct sig_template {
	int len;		/*!< stored frames */
	int win;		/*!< frames a path must span */
	float *v;		/*!< len * SIG_NBANDS, the spectral shapes */
	float *ptot;		/*!< len, the total power of each frame */
	float *lift;		/*!< len * SIG_NCEPS, the envelopes, unit length */
	float *act;		/*!< len, 1 for a frame that is scored, 0 for one gated out */
	int nsources;		/*!< recordings and calls a .sig was averaged from */
	char *key;		/*!< cache key, "<path>|<ms>|<ms>" */
	char name[80];		/*!< file name without extension */
	char seed[80];		/*!< the recording a .sig was started from */
	char path[512];		/*!< as loaded */
};

/*! The references of one language, as found in its directory. */
struct sig_lang {
	struct sig_template *tmpl[SIG_MAX_TEMPLATES];
	int ntmpl;
	char dir[512];		/*!< where they were looked for */
	char lang[64];		/*!< as asked for, the cache key */
};

/*!
 * The alignment of one reference with the call so far: for each reference
 * frame, the best path ending there at the last two hops. H is the path's gain,
 * A how many active reference frames it covers, S the reference frame it
 * started at. Three columns of each rotate: the last hop, the one before, and
 * the one being computed.
 */
struct sig_track {
	float *h[3];
	float *a[3];
	int *s[3];
	int cur;		/*!< which of the three is the last hop */
};

/*! Streaming matcher over one or more references. */
struct amd_signature {
	struct sig_fe fe;
	struct sig_template *tmpl[SIG_MAX_TEMPLATES];
	struct sig_track track[SIG_MAX_TEMPLATES];
	int hits[SIG_MAX_TEMPLATES];
	int ntmpl;
	int cap;		/*!< level ring capacity, the longest match window */
	int maxlen;		/*!< the longest reference */
	float *rlevel;		/*!< cap, the level of the last hops */
	float *g;		/*!< maxlen, scratch */
	int head;		/*!< where the next level goes */
	int count;		/*!< hops seen so far */
	float threshold;
	int silence;		/*!< windows quieter than this cannot match */
	int debounce;
	float best;		/*!< highest score seen */
	int best_frame;
	int best_offset;	/*!< ms into the reference where the best path started */
	float best_level;	/*!< window level where best was seen */
	int matched;		/*!< index into tmpl, or -1 */
	int match_frame;
};

static struct ao2_container *templates;
static struct ao2_container *langs;

typedef void (*sig_frame_fn)(void *arg, const struct sig_frame *fr);

/*! \brief Build the RBJ constant skirt bandpass sections and the lifter, once. */
static void sig_init_coeffs(void)
{
	int j, k;
	double ratio = pow(SIG_FREQ_HIGH / SIG_FREQ_LOW, 1.0 / (SIG_NBANDS - 1));
	double f0 = SIG_FREQ_LOW;

	for (j = 0; j < SIG_NBANDS; j++, f0 *= ratio) {
		double w0 = 2.0 * M_PI * f0 / DEFAULT_SAMPLE_RATE;
		double alpha = sin(w0) / (2.0 * SIG_Q);
		double a0 = 1.0 + alpha;

		sig_bq[j].b0 = alpha / a0;
		sig_bq[j].b2 = -alpha / a0;
		sig_bq[j].a1 = -2.0 * cos(w0) / a0;
		sig_bq[j].a2 = (1.0 - alpha) / a0;
	}

	/* DCT-II rows c1 .. c6 */
	for (k = 0; k < SIG_NCEPS; k++) {
		for (j = 0; j < SIG_NBANDS; j++) {
			sig_lifter[k][j] = cos(M_PI * (k + 1) * (j + 0.5) / SIG_NBANDS);
		}
	}

	snprintf(sig_frontend, sizeof(sig_frontend), "bands=%d hop=%d q=%g low=%g high=%g floor=%.2f smooth=%d",
		SIG_NBANDS, SIG_HOP_MS, SIG_Q, SIG_FREQ_LOW, SIG_FREQ_HIGH, SIG_FLOOR_REL, SIG_SMOOTH);
}

/*! \brief Reduce a spectral shape to its envelope, of unit length. */
static void sig_lift(const float *v, float *out)
{
	float norm = 0.0f;
	int j, k;

	for (k = 0; k < SIG_NCEPS; k++) {
		float c = 0.0f;

		for (j = 0; j < SIG_NBANDS; j++) {
			c += sig_lifter[k][j] * v[j];
		}
		out[k] = c;
		norm += c * c;
	}
	norm = sqrtf(norm);
	if (norm > 0.0f) {
		for (k = 0; k < SIG_NCEPS; k++) {
			out[k] /= norm;
		}
	}
}

static void sig_fe_reset(struct sig_fe *fe)
{
	memset(fe, 0, sizeof(*fe));
}

/*!
 * \brief Turn a completed hop into a frame and hand it to the caller.
 *
 * The band powers are smoothed over the last SIG_SMOOTH hops. The average is
 * causal, which delays the output by one hop relative to a centred one, but the
 * reference goes through the same filter so the alignment is unaffected.
 */
static void sig_fe_emit(struct sig_fe *fe, sig_frame_fn fn, void *arg)
{
	struct sig_frame fr;
	float mean = 0.0f, norm = 0.0f, ptot = 0.0f;
	float p[SIG_NBANDS];
	int j, k, n;

	for (j = 0; j < SIG_NBANDS; j++) {
		fe->hist[fe->hpos][j] = fe->acc[j] / SIG_HOP;
		fe->acc[j] = 0.0;
	}
	fe->hpos = (fe->hpos + 1) % SIG_SMOOTH;
	if (fe->hcount < SIG_SMOOTH) {
		fe->hcount++;
	}
	n = fe->hcount;

	for (j = 0; j < SIG_NBANDS; j++) {
		float sum = 0.0f;
		for (k = 0; k < n; k++) {
			sum += fe->hist[k][j];
		}
		p[j] = sum / n;
		ptot += p[j];
	}

	for (j = 0; j < SIG_NBANDS; j++) {
		fr.v[j] = logf(p[j] + SIG_FLOOR_REL * ptot + 1e-9f);
		mean += fr.v[j];
	}
	mean /= SIG_NBANDS;
	for (j = 0; j < SIG_NBANDS; j++) {
		fr.v[j] -= mean;
		norm += fr.v[j] * fr.v[j];
	}
	norm = sqrtf(norm);
	if (norm > 0.0f) {
		for (j = 0; j < SIG_NBANDS; j++) {
			fr.v[j] /= norm;
		}
	}
	fr.ptot = ptot;
	fr.level = fe->lacc / SIG_HOP;
	fe->lacc = 0.0;

	fn(arg, &fr);
}

/*! \brief Run signed linear samples through the filterbank. */
static void sig_fe_feed(struct sig_fe *fe, const int16_t *s, int n, sig_frame_fn fn, void *arg)
{
	int i, j;

	for (i = 0; i < n; i++) {
		float x = s[i];

		for (j = 0; j < SIG_NBANDS; j++) {
			const struct sig_biquad *b = &sig_bq[j];
			float y = b->b0 * x + b->b2 * fe->x2[j] - b->a1 * fe->y1[j] - b->a2 * fe->y2[j];

			fe->x2[j] = fe->x1[j];
			fe->x1[j] = x;
			fe->y2[j] = fe->y1[j];
			fe->y1[j] = y;
			fe->acc[j] += (double) y * y;
		}
		fe->lacc += fabsf(x);

		if (++fe->nacc == SIG_HOP) {
			fe->nacc = 0;
			sig_fe_emit(fe, fn, arg);
		}
	}
}

/*! \brief Build "<base>.<ext>", rooted under the sounds directory for relative names. */
static char *sig_build_filename(const char *base, const char *ext)
{
	char *fn = NULL;

	if (base[0] == '/') {
		if (ast_asprintf(&fn, "%s.%s", base, ext) < 0) {
			return NULL;
		}
	} else {
		if (ast_asprintf(&fn, "%s/sounds/%s.%s", ast_config_AST_DATA_DIR, base, ext) < 0) {
			return NULL;
		}
	}

	return fn;
}

/*!
 * \brief Find which extension a recording exists in, for the CLI.
 *
 * Probes with access() rather than ast_readfile() so that a missing candidate
 * does not log a warning.
 *
 * \retval the extension, or NULL if the file does not exist in any known format
 */
static const char *sig_probe(const char *base)
{
	int i;

	for (i = 0; i < ARRAY_LEN(sig_exts); i++) {
		char *fn = sig_build_filename(base, sig_exts[i]);

		if (!fn) {
			continue;
		}
		if (!access(fn, R_OK)) {
			ast_free(fn);
			return sig_exts[i];
		}
		ast_free(fn);
	}

	return NULL;
}

/*!
 * \brief Read a sound file into one signed linear buffer at 8kHz.
 *
 * \note This blocks on disk I/O and must never be called from a frame loop.
 *
 * \retval the samples, which the caller must ast_free(), or NULL
 */
static int16_t *sig_load_slin(const char *base, const char *ext, int *nsamples)
{
	struct ast_filestream *fs;
	struct ast_trans_pvt *trans = NULL;
	struct ast_frame *f;
	int16_t *buf = NULL;
	int n = 0;

	if (!(fs = ast_readfile(base, ext, NULL, O_RDONLY, 0, 0))) {
		return NULL;
	}

	while ((f = ast_readframe(fs))) {
		struct ast_frame *sln = f;

		if (f->frametype != AST_FRAME_VOICE) {
			ast_frfree(f);
			continue;
		}

		if (ast_format_cmp(f->subclass.format, ast_format_slin) != AST_FORMAT_CMP_EQUAL) {
			if (!trans && !(trans = ast_translator_build_path(ast_format_slin, f->subclass.format))) {
				ast_log(LOG_WARNING, "AMD: signature: no path from %s to slin for '%s'\n",
					ast_format_get_name(f->subclass.format), base);
				ast_frfree(f);
				break;
			}
			if (!(sln = ast_translate(trans, f, 0))) {
				ast_frfree(f);
				continue;
			}
		}

		if (sln->datalen > 0) {
			int add = sln->datalen / 2;
			int16_t *grown = ast_realloc(buf, (n + add) * sizeof(*buf));

			if (!grown) {
				if (sln != f) {
					ast_frfree(sln);
				}
				ast_frfree(f);
				ast_free(buf);
				buf = NULL;
				n = 0;
				break;
			}
			buf = grown;
			memcpy(buf + n, sln->data.ptr, add * sizeof(*buf));
			n += add;
		}

		if (sln != f) {
			ast_frfree(sln);
		}
		ast_frfree(f);
	}

	if (trans) {
		ast_translator_free_path(trans);
	}
	ast_closestream(fs);

	if (!buf || !n) {
		ast_free(buf);
		return NULL;
	}

	*nsamples = n;

	return buf;
}

/*! Collects the frames of a reference, from audio or from a .sig file. */
struct sig_builder {
	struct sig_frame *fr;
	int n;
	int cap;
	int failed;
	int nsources;
	char seed[80];
};

static void sig_builder_frame(void *arg, const struct sig_frame *fr)
{
	struct sig_builder *b = arg;

	if (b->failed) {
		return;
	}
	if (b->n == b->cap) {
		int cap = b->cap ? b->cap * 2 : 256;
		struct sig_frame *grown = ast_realloc(b->fr, cap * sizeof(*grown));

		if (!grown) {
			b->failed = 1;
			return;
		}
		b->fr = grown;
		b->cap = cap;
	}
	b->fr[b->n++] = *fr;
}

/*! \brief Run a whole recording through the front end. */
static int sig_frames_from_audio(const char *path, const char *ext, struct sig_builder *b)
{
	struct sig_fe fe;
	int16_t *samples;
	int nsamples = 0;

	if (!(samples = sig_load_slin(path, ext, &nsamples))) {
		ast_log(LOG_WARNING, "AMD: signature: unable to read '%s.%s'\n", path, ext);
		return -1;
	}
	sig_fe_reset(&fe);
	sig_fe_feed(&fe, samples, nsamples, sig_builder_frame, b);
	ast_free(samples);

	return b->failed ? -1 : 0;
}

/*!
 * \brief Read the frames of a .sig file.
 *
 * A header of "key: value" lines, then one line per hop: the log of the total
 * power, and the 12 values of the spectral shape. A file made for another
 * front end is refused, since its frames would not be comparable. The shapes
 * are written to a few digits, and are made unit length again here, as the
 * sweep script does when it reads them.
 */
static int sig_frames_from_sig(const char *path, struct sig_builder *b)
{
	char line[1024];
	FILE *fp;
	int lineno = 0, frames = -1, format = 0, frontend = 0, res = -1;

	if (!(fp = fopen(path, "r"))) {
		ast_log(LOG_WARNING, "AMD: signature: unable to read '%s': %s\n", path, strerror(errno));
		return -1;
	}

	while (fgets(line, sizeof(line), fp)) {
		struct sig_frame fr = { .level = 0.0f };
		char *s = line, *colon, *end;
		float norm = 0.0f, logp;
		int j;

		lineno++;
		if ((end = strchr(s, ';'))) {
			*end = '\0';
		}
		s = ast_strip(s);
		if (ast_strlen_zero(s)) {
			continue;
		}

		if ((colon = strchr(s, ':'))) {
			char *key = s, *value = ast_strip(colon + 1);

			*colon = '\0';
			key = ast_strip(key);
			if (!strcasecmp(key, "format")) {
				if (strcmp(value, SIG_FORMAT)) {
					ast_log(LOG_WARNING, "AMD: signature: '%s' is in format '%s', need '%s'\n",
						path, value, SIG_FORMAT);
					goto done;
				}
				format = 1;
			} else if (!strcasecmp(key, "frontend")) {
				if (strcmp(value, sig_frontend)) {
					ast_log(LOG_WARNING, "AMD: signature: '%s' was made for front end '%s', this one is '%s'\n",
						path, value, sig_frontend);
					goto done;
				}
				frontend = 1;
			} else if (!strcasecmp(key, "seed")) {
				ast_copy_string(b->seed, value, sizeof(b->seed));
			} else if (!strcasecmp(key, "sources")) {
				b->nsources = atoi(value);
			} else if (!strcasecmp(key, "frames")) {
				frames = atoi(value);
			}
			continue;
		}

		logp = strtof(s, &end);
		if (end == s) {
			goto bad;
		}
		for (j = 0; j < SIG_NBANDS; j++) {
			s = end;
			fr.v[j] = strtof(s, &end);
			if (end == s) {
				goto bad;
			}
			norm += fr.v[j] * fr.v[j];
		}
		if (!ast_strlen_zero(ast_skip_blanks(end))) {
			goto bad;
		}
		norm = sqrtf(norm);
		if (norm > 0.0f) {
			for (j = 0; j < SIG_NBANDS; j++) {
				fr.v[j] /= norm;
			}
		}
		fr.ptot = expf(logp);
		sig_builder_frame(b, &fr);
	}

	if (!format || !frontend) {
		ast_log(LOG_WARNING, "AMD: signature: '%s' has no %s line\n", path, format ? "frontend" : "format");
	} else if (!b->n || (frames >= 0 && frames != b->n)) {
		ast_log(LOG_WARNING, "AMD: signature: '%s' has %d frames, its header says %d\n", path, b->n, frames);
	} else if (!b->failed) {
		res = 0;
	}
	goto done;

bad:
	ast_log(LOG_WARNING, "AMD: signature: '%s' line %d is not a frame of %d values\n",
		path, lineno, SIG_NBANDS + 1);
done:
	fclose(fp);

	return res;
}

static void sig_template_destroy(void *obj)
{
	struct sig_template *t = obj;

	ast_free(t->v);
	ast_free(t->ptot);
	ast_free(t->lift);
	ast_free(t->act);
	ast_free(t->key);
}

/*!
 * \brief Make a reference of the frames of a recording or of a .sig file.
 *
 * The reference starts at the first frame which is not silence and runs for
 * template_ms. Within it, frames more than 25dB below the loudest frame within
 * half a match window either side are not scored. The gate is local rather
 * than over the whole reference because a quiet passage of a loud prompt still
 * has to be scored on its own terms.
 */
static struct sig_template *sig_template_from_frames(const char *name, const char *path,
	const char *key, const struct sig_builder *b, int template_ms, int window_ms)
{
	struct sig_template *t;
	float peak = 0.0f;
	int onset, len, win, half, i, j;

	for (i = 0; i < b->n; i++) {
		if (b->fr[i].ptot > peak) {
			peak = b->fr[i].ptot;
		}
	}
	for (onset = 0; onset < b->n && b->fr[onset].ptot < peak * SIG_ONSET; onset++) {
	}
	if (onset >= b->n) {
		ast_log(LOG_WARNING, "AMD: signature: '%s' appears to be silent\n", path);
		return NULL;
	}

	win = window_ms / SIG_HOP_MS;
	len = template_ms / SIG_HOP_MS;
	if (len > b->n - onset) {
		len = b->n - onset;
	}
	if (len < win) {
		ast_log(LOG_WARNING, "AMD: signature: '%s' is too short, %dms after silence, need %dms\n",
			path, len * SIG_HOP_MS, window_ms);
		return NULL;
	}

	if (!(t = ao2_alloc(sizeof(*t), sig_template_destroy))) {
		return NULL;
	}
	t->v = ast_malloc(len * SIG_NBANDS * sizeof(*t->v));
	t->ptot = ast_malloc(len * sizeof(*t->ptot));
	t->lift = ast_malloc(len * SIG_NCEPS * sizeof(*t->lift));
	t->act = ast_malloc(len * sizeof(*t->act));
	t->key = ast_strdup(key);
	if (!t->v || !t->ptot || !t->lift || !t->act || !t->key) {
		ao2_ref(t, -1);
		return NULL;
	}
	t->len = len;
	t->win = win;
	t->nsources = b->nsources;
	ast_copy_string(t->name, name, sizeof(t->name));
	ast_copy_string(t->seed, b->seed, sizeof(t->seed));
	ast_copy_string(t->path, path, sizeof(t->path));

	for (i = 0; i < len; i++) {
		memcpy(&t->v[i * SIG_NBANDS], b->fr[onset + i].v, SIG_NBANDS * sizeof(*t->v));
		t->ptot[i] = b->fr[onset + i].ptot;
		sig_lift(&t->v[i * SIG_NBANDS], &t->lift[i * SIG_NCEPS]);
	}

	half = win / 2;
	for (i = 0; i < len; i++) {
		int lo = i - half < 0 ? 0 : i - half;
		int hi = i + half > len ? len : i + half;

		peak = 0.0f;
		for (j = lo; j < hi; j++) {
			if (t->ptot[j] > peak) {
				peak = t->ptot[j];
			}
		}
		t->act[i] = t->ptot[i] >= peak * SIG_GATE ? 1.0f : 0.0f;
	}

	ast_debug(1, "AMD: signature: loaded '%s' from '%s', onset %dms, %dms stored\n",
		name, path, onset * SIG_HOP_MS, t->len * SIG_HOP_MS);

	return t;
}

/*! \brief Load a reference from a .sig file, or from a recording if ext is not "sig". */
static struct sig_template *sig_template_build(const char *name, const char *path,
	const char *ext, const char *key, int template_ms, int window_ms)
{
	struct sig_builder b = { 0 };
	struct sig_template *t = NULL;
	char *fn = NULL;
	int res;

	if (!strcmp(ext, "sig")) {
		if (ast_asprintf(&fn, "%s.sig", path) < 0) {
			return NULL;
		}
		res = sig_frames_from_sig(fn, &b);
		ast_free(fn);
	} else {
		res = sig_frames_from_audio(path, ext, &b);
		ast_copy_string(b.seed, name, sizeof(b.seed));
		b.nsources = 1;
	}

	if (!res) {
		t = sig_template_from_frames(name, path, key, &b, template_ms, window_ms);
	}
	ast_free(b.fr);

	return t;
}

static int sig_template_hash(const void *obj, int flags)
{
	const struct sig_template *t = obj;
	const char *key = (flags & OBJ_SEARCH_KEY) ? obj : t->key;

	return ast_str_hash(key);
}

static int sig_template_cmp(void *obj, void *arg, int flags)
{
	const struct sig_template *t = obj;
	const char *key = (flags & OBJ_SEARCH_KEY) ? arg : ((struct sig_template *) arg)->key;

	return strcmp(t->key, key) ? 0 : CMP_MATCH;
}

/*!
 * \brief Fetch the reference in "<path>.sig", loading it the first time it is asked for.
 *
 * \retval a reference the caller must release with ao2_ref(), or NULL
 */
static struct sig_template *sig_template_get(const char *name, const char *path,
	int template_ms, int window_ms)
{
	struct sig_template *t;
	char key[700];

	snprintf(key, sizeof(key), "%s|%d|%d", path, template_ms, window_ms);

	ao2_lock(templates);
	if (!(t = ao2_find(templates, key, OBJ_SEARCH_KEY | OBJ_NOLOCK))) {
		if ((t = sig_template_build(name, path, "sig", key, template_ms, window_ms))) {
			ao2_link_flags(templates, t, OBJ_NOLOCK);
		}
	}
	ao2_unlock(templates);

	return t;
}

static void sig_lang_destroy(void *obj)
{
	struct sig_lang *l = obj;
	int i;

	for (i = 0; i < l->ntmpl; i++) {
		ao2_cleanup(l->tmpl[i]);
	}
}

static int sig_lang_hash(const void *obj, int flags)
{
	const struct sig_lang *l = obj;
	const char *key = (flags & OBJ_SEARCH_KEY) ? obj : l->lang;

	return ast_str_hash(key);
}

static int sig_lang_cmp(void *obj, void *arg, int flags)
{
	const struct sig_lang *l = obj;
	const char *key = (flags & OBJ_SEARCH_KEY) ? arg : ((struct sig_lang *) arg)->lang;

	return strcmp(l->lang, key) ? 0 : CMP_MATCH;
}

static int sig_is_dir(const char *path)
{
	struct stat st;

	return !stat(path, &st) && S_ISDIR(st.st_mode);
}

static int sig_name_cmp(const void *a, const void *b)
{
	return strcmp(*(char * const *) a, *(char * const *) b);
}

/*!
 * \brief Load every .sig file of a directory, in name order.
 *
 * The order only decides which reference is reported when two match on the
 * same hop, and which are left out past SIG_MAX_TEMPLATES; making it the name
 * order makes both reproducible.
 */
static void sig_lang_scan(struct sig_lang *l, int template_ms, int window_ms)
{
	struct dirent *de;
	char **names = NULL;
	int n = 0, cap = 0, i;
	DIR *dir;

	if (!(dir = opendir(l->dir))) {
		return;
	}
	while ((de = readdir(dir))) {
		size_t len = strlen(de->d_name);

		if (de->d_name[0] == '.' || len <= 4 || strcmp(de->d_name + len - 4, ".sig")) {
			continue;
		}
		if (n == cap) {
			char **grown = ast_realloc(names, (cap = cap ? cap * 2 : 16) * sizeof(*names));

			if (!grown) {
				break;
			}
			names = grown;
		}
		if (!(names[n] = ast_strdup(de->d_name))) {
			break;
		}
		n++;
	}
	closedir(dir);

	if (n) {
		qsort(names, n, sizeof(*names), sig_name_cmp);
	}
	if (n > SIG_MAX_TEMPLATES) {
		struct ast_str *skipped = ast_str_alloca(512);

		for (i = SIG_MAX_TEMPLATES; i < n; i++) {
			ast_str_append(&skipped, 0, "%s%s", i > SIG_MAX_TEMPLATES ? ", " : "", names[i]);
		}
		ast_log(LOG_WARNING, "AMD: signature: %d references in %s, only %d are used; skipping %s\n",
			n, l->dir, SIG_MAX_TEMPLATES, ast_str_buffer(skipped));
	}

	for (i = 0; i < n; i++) {
		if (l->ntmpl < SIG_MAX_TEMPLATES) {
			char name[80], path[600];
			struct sig_template *t;

			snprintf(name, sizeof(name), "%.*s", (int) strlen(names[i]) - 4, names[i]);
			snprintf(path, sizeof(path), "%s/%s", l->dir, name);
			if ((t = sig_template_get(name, path, template_ms, window_ms))) {
				l->tmpl[l->ntmpl++] = t;
			}
		}
		ast_free(names[i]);
	}
	ast_free(names);
}

/*!
 * \brief Fetch the references of a language, scanning its directory the first
 * time it is asked for after a reload.
 *
 * A language with a dialect, such as fr_CA, falls back to fr when it has no
 * directory of its own, and to nothing else: a call in a language without
 * references runs plain AMD, with a warning the first time.
 *
 * \retval a reference the caller must release with ao2_ref(), or NULL
 */
static struct sig_lang *sig_lang_get(const char *lang, int template_ms, int window_ms)
{
	struct sig_lang *l;

	ao2_lock(langs);
	if ((l = ao2_find(langs, lang, OBJ_SEARCH_KEY | OBJ_NOLOCK))) {
		ao2_unlock(langs);
		return l;
	}

	if (!(l = ao2_alloc(sizeof(*l), sig_lang_destroy))) {
		ao2_unlock(langs);
		return NULL;
	}
	ast_copy_string(l->lang, lang, sizeof(l->lang));
	snprintf(l->dir, sizeof(l->dir), "%s/%s/%s", ast_config_AST_CONFIG_DIR, SIG_DIR, lang);
	if (!sig_is_dir(l->dir) && strchr(lang, '_')) {
		snprintf(l->dir, sizeof(l->dir), "%s/%s/%.*s", ast_config_AST_CONFIG_DIR, SIG_DIR,
			(int) (strchr(lang, '_') - lang), lang);
	}
	if (!ast_strlen_zero(lang)) {
		sig_lang_scan(l, template_ms, window_ms);
	}
	if (!l->ntmpl) {
		ast_log(LOG_WARNING, "AMD: signature: no references for language '%s' in %s, detecting without them\n",
			lang, l->dir);
	}
	ao2_link_flags(langs, l, OBJ_NOLOCK);
	ao2_unlock(langs);

	return l;
}

static void sig_detector_free(struct amd_signature *d)
{
	int i, k;

	if (!d) {
		return;
	}
	for (i = 0; i < d->ntmpl; i++) {
		ao2_cleanup(d->tmpl[i]);
		for (k = 0; k < 3; k++) {
			ast_free(d->track[i].h[k]);
			ast_free(d->track[i].a[k]);
			ast_free(d->track[i].s[k]);
		}
	}
	ast_free(d->rlevel);
	ast_free(d->g);
	ast_free(d);
}

/*! \brief Start an empty matcher; references are added with sig_detector_add(). */
static struct amd_signature *sig_detector_alloc(float threshold, int silence, int debounce)
{
	struct amd_signature *d;

	if (!(d = ast_calloc(1, sizeof(*d)))) {
		return NULL;
	}
	d->threshold = threshold;
	d->silence = silence;
	d->debounce = debounce;
	d->matched = -1;
	d->best_frame = -1;
	d->best_offset = -1;
	sig_fe_reset(&d->fe);

	return d;
}

/*!
 * \brief Add a reference to a matcher, which takes over the caller's reference to it.
 *
 * \retval 0 on success, -1 if there is no memory, in which case the reference
 * is released
 */
static int sig_detector_add(struct amd_signature *d, struct sig_template *t)
{
	struct sig_track *tr = &d->track[d->ntmpl];
	int i, k;

	d->tmpl[d->ntmpl++] = t;
	if (t->win > d->cap) {
		d->cap = t->win;
	}
	if (t->len > d->maxlen) {
		d->maxlen = t->len;
	}
	for (k = 0; k < 3; k++) {
		tr->h[k] = ast_malloc(t->len * sizeof(*tr->h[k]));
		tr->a[k] = ast_calloc(t->len, sizeof(*tr->a[k]));
		tr->s[k] = ast_calloc(t->len, sizeof(*tr->s[k]));
		if (!tr->h[k] || !tr->a[k] || !tr->s[k]) {
			return -1;
		}
		/* No path ends anywhere before the first hop. */
		for (i = 0; i < t->len; i++) {
			tr->h[k][i] = -INFINITY;
		}
	}

	return 0;
}

/*!
 * \brief Allocate the history once every reference has been added.
 *
 * \retval 0 on success, -1 if there is nothing to match or no memory
 */
static int sig_detector_ready(struct amd_signature *d)
{
	if (!d->ntmpl) {
		return -1;
	}
	d->rlevel = ast_calloc(d->cap, sizeof(*d->rlevel));
	d->g = ast_calloc(d->maxlen, sizeof(*d->g));

	return d->rlevel && d->g ? 0 : -1;
}

/*!
 * \brief Extend the alignment of one reference by one hop.
 *
 * For every reference frame i, the best path ending there comes either from
 * nothing, a restart, or from one of three steps: (1,1) from frame i-1 at the
 * last hop, (1,2) from frame i-1 two hops ago, skipping a hop of the call, or
 * (2,1) from frame i-2 at the last hop, covering frame i-1 on the way. Every
 * way in comes from an earlier hop, and the ways are tried in that order, the
 * first best winning, exactly as amd_signature_sweep.py computes them.
 *
 * \return the best score of a path spanning at least a match window, at least
 * half of it active, or 0 if none does; \a start is where it began
 */
static float sig_track_hop(struct sig_track *tr, const struct sig_template *t, const float *u,
	float *g, int *start)
{
	const float *h1 = tr->h[tr->cur], *h2 = tr->h[(tr->cur + 2) % 3];
	const float *a1 = tr->a[tr->cur], *a2 = tr->a[(tr->cur + 2) % 3];
	const int *s1 = tr->s[tr->cur], *s2 = tr->s[(tr->cur + 2) % 3];
	int next = (tr->cur + 1) % 3;
	float *hn = tr->h[next], *an = tr->a[next];
	int *sn = tr->s[next];
	float best = 0.0f;
	int i, k;

	for (i = 0; i < t->len; i++) {
		const float *r = &t->lift[i * SIG_NCEPS];
		float dot = 0.0f;

		for (k = 0; k < SIG_NCEPS; k++) {
			dot += r[k] * u[k];
		}
		g[i] = t->act[i] * (dot - SIG_THETA);
	}

	for (i = 0; i < t->len; i++) {
		float p = SIG_WARP_PENALTY * t->act[i];
		float h = 0.0f, a = 0.0f, c;
		int s = i;

		if (i >= 1) {
			if ((c = h1[i - 1]) > h) {
				h = c;
				a = a1[i - 1];
				s = s1[i - 1];
			}
			if ((c = h2[i - 1] - p) > h) {
				h = c;
				a = a2[i - 1];
				s = s2[i - 1];
			}
		}
		if (i >= 2 && (c = h1[i - 2] + g[i - 1] - p) > h) {
			h = c;
			a = a1[i - 2] + t->act[i - 1];
			s = s1[i - 2];
		}
		hn[i] = h + g[i];
		an[i] = a + t->act[i];
		sn[i] = s;

		/*
		 * Without the second condition, a reference which is mostly pauses
		 * matches the noise floor of any call on the few frames it has.
		 */
		if (i - s + 1 >= t->win && 2.0f * an[i] >= t->win) {
			float score = 100.0f * (SIG_THETA + hn[i] / an[i]);

			if (score > best) {
				best = score;
				*start = s;
			}
		}
	}
	tr->cur = next;

	return best;
}

/*!
 * \brief Align every reference one hop further, and score the ones a full
 * match window of audio has been heard for.
 */
static void sig_detector_frame(void *arg, const struct sig_frame *fr)
{
	struct amd_signature *d = arg;
	float u[SIG_NCEPS];
	int i, k;

	sig_lift(fr->v, u);
	d->rlevel[d->head] = fr->level;
	d->head = (d->head + 1) % d->cap;
	d->count++;

	for (i = 0; i < d->ntmpl; i++) {
		const struct sig_template *t = d->tmpl[i];
		float level = 0.0f, score;
		int start = -1;

		/* The alignment runs from the first hop, the scoring once a window has been heard. */
		score = sig_track_hop(&d->track[i], t, u, d->g, &start);
		if (d->count < t->win) {
			continue;
		}

		for (k = 0; k < t->win; k++) {
			level += d->rlevel[(d->head - 1 - k + d->cap) % d->cap];
		}
		level /= t->win;

		if (score > d->best) {
			d->best = score;
			d->best_frame = d->count;
			d->best_level = level;
			d->best_offset = start * SIG_HOP_MS;
		}

		if (level < d->silence) {
			/* Essentially silence: the shape is the noise floor, not a prompt. */
			d->hits[i] = 0;
			continue;
		}

		if (score >= d->threshold) {
			if (++d->hits[i] >= d->debounce && d->matched < 0) {
				d->matched = i;
				d->match_frame = d->count;
			}
		} else {
			d->hits[i] = 0;
		}
	}
}

/*!
 * \brief Feed signed linear audio to the matcher.
 *
 * \retval 1 if a reference has matched, 0 otherwise
 */
static int sig_detector_feed(struct amd_signature *d, const int16_t *samples, int n)
{
	sig_fe_feed(&d->fe, samples, n, sig_detector_frame, d);

	return d->matched >= 0;
}

void amd_signature_flush(void)
{
	if (langs) {
		ao2_callback(langs, OBJ_NODATA | OBJ_MULTIPLE | OBJ_UNLINK, NULL, NULL);
	}
	if (templates) {
		ao2_callback(templates, OBJ_NODATA | OBJ_MULTIPLE | OBJ_UNLINK, NULL, NULL);
	}
}

/*!
 * \brief Read the [signature] section of amd.conf.
 *
 * It has to be its own section rather than keys in [general]: AMD warns about
 * keys it does not recognise in [general], but silently skips any other
 * category. Without the section, or unless it sets enabled, AMD() runs exactly
 * as it always has.
 */
void amd_signature_load_config(struct ast_config *cfg)
{
	struct ast_variable *var;
	int enabled = 0;
	float threshold = SIG_DEF_THRESHOLD;
	int length = SIG_DEF_TEMPLATE_MS;
	int window = SIG_DEF_WINDOW_MS;
	int silence = SIG_DEF_SILENCE;

	for (var = ast_variable_browse(cfg, "signature"); var; var = var->next) {
		if (!strcasecmp(var->name, "enabled")) {
			enabled = ast_true(var->value);
		} else if (!strcasecmp(var->name, "threshold")) {
			char *end;

			threshold = strtof(var->value, &end);
			if (end == var->value || !ast_strlen_zero(ast_skip_blanks(end)) || threshold < 0 || threshold > 100) {
				ast_log(LOG_WARNING, "AMD: signature: threshold '%s' invalid at line %d of amd.conf, using %.1f\n",
					var->value, var->lineno, SIG_DEF_THRESHOLD);
				threshold = SIG_DEF_THRESHOLD;
			}
		} else if (!strcasecmp(var->name, "template_length")) {
			length = atoi(var->value);
			if (length < SIG_MIN_TEMPLATE_MS || length > SIG_MAX_TEMPLATE_MS) {
				ast_log(LOG_WARNING, "AMD: signature: template_length %d out of range at line %d of amd.conf, using %d\n",
					length, var->lineno, SIG_DEF_TEMPLATE_MS);
				length = SIG_DEF_TEMPLATE_MS;
			}
		} else if (!strcasecmp(var->name, "match_window")) {
			window = atoi(var->value);
			if (window < SIG_MIN_WINDOW_MS || window > SIG_MAX_TEMPLATE_MS) {
				ast_log(LOG_WARNING, "AMD: signature: match_window %d out of range at line %d of amd.conf, using %d\n",
					window, var->lineno, SIG_DEF_WINDOW_MS);
				window = SIG_DEF_WINDOW_MS;
			}
		} else if (!strcasecmp(var->name, "silence_threshold")) {
			silence = atoi(var->value);
			if (silence < 0 || silence > 32767) {
				ast_log(LOG_WARNING, "AMD: signature: silence_threshold %d out of range at line %d of amd.conf, using %d\n",
					silence, var->lineno, SIG_DEF_SILENCE);
				silence = SIG_DEF_SILENCE;
			}
		} else {
			ast_log(LOG_WARNING, "AMD: signature: Unknown keyword %s at line %d of amd.conf\n",
				var->name, var->lineno);
		}
	}

	ast_mutex_lock(&config_lock);
	dfltEnabled = enabled;
	dfltThreshold = threshold;
	dfltTemplateLength = length;
	dfltWindow = window;
	dfltSilence = silence;
	ast_mutex_unlock(&config_lock);

	amd_signature_flush();

	ast_verb(5, "AMD signature defaults: enabled [%s] threshold [%.1f] templateLength [%d] matchWindow [%d] silenceThreshold [%d]\n",
		AST_YESNO(enabled), threshold, length, window, silence);
}

struct amd_signature *amd_signature_start(struct ast_channel *chan)
{
	struct amd_signature *d;
	struct sig_lang *l;
	const char *lang = ast_channel_language(chan);
	int enabled, length, window, silence, i;
	float threshold;

	ast_mutex_lock(&config_lock);
	enabled = dfltEnabled;
	threshold = dfltThreshold;
	length = dfltTemplateLength;
	window = dfltWindow;
	silence = dfltSilence;
	ast_mutex_unlock(&config_lock);

	if (!enabled || !(l = sig_lang_get(S_OR(lang, ""), length, window))) {
		return NULL;
	}
	if (!l->ntmpl || !(d = sig_detector_alloc(threshold, silence, SIG_DEF_DEBOUNCE))) {
		ao2_ref(l, -1);
		return NULL;
	}
	for (i = 0; i < l->ntmpl; i++) {
		ao2_ref(l->tmpl[i], +1);
		if (sig_detector_add(d, l->tmpl[i])) {
			break;
		}
	}
	if (i < l->ntmpl || sig_detector_ready(d)) {
		ao2_ref(l, -1);
		sig_detector_free(d);
		return NULL;
	}

	ast_verb(3, "AMD: Channel [%s]. Signature references [%d from %s] threshold [%.1f] templateLength [%d] matchWindow [%d] silenceThreshold [%d]\n",
		ast_channel_name(chan), l->ntmpl, l->dir, threshold, length, window, silence);
	ao2_ref(l, -1);

	return d;
}

int amd_signature_feed(struct amd_signature *s, struct ast_frame *f)
{
	if (!s || f->frametype != AST_FRAME_VOICE || f->datalen <= 0
		|| ast_format_cmp(f->subclass.format, ast_format_slin) != AST_FORMAT_CMP_EQUAL) {
		return 0;
	}

	return sig_detector_feed(s, f->data.ptr, f->datalen / 2);
}

const char *amd_signature_name(struct amd_signature *s)
{
	return s->matched >= 0 ? s->tmpl[s->matched]->name : "";
}

float amd_signature_score(struct amd_signature *s)
{
	return s->best;
}

void amd_signature_free(struct amd_signature *s)
{
	sig_detector_free(s);
}

/*!
 * \brief Resolve the reference given to "amd signature test".
 *
 * A .sig file, or any recording, named absolutely or relative to the amd
 * directory, with or without its extension.
 *
 * \retval the extension found, "sig" for a .sig file, or NULL
 */
static const char *sig_resolve_reference(const char *arg, char *path, size_t pathlen)
{
	char *fn = NULL, *dot;
	int found, i;

	if (arg[0] == '/') {
		ast_copy_string(path, arg, pathlen);
	} else {
		snprintf(path, pathlen, "%s/%s/%s", ast_config_AST_CONFIG_DIR, SIG_DIR, arg);
	}

	/* An extension given explicitly is taken off, and tried first. */
	if ((dot = strrchr(path, '.')) && !strchr(dot, '/')) {
		if (!strcmp(dot + 1, "sig")) {
			*dot = '\0';
		} else {
			for (i = 0; i < ARRAY_LEN(sig_exts); i++) {
				if (!strcmp(dot + 1, sig_exts[i]) && !access(path, R_OK)) {
					*dot = '\0';
					return sig_exts[i];
				}
			}
		}
	}

	if (ast_asprintf(&fn, "%s.sig", path) < 0) {
		return NULL;
	}
	found = !access(fn, R_OK);
	ast_free(fn);
	if (found) {
		return "sig";
	}

	return sig_probe(path);
}

static char *handle_cli_signature_test(struct ast_cli_entry *e, int cmd, struct ast_cli_args *a)
{
	struct sig_template *t;
	const char *ext, *name;
	char path[512];
	int template_ms, window_ms;
	int last, i;

	switch (cmd) {
	case CLI_INIT:
		e->command = "amd signature test";
		e->usage =
			"Usage: amd signature test <reference> <sample> [<sample>...] [templateLength]\n"
			"       Score one or more recordings against a reference, without placing a\n"
			"       call. The reference is a .sig file or a recording, named absolutely\n"
			"       or relative to the amd directory under the configuration directory,\n"
			"       such as fr/ios-screening-2. The samples are resolved as sound files.\n"
			"       Prints the best score, where in the sample it occurred, the mean\n"
			"       level of the match window there, and where in the reference the\n"
			"       best path started. A high score at a level below the configured\n"
			"       silence_threshold is a match against a noise floor and is\n"
			"       suppressed in AMD.\n";
		return NULL;
	case CLI_GENERATE:
		return NULL;
	}

	if (a->argc < 5) {
		return CLI_SHOWUSAGE;
	}

	ast_mutex_lock(&config_lock);
	template_ms = dfltTemplateLength;
	window_ms = dfltWindow;
	ast_mutex_unlock(&config_lock);

	/* A trailing all-digits argument is the template length, not a sample. */
	last = a->argc;
	if (a->argc >= 6 && strspn(a->argv[a->argc - 1], "0123456789") == strlen(a->argv[a->argc - 1])) {
		template_ms = atoi(a->argv[a->argc - 1]);
		if (template_ms < SIG_MIN_TEMPLATE_MS || template_ms > SIG_MAX_TEMPLATE_MS) {
			ast_cli(a->fd, "templateLength must be between %d and %d\n",
				SIG_MIN_TEMPLATE_MS, SIG_MAX_TEMPLATE_MS);
			return CLI_FAILURE;
		}
		last = a->argc - 1;
	}

	if (!(ext = sig_resolve_reference(a->argv[3], path, sizeof(path)))) {
		ast_cli(a->fd, "No reference found for '%s'\n", a->argv[3]);
		return CLI_FAILURE;
	}
	name = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
	if (!(t = sig_template_build(name, path, ext, "", template_ms, window_ms))) {
		ast_cli(a->fd, "Unable to build a reference from '%s'\n", a->argv[3]);
		return CLI_FAILURE;
	}

	ast_cli(a->fd, "Reference %s (%s.%s), %dms stored, paths of at least %dms\n\n",
		a->argv[3], path, ext, t->len * SIG_HOP_MS, t->win * SIG_HOP_MS);
	ast_cli(a->fd, "%-44s %6s %10s %7s %8s\n", "SAMPLE", "SCORE", "AT", "LEVEL", "OFFSET");

	for (i = 4; i < last; i++) {
		struct amd_signature *d;
		const char *sext;
		int16_t *samples;
		int nsamples = 0;

		if (!(sext = sig_probe(a->argv[i])) || !(samples = sig_load_slin(a->argv[i], sext, &nsamples))) {
			ast_cli(a->fd, "%-44s %6s %10s %7s %8s\n", a->argv[i], "-", "-", "-", "-");
			continue;
		}

		/*
		 * A threshold over 100 is never reached, so the whole sample is scored,
		 * and a silence floor of 0 reports the raw score: the level is printed
		 * alongside instead.
		 */
		if (!(d = sig_detector_alloc(101.0f, 0, SIG_DEF_DEBOUNCE))) {
			ast_free(samples);
			continue;
		}
		ao2_ref(t, +1);
		if (sig_detector_add(d, t) || sig_detector_ready(d)) {
			sig_detector_free(d);
			ast_free(samples);
			continue;
		}

		sig_detector_feed(d, samples, nsamples);

		ast_cli(a->fd, "%-44s %6.1f %8dms %7d %6dms\n", a->argv[i], d->best,
			d->best_frame < 0 ? 0 : (d->best_frame - t->win) * SIG_HOP_MS,
			(int) d->best_level, d->best_offset);

		sig_detector_free(d);
		ast_free(samples);
	}

	ao2_ref(t, -1);

	return CLI_SUCCESS;
}

static char *handle_cli_signature_show(struct ast_cli_entry *e, int cmd, struct ast_cli_args *a)
{
	struct dirent *de;
	char root[512];
	int template_ms, window_ms, n = 0;
	DIR *dir;

	switch (cmd) {
	case CLI_INIT:
		e->command = "amd signature show templates";
		e->usage =
			"Usage: amd signature show templates\n"
			"       List the references a call in each language would be matched\n"
			"       against, one language per directory under the amd directory of\n"
			"       the configuration directory.\n";
		return NULL;
	case CLI_GENERATE:
		return NULL;
	}

	if (a->argc != 4) {
		return CLI_SHOWUSAGE;
	}

	ast_mutex_lock(&config_lock);
	template_ms = dfltTemplateLength;
	window_ms = dfltWindow;
	ast_mutex_unlock(&config_lock);

	snprintf(root, sizeof(root), "%s/%s", ast_config_AST_CONFIG_DIR, SIG_DIR);
	if (!(dir = opendir(root))) {
		ast_cli(a->fd, "No reference directory %s\n", root);
		return CLI_SUCCESS;
	}

	ast_cli(a->fd, "%-10s %-30s %8s %8s %s\n", "LANGUAGE", "NAME", "STORED", "SOURCES", "SEED");
	while ((de = readdir(dir))) {
		char sub[800];
		struct sig_lang *l;
		int i;

		snprintf(sub, sizeof(sub), "%s/%s", root, de->d_name);
		if (de->d_name[0] == '.' || !sig_is_dir(sub) || !(l = sig_lang_get(de->d_name, template_ms, window_ms))) {
			continue;
		}
		for (i = 0; i < l->ntmpl; i++) {
			const struct sig_template *t = l->tmpl[i];

			ast_cli(a->fd, "%-10s %-30s %6dms %8d %s\n", l->lang, t->name,
				t->len * SIG_HOP_MS, t->nsources, t->seed);
			n++;
		}
		ao2_ref(l, -1);
	}
	closedir(dir);

	ast_cli(a->fd, "\n%d reference%s\n", n, n == 1 ? "" : "s");

	return CLI_SUCCESS;
}

static struct ast_cli_entry cli_signature[] = {
	AST_CLI_DEFINE(handle_cli_signature_test, "Score recordings against a reference."),
	AST_CLI_DEFINE(handle_cli_signature_show, "List the references of each language."),
};

#ifdef TEST_FRAMEWORK
/*!
 * \brief A signal which, like speech, changes smoothly from hop to hop:
 * harmonics on a gliding pitch with a moving tilt, broken into syllables.
 * White noise would not do as a reference, because it only matches at exact
 * alignments.
 */
static int16_t *sig_test_signal(int n)
{
	int16_t *x = ast_malloc(n * sizeof(*x));
	double phase = 0.0;
	int i, h;

	if (!x) {
		return NULL;
	}
	for (i = 0; i < n; i++) {
		double t = (double) i / DEFAULT_SAMPLE_RATE;
		double f0 = 180 + 60 * sin(2 * M_PI * 0.8 * t) + 40 * sin(2 * M_PI * 2.3 * t);
		double tilt = 0.5 + 0.5 * sin(2 * M_PI * 1.7 * t);
		double y = 0.0;

		phase += 2 * M_PI * f0 / DEFAULT_SAMPLE_RATE;
		for (h = 1; h < 16; h++) {
			y += sin(h * phase) * (h % 2 ? tilt : 1 - tilt) / h;
		}
		x[i] = sin(2 * M_PI * 3 * t) > -0.6 ? (int16_t) (y * 3000) : 0;
	}

	return x;
}

/*! \brief The best score and match of \a t over \a x, as a call would see them. */
static void sig_test_run(struct sig_template *t, const int16_t *x, int n, float *best, int *matched)
{
	struct amd_signature *d = sig_detector_alloc(SIG_DEF_THRESHOLD, SIG_DEF_SILENCE, SIG_DEF_DEBOUNCE);

	*best = -1.0f;
	*matched = 0;
	if (!d) {
		return;
	}
	ao2_ref(t, +1);
	if (!sig_detector_add(d, t) && !sig_detector_ready(d)) {
		*matched = sig_detector_feed(d, x, n);
		*best = d->best;
	}
	sig_detector_free(d);
}

AST_TEST_DEFINE(signature_local)
{
	struct sig_builder b = { 0 }, back = { 0 };
	struct sig_template *t = NULL, *t2 = NULL;
	enum ast_test_result_state res = AST_TEST_FAIL;
	int16_t *x = NULL, *noise = NULL;
	char fn[] = "/tmp/amd_signature_XXXXXX";
	float best, best2;
	int n = 3 * DEFAULT_SAMPLE_RATE, matched, fd, i, j;
	struct sig_fe fe;
	FILE *fp;

	switch (cmd) {
	case TEST_INIT:
		info->name = "signature_local";
		info->category = "/apps/amd/";
		info->summary = "Signature matching by local alignment";
		info->description =
			"Builds a reference from a speech-like signal, checks that it matches "
			"itself and not white noise, and that it survives a round trip through "
			"a .sig file.";
		return AST_TEST_NOT_RUN;
	case TEST_EXECUTE:
		break;
	}

	if (!(x = sig_test_signal(n)) || !(noise = ast_malloc(5 * DEFAULT_SAMPLE_RATE * sizeof(*noise)))) {
		goto done;
	}
	for (i = 0; i < 5 * DEFAULT_SAMPLE_RATE; i++) {
		/* A sum of uniforms is close enough to Gaussian white noise here. */
		noise[i] = (int16_t) ((ast_random() % 2001 + ast_random() % 2001 + ast_random() % 2001 - 3000) * 1.5);
	}

	sig_fe_reset(&fe);
	sig_fe_feed(&fe, x, n, sig_builder_frame, &b);
	if (!(t = sig_template_from_frames("test", "test", "", &b, SIG_DEF_TEMPLATE_MS, SIG_DEF_WINDOW_MS))) {
		ast_test_status_update(test, "could not build a reference\n");
		goto done;
	}

	sig_test_run(t, x, n, &best, &matched);
	ast_test_status_update(test, "reference against itself: best %.2f, matched %d\n", best, matched);
	if (best < 99.9f || !matched) {
		goto done;
	}

	sig_test_run(t, noise, 5 * DEFAULT_SAMPLE_RATE, &best2, &matched);
	ast_test_status_update(test, "reference against white noise: best %.2f, matched %d\n", best2, matched);
	if (matched) {
		goto done;
	}

	/* Written as the sweep script writes it, to five digits, and read back. */
	if ((fd = mkstemp(fn)) < 0 || !(fp = fdopen(fd, "w"))) {
		goto done;
	}
	fprintf(fp, "; test\nformat: %s\nfrontend: %s\nseed: test\nsources: 1\nframes: %d\n",
		SIG_FORMAT, sig_frontend, t->len);
	for (i = 0; i < t->len; i++) {
		fprintf(fp, "%.5g", logf(t->ptot[i] + 1e-9f));
		for (j = 0; j < SIG_NBANDS; j++) {
			fprintf(fp, " %.5g", t->v[i * SIG_NBANDS + j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	i = sig_frames_from_sig(fn, &back);
	unlink(fn);
	if (i || !(t2 = sig_template_from_frames("test", "test", "", &back, SIG_DEF_TEMPLATE_MS, SIG_DEF_WINDOW_MS))) {
		ast_test_status_update(test, "could not read the reference back\n");
		goto done;
	}
	sig_test_run(t2, x, n, &best2, &matched);
	ast_test_status_update(test, "after a round trip through a .sig file: best %.2f, matched %d\n", best2, matched);
	if (fabsf(best2 - best) > 0.05f || !matched) {
		goto done;
	}

	res = AST_TEST_PASS;

done:
	ao2_cleanup(t);
	ao2_cleanup(t2);
	ast_free(b.fr);
	ast_free(back.fr);
	ast_free(x);
	ast_free(noise);

	return res;
}
#endif

int amd_signature_init(void)
{
	sig_init_coeffs();

	templates = ao2_container_alloc_hash(AO2_ALLOC_OPT_LOCK_MUTEX, 0, 7,
		sig_template_hash, NULL, sig_template_cmp);
	langs = ao2_container_alloc_hash(AO2_ALLOC_OPT_LOCK_MUTEX, 0, 7,
		sig_lang_hash, NULL, sig_lang_cmp);
	if (!templates || !langs) {
		ao2_cleanup(templates);
		ao2_cleanup(langs);
		templates = langs = NULL;
		return -1;
	}

	ast_cli_register_multiple(cli_signature, ARRAY_LEN(cli_signature));
	AST_TEST_REGISTER(signature_local);

	return 0;
}

void amd_signature_cleanup(void)
{
	AST_TEST_UNREGISTER(signature_local);
	ast_cli_unregister_multiple(cli_signature, ARRAY_LEN(cli_signature));
	ao2_cleanup(langs);
	langs = NULL;
	ao2_cleanup(templates);
	templates = NULL;
}
