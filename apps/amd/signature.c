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
 * Matches the inbound audio against one or more reference recordings. This is
 * aimed at call screening services such as the one recent iOS versions place in
 * front of a call: they answer, play a fixed synthetic prompt, and only connect
 * a human once the caller has said something. To AMD() that prompt is
 * indistinguishable from an answering machine greeting, because on energy and
 * timing features it is one. The prompt is however a fixed recording, so it can
 * be recognised directly, and AMD() reports it as SCREENED.
 *
 * The detector runs a 12 band filterbank over 10ms hops, converts each hop to a
 * gain invariant log spectral shape, and slides the reference over the inbound
 * audio scoring cosine similarity. A verdict is available roughly one second
 * after the prompt starts, earlier than AMD() would reach MACHINE on the same
 * audio, which leaves the dialplan time to respond while the prompt is still
 * playing.
 *
 * \author Jeremy Lainé <jeremy.laine@m4x.org>
 */

#include "asterisk.h"

#include <math.h>
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
#include "asterisk/translate.h"
#include "asterisk/utils.h"

#include "signature.h"

/*! Number of filterbank bands. */
#define SIG_NBANDS		12
/*! Analysis hop, in samples at 8kHz. 80 samples is 10ms. */
#define SIG_HOP			80
/*! Hops per millisecond. */
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
/*! Template frames this far below the template's loudest frame are not scored. -25dB. */
#define SIG_GATE		0.0031622777
/*! The template starts at the first frame this far below the loudest one. -30dB. */
#define SIG_ONSET		0.001

#define SIG_MAX_TEMPLATES	16
#define SIG_MIN_TEMPLATE_MS	200
#define SIG_MAX_TEMPLATE_MS	8000
#define SIG_MIN_WINDOW_MS	200
/*!
 * Defaults chosen by cross-validating against 793 labelled calls: 36 screened,
 * 757 answering machine, human or blocked. Four references drawn from real
 * calls detect all 36 with no false positive.
 *
 * The match window is what a call is compared against; the template is longer,
 * and the window is tried at every offset within it, so that a call whose
 * beginning is missing still matches further in. 1200ms is the shortest window
 * that keeps the threshold forgiving, and 3000ms of template covers being up to
 * 1800ms late. 4000ms adds nothing.
 *
 * The reference recordings matter far more than any of these numbers. Four
 * references captured by hand, one per voice, detect 21 of the 36 screened calls
 * at these settings; adding two cut from real calls, for a voice heard with
 * different intonation and for one not captured at all, detects all 36.
 */
#define SIG_DEF_TEMPLATE_MS	3000
#define SIG_DEF_WINDOW_MS	1200
#define SIG_DEF_OFFSET_MS	100
#define SIG_DEF_THRESHOLD	95
/*!
 * Minimum mean absolute sample value, over the window being scored, for a match
 * to count. Removing the per frame mean and normalising discards absolute level
 * by design, so that a quiet prompt matches a loud reference; the cost is that
 * near silence still produces a unit vector, and a steady noise floor can sit
 * within a few points of a template. Two calls in a sample of 546 scored above
 * 0.91 on stretches whose level was under 45, some thirty times below the
 * quietest real prompt. This is the floor that rules those out. It is the same
 * measure and default as AMD's own silenceThreshold.
 */
#define SIG_DEF_SILENCE		256
/*!
 * Consecutive hops a template must stay above the threshold before it counts.
 * Two is enough to rule out a single anomalous frame, and three is too many:
 * a good match can peak sharply, and requiring a third hop lost two of 36
 * screened calls which scored 99, while gaining nothing against the 757 that
 * were not screened.
 */
#define SIG_DEF_DEBOUNCE	2

/*! Defaults read from the [signature] section of amd.conf. */
AST_MUTEX_DEFINE_STATIC(config_lock);
static char *dfltTemplates;
static int dfltThreshold = SIG_DEF_THRESHOLD;
static int dfltTemplateLength = SIG_DEF_TEMPLATE_MS;
static int dfltWindow = SIG_DEF_WINDOW_MS;
static int dfltOffset = SIG_DEF_OFFSET_MS;
static int dfltSilence = SIG_DEF_SILENCE;

/*! Extensions probed when resolving a reference recording, in preference order. */
static const char * const sig_exts[] = { "sln", "wav", "WAV", "ulaw", "alaw", "g722", "sln16", "gsm" };

/*! One biquad section. The bandpass form has b1 == 0, so it is not stored. */
struct sig_biquad {
	float b0;
	float b2;
	float a1;
	float a2;
};

static struct sig_biquad sig_bq[SIG_NBANDS];

/*! One 10ms analysis frame: a unit length spectral shape, its total power, and its level. */
struct sig_frame {
	float v[SIG_NBANDS];
	float ptot;
	float level;		/*!< mean absolute sample value over the hop */
};

/*! Filterbank state. Identical for the reference and for the live audio. */
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

/*!
 * One way of matching a template: a window of it, starting \a offset frames in.
 *
 * A call whose first seconds are missing cannot match a window anchored at the
 * start of the prompt, however the audio is aligned, because the audio the
 * window describes was never recorded. Offsetting the window into the template
 * covers that, and covers amounts of loss no reference recording happens to
 * demonstrate.
 */
struct sig_window {
	int offset;		/*!< frames into the template */
	int active;		/*!< frames of this window which pass the energy gate */
	unsigned char *act;	/*!< win entries, owned by the template */
};

/*! A loaded reference recording, reduced to the frames worth scoring. */
struct sig_template {
	int len;		/*!< stored frames */
	int win;		/*!< frames compared at a time */
	float *v;		/*!< len * SIG_NBANDS */
	struct sig_window *w;	/*!< nwin */
	unsigned char *actbuf;	/*!< nwin * win, carved up between the windows */
	int nwin;
	char *key;		/*!< cache key, "<path>|<ms>|<ms>|<ms>" */
	char name[80];		/*!< as named by the caller */
	char path[512];		/*!< resolved, without extension */
};

/*! Streaming matcher over one or more templates. */
struct amd_signature {
	struct sig_fe fe;
	struct sig_template *tmpl[SIG_MAX_TEMPLATES];
	int hits[SIG_MAX_TEMPLATES];
	int ntmpl;
	int cap;		/*!< ring capacity, the longest template */
	float *ring;		/*!< cap * SIG_NBANDS */
	float *rlevel;		/*!< cap, the level of each ring frame */
	int head;		/*!< where the next frame goes */
	int count;		/*!< frames pushed so far */
	int threshold;
	int silence;		/*!< windows quieter than this cannot match */
	int debounce;
	int best;		/*!< highest score seen, percent */
	int best_frame;
	int best_offset;	/*!< ms into the template where best was seen */
	float best_level;	/*!< window level where best was seen */
	int matched;		/*!< index into tmpl, or -1 */
	int match_frame;
};

static struct ao2_container *templates;

typedef void (*sig_frame_fn)(void *arg, const struct sig_frame *fr);

/*! \brief Build the RBJ constant skirt bandpass sections, once. */
static void sig_init_coeffs(void)
{
	int j;
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
 * \brief Find which extension a reference recording exists in.
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
 * \brief Resolve a name the way sound files are resolved, honouring the language.
 *
 * fileexists_core() already does this but is private to main/file.c, and the
 * public ast_fileexists() discards the path it found, so the search is repeated
 * here: the language, the language without its dialect suffix, no language at
 * all, then the default language.
 *
 * \retval the extension found, or NULL if the name does not resolve
 */
static const char *sig_resolve(const char *name, const char *lang, char *buf, size_t buflen)
{
	char stripped[64] = "";
	const char *langs[4];
	const char *ext;
	int i, nlangs = 0;

	if (name[0] == '/') {
		ast_copy_string(buf, name, buflen);
		return sig_probe(buf);
	}

	if (!ast_strlen_zero(lang)) {
		char *end;

		langs[nlangs++] = lang;
		ast_copy_string(stripped, lang, sizeof(stripped));
		if ((end = strrchr(stripped, '_'))) {
			*end = '\0';
			langs[nlangs++] = stripped;
		}
	}
	langs[nlangs++] = NULL;
	if (ast_strlen_zero(lang) || strcmp(lang, "en")) {
		langs[nlangs++] = "en";
	}

	for (i = 0; i < nlangs; i++) {
		if (!langs[i]) {
			ast_copy_string(buf, name, buflen);
		} else if (ast_language_is_prefix) {
			snprintf(buf, buflen, "%s/%s", langs[i], name);
		} else {
			const char *slash = strrchr(name, '/');
			int off = slash ? slash - name + 1 : 0;

			snprintf(buf, buflen, "%.*s%s/%s", off, name, langs[i], name + off);
		}
		if ((ext = sig_probe(buf))) {
			return ext;
		}
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

/*! Collects every frame the filterbank produces while a reference is decoded. */
struct sig_builder {
	struct sig_frame *fr;
	int n;
	int cap;
	int failed;
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

static void sig_template_destroy(void *obj)
{
	struct sig_template *t = obj;

	ast_free(t->v);
	ast_free(t->w);
	ast_free(t->actbuf);
	ast_free(t->key);
}

/*!
 * \brief Reduce a reference recording to the windows worth scoring.
 *
 * The template starts at the first frame which is not silence and runs for
 * template_ms. It is then carved into overlapping windows of window_ms, every
 * offset_ms, so that a call which is missing its first seconds still matches a
 * window further into the prompt.
 *
 * Within each window, frames more than 25dB below that window's loudest frame
 * are left out of the score: they carry no reliable spectral shape and are the
 * first thing noise corrupts. The gate is per window, not per template, because
 * a quiet window of a loud template still has to be scored on its own terms.
 */
static struct sig_template *sig_template_build(const char *name, const char *path,
	const char *ext, const char *key, int template_ms, int window_ms, int offset_ms)
{
	struct sig_builder b = { 0 };
	struct sig_template *t;
	struct sig_fe fe;
	int16_t *samples;
	int nsamples = 0;
	float peak = 0.0f;
	int onset, len, win, step, nwin, i, j, o;

	if (!(samples = sig_load_slin(path, ext, &nsamples))) {
		ast_log(LOG_WARNING, "AMD: signature: unable to read '%s.%s'\n", path, ext);
		return NULL;
	}

	sig_fe_reset(&fe);
	sig_fe_feed(&fe, samples, nsamples, sig_builder_frame, &b);
	ast_free(samples);

	if (b.failed || !b.n) {
		ast_free(b.fr);
		return NULL;
	}

	for (i = 0; i < b.n; i++) {
		if (b.fr[i].ptot > peak) {
			peak = b.fr[i].ptot;
		}
	}
	for (onset = 0; onset < b.n && b.fr[onset].ptot < peak * SIG_ONSET; onset++) {
	}
	if (onset >= b.n) {
		ast_log(LOG_WARNING, "AMD: signature: '%s' appears to be silent\n", path);
		ast_free(b.fr);
		return NULL;
	}

	win = window_ms / SIG_HOP_MS;
	step = offset_ms / SIG_HOP_MS;
	if (step < 1) {
		step = 1;
	}
	len = template_ms / SIG_HOP_MS;
	if (len > b.n - onset) {
		len = b.n - onset;
	}
	if (len < win) {
		ast_log(LOG_WARNING, "AMD: signature: '%s' is too short, %dms of audio after silence, need %dms\n",
			path, len * SIG_HOP_MS, window_ms);
		ast_free(b.fr);
		return NULL;
	}
	nwin = (len - win) / step + 1;

	if (!(t = ao2_alloc(sizeof(*t), sig_template_destroy))) {
		ast_free(b.fr);
		return NULL;
	}
	t->v = ast_malloc(len * SIG_NBANDS * sizeof(*t->v));
	t->w = ast_malloc(nwin * sizeof(*t->w));
	t->actbuf = ast_malloc((size_t) nwin * win);
	t->key = ast_strdup(key);
	if (!t->v || !t->w || !t->actbuf || !t->key) {
		ast_free(b.fr);
		ao2_ref(t, -1);
		return NULL;
	}
	t->len = len;
	t->win = win;
	ast_copy_string(t->name, name, sizeof(t->name));
	ast_copy_string(t->path, path, sizeof(t->path));

	for (i = 0; i < len; i++) {
		for (j = 0; j < SIG_NBANDS; j++) {
			t->v[i * SIG_NBANDS + j] = b.fr[onset + i].v[j];
		}
	}

	for (o = 0; o < nwin; o++) {
		struct sig_window *w = &t->w[t->nwin];
		int base = o * step;
		float gate;

		peak = 0.0f;
		for (i = 0; i < win; i++) {
			if (b.fr[onset + base + i].ptot > peak) {
				peak = b.fr[onset + base + i].ptot;
			}
		}
		gate = peak * SIG_GATE;

		w->offset = base;
		w->active = 0;
		w->act = t->actbuf + (size_t) t->nwin * win;
		for (i = 0; i < win; i++) {
			w->act[i] = b.fr[onset + base + i].ptot >= gate;
			if (w->act[i]) {
				w->active++;
			}
		}
		if (w->active) {
			t->nwin++;
		}
	}
	ast_free(b.fr);

	if (!t->nwin) {
		ao2_ref(t, -1);
		return NULL;
	}

	ast_debug(1, "AMD: signature: loaded '%s' from '%s.%s', onset %dms, %dms stored, %d windows of %dms\n",
		name, path, ext, onset * SIG_HOP_MS, t->len * SIG_HOP_MS, t->nwin, t->win * SIG_HOP_MS);

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
 * \brief Fetch a template, loading it the first time it is asked for.
 *
 * \retval a reference the caller must release with ao2_ref(), or NULL
 */
static struct sig_template *sig_template_get(const char *name, const char *lang,
	int template_ms, int window_ms, int offset_ms)
{
	struct sig_template *t;
	char path[512];
	char key[600];
	const char *ext;

	if (!(ext = sig_resolve(name, lang, path, sizeof(path)))) {
		ast_log(LOG_WARNING, "AMD: signature: no reference recording found for '%s'\n", name);
		return NULL;
	}

	snprintf(key, sizeof(key), "%s|%d|%d|%d", path, template_ms, window_ms, offset_ms);

	ao2_lock(templates);
	if (!(t = ao2_find(templates, key, OBJ_SEARCH_KEY | OBJ_NOLOCK))) {
		if ((t = sig_template_build(name, path, ext, key, template_ms, window_ms, offset_ms))) {
			ao2_link_flags(templates, t, OBJ_NOLOCK);
		}
	}
	ao2_unlock(templates);

	return t;
}

static void sig_detector_free(struct amd_signature *d)
{
	int i;

	if (!d) {
		return;
	}
	for (i = 0; i < d->ntmpl; i++) {
		ao2_cleanup(d->tmpl[i]);
	}
	ast_free(d->ring);
	ast_free(d->rlevel);
	ast_free(d);
}

/*! \brief Start an empty matcher; templates are added with sig_detector_add(). */
static struct amd_signature *sig_detector_alloc(int threshold, int silence, int debounce)
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

/*! \brief Add a template to a matcher, which takes over the caller's reference. */
static void sig_detector_add(struct amd_signature *d, struct sig_template *t)
{
	d->tmpl[d->ntmpl++] = t;
	if (t->win > d->cap) {
		d->cap = t->win;
	}
}

/*!
 * \brief Allocate the audio history once every template has been added.
 *
 * \retval 0 on success, -1 if there is nothing to match or no memory
 */
static int sig_detector_ready(struct amd_signature *d)
{
	if (!d->ntmpl) {
		return -1;
	}
	d->ring = ast_calloc(d->cap * SIG_NBANDS, sizeof(*d->ring));
	d->rlevel = ast_calloc(d->cap, sizeof(*d->rlevel));

	return d->ring && d->rlevel ? 0 : -1;
}

/*!
 * \brief Build a matcher over an '&' separated list of reference recordings.
 *
 * \note Takes ownership of nothing; \a names is copied before being split.
 */
static struct amd_signature *sig_detector_new(const char *names, const char *lang,
	int template_ms, int window_ms, int offset_ms, int threshold, int silence, int debounce)
{
	struct amd_signature *d;
	char *list, *name;

	if (!(d = sig_detector_alloc(threshold, silence, debounce))) {
		return NULL;
	}

	list = ast_strdupa(names);
	while ((name = strsep(&list, "&"))) {
		struct sig_template *t;

		name = ast_strip(name);
		if (ast_strlen_zero(name)) {
			continue;
		}
		if (d->ntmpl == SIG_MAX_TEMPLATES) {
			ast_log(LOG_WARNING, "AMD: signature: at most %d templates, ignoring '%s'\n",
				SIG_MAX_TEMPLATES, name);
			break;
		}
		if (!(t = sig_template_get(name, lang, template_ms, window_ms, offset_ms))) {
			continue;
		}
		sig_detector_add(d, t);
	}

	if (sig_detector_ready(d)) {
		sig_detector_free(d);
		return NULL;
	}

	return d;
}

/*!
 * \brief Score every window of every template against the frames in the ring.
 *
 * The ring holds the last window_ms of audio, and a template offers several
 * windows of itself to compare it against. A template counts as matching this
 * hop if any of its windows does.
 */
static void sig_detector_frame(void *arg, const struct sig_frame *fr)
{
	struct amd_signature *d = arg;
	int i, k, j, o;

	memcpy(&d->ring[d->head * SIG_NBANDS], fr->v, SIG_NBANDS * sizeof(*d->ring));
	d->rlevel[d->head] = fr->level;
	d->head = (d->head + 1) % d->cap;
	d->count++;

	for (i = 0; i < d->ntmpl; i++) {
		const struct sig_template *t = d->tmpl[i];
		float level = 0.0f;
		int base, hit = 0;

		if (d->count < t->win) {
			continue;
		}
		base = (d->head - t->win + d->cap) % d->cap;

		for (k = 0; k < t->win; k++) {
			level += d->rlevel[(base + k) % d->cap];
		}
		level /= t->win;

		for (o = 0; o < t->nwin; o++) {
			const struct sig_window *w = &t->w[o];
			float sum = 0.0f;
			int score;

			for (k = 0; k < t->win; k++) {
				const float *a, *b;
				float dot = 0.0f;

				if (!w->act[k]) {
					continue;
				}
				a = &d->ring[((base + k) % d->cap) * SIG_NBANDS];
				b = &t->v[(w->offset + k) * SIG_NBANDS];
				for (j = 0; j < SIG_NBANDS; j++) {
					dot += a[j] * b[j];
				}
				sum += dot;
			}

			score = (int) (100.0f * sum / w->active + 0.5f);
			if (score < 0) {
				score = 0;
			} else if (score > 100) {
				score = 100;
			}

			if (score > d->best) {
				d->best = score;
				d->best_frame = d->count;
				d->best_level = level;
				d->best_offset = w->offset * SIG_HOP_MS;
			}
			if (score >= d->threshold) {
				hit = 1;
			}
		}

		if (level < d->silence) {
			/* Essentially silence: the shape is the noise floor, not a prompt. */
			d->hits[i] = 0;
			continue;
		}

		if (hit) {
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
 * \retval 1 if a template has matched, 0 otherwise
 */
static int sig_detector_feed(struct amd_signature *d, const int16_t *samples, int n)
{
	sig_fe_feed(&d->fe, samples, n, sig_detector_frame, d);

	return d->matched >= 0;
}

/*!
 * \brief Read the [signature] section of amd.conf.
 *
 * It has to be its own section rather than keys in [general]: AMD warns about
 * keys it does not recognise in [general], but silently skips any other
 * category. Without the section, or without templates in it, AMD() runs
 * exactly as it always has.
 */
void amd_signature_load_config(struct ast_config *cfg)
{
	struct ast_variable *var;
	int threshold = SIG_DEF_THRESHOLD;
	int length = SIG_DEF_TEMPLATE_MS;
	int window = SIG_DEF_WINDOW_MS;
	int offset = SIG_DEF_OFFSET_MS;
	int silence = SIG_DEF_SILENCE;
	char *names = NULL;

	for (var = ast_variable_browse(cfg, "signature"); var; var = var->next) {
		if (!strcasecmp(var->name, "templates")) {
			if (!ast_strlen_zero(var->value)) {
				ast_free(names);
				names = ast_strdup(var->value);
			}
		} else if (!strcasecmp(var->name, "threshold")) {
			threshold = atoi(var->value);
			if (threshold < 0 || threshold > 100) {
				ast_log(LOG_WARNING, "AMD: signature: threshold %d out of range at line %d of amd.conf, using %d\n",
					threshold, var->lineno, SIG_DEF_THRESHOLD);
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
		} else if (!strcasecmp(var->name, "offset_step")) {
			offset = atoi(var->value);
			if (offset < SIG_HOP_MS || offset > SIG_MAX_TEMPLATE_MS) {
				ast_log(LOG_WARNING, "AMD: signature: offset_step %d out of range at line %d of amd.conf, using %d\n",
					offset, var->lineno, SIG_DEF_OFFSET_MS);
				offset = SIG_DEF_OFFSET_MS;
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
	ast_free(dfltTemplates);
	dfltTemplates = names;
	dfltThreshold = threshold;
	dfltTemplateLength = length;
	dfltWindow = window;
	dfltOffset = offset;
	dfltSilence = silence;
	ast_mutex_unlock(&config_lock);

	/* Drop cached templates so a reload picks up re-recorded references. */
	if (templates) {
		ao2_callback(templates, OBJ_NODATA | OBJ_MULTIPLE | OBJ_UNLINK, NULL, NULL);
	}

	ast_verb(5, "AMD signature defaults: templates [%s] threshold [%d] templateLength [%d] matchWindow [%d] offsetStep [%d] silenceThreshold [%d]\n",
		S_OR(names, "(none)"), threshold, length, window, offset, silence);
}

struct amd_signature *amd_signature_start(struct ast_channel *chan)
{
	struct amd_signature *d;
	char names[512] = "";
	int threshold, length, window, offset, silence;

	ast_mutex_lock(&config_lock);
	if (dfltTemplates) {
		ast_copy_string(names, dfltTemplates, sizeof(names));
	}
	threshold = dfltThreshold;
	length = dfltTemplateLength;
	window = dfltWindow;
	offset = dfltOffset;
	silence = dfltSilence;
	ast_mutex_unlock(&config_lock);

	if (ast_strlen_zero(names)) {
		return NULL;
	}

	d = sig_detector_new(names, ast_channel_language(chan), length, window, offset,
		threshold, silence, SIG_DEF_DEBOUNCE);
	if (!d) {
		ast_log(LOG_WARNING, "AMD: Channel [%s]. None of the signature templates [%s] could be loaded, detecting without them\n",
			ast_channel_name(chan), names);
		return NULL;
	}

	ast_verb(3, "AMD: Channel [%s]. Signature templates [%s] threshold [%d] templateLength [%d] matchWindow [%d] offsetStep [%d] silenceThreshold [%d]\n",
		ast_channel_name(chan), names, threshold, length, window, offset, silence);

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

int amd_signature_score(struct amd_signature *s)
{
	return s->best;
}

void amd_signature_free(struct amd_signature *s)
{
	sig_detector_free(s);
}

static char *handle_cli_signature_test(struct ast_cli_entry *e, int cmd, struct ast_cli_args *a)
{
	struct sig_template *t;
	const char *ext;
	char path[512];
	char key[600];
	int template_ms, window_ms, offset_ms;
	int last, i;

	switch (cmd) {
	case CLI_INIT:
		e->command = "amd signature test";
		e->usage =
			"Usage: amd signature test <reference> <sample> [<sample>...] [templateLength]\n"
			"       Score one or more recordings against a reference recording,\n"
			"       without placing a call. Names are resolved the same way as for\n"
			"       AMD's signature templates, but without a channel language.\n"
			"       Prints the best score, where in the sample it occurred, and the\n"
			"       mean level of that window. A high score at a level below the\n"
			"       configured silence_threshold is a match against a noise floor and\n"
			"       is suppressed in AMD.\n";
		return NULL;
	case CLI_GENERATE:
		return NULL;
	}

	if (a->argc < 5) {
		return CLI_SHOWUSAGE;
	}

	ast_mutex_lock(&config_lock);
	template_ms = dfltTemplateLength;
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

	if (!(ext = sig_resolve(a->argv[3], "", path, sizeof(path)))) {
		ast_cli(a->fd, "No reference recording found for '%s'\n", a->argv[3]);
		return CLI_FAILURE;
	}
	ast_mutex_lock(&config_lock);
	window_ms = dfltWindow;
	offset_ms = dfltOffset;
	ast_mutex_unlock(&config_lock);

	snprintf(key, sizeof(key), "%s|%d|%d|%d", path, template_ms, window_ms, offset_ms);
	if (!(t = sig_template_build(a->argv[3], path, ext, key, template_ms, window_ms, offset_ms))) {
		ast_cli(a->fd, "Unable to build a template from '%s'\n", a->argv[3]);
		return CLI_FAILURE;
	}

	ast_cli(a->fd, "Reference %s (%s.%s), %dms stored, %d windows of %dms every %dms\n\n",
		a->argv[3], path, ext, t->len * SIG_HOP_MS, t->nwin, t->win * SIG_HOP_MS, offset_ms);
	ast_cli(a->fd, "%-44s %6s %10s %7s %8s\n", "SAMPLE", "SCORE", "AT", "LEVEL", "OFFSET");

	for (i = 4; i < last; i++) {
		struct amd_signature *d;
		const char *sext;
		char spath[512];
		int16_t *samples;
		int nsamples = 0;

		if (!(sext = sig_resolve(a->argv[i], "", spath, sizeof(spath)))
			|| !(samples = sig_load_slin(spath, sext, &nsamples))) {
			ast_cli(a->fd, "%-44s %6s %10s %7s %8s\n", a->argv[i], "-", "-", "-", "-");
			continue;
		}

		/*
		 * A threshold of 101 is never reached, so the whole sample is scored,
		 * and a silence floor of 0 reports the raw score: the level is printed
		 * alongside instead.
		 */
		if (!(d = sig_detector_alloc(101, 0, SIG_DEF_DEBOUNCE))) {
			ast_free(samples);
			continue;
		}
		ao2_ref(t, +1);
		sig_detector_add(d, t);
		if (sig_detector_ready(d)) {
			sig_detector_free(d);
			ast_free(samples);
			continue;
		}

		sig_detector_feed(d, samples, nsamples);

		ast_cli(a->fd, "%-44s %6d %8dms %7d %6dms\n", a->argv[i], d->best,
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
	struct ao2_iterator it;
	struct sig_template *t;
	int n = 0;

	switch (cmd) {
	case CLI_INIT:
		e->command = "amd signature show templates";
		e->usage =
			"Usage: amd signature show templates\n"
			"       List the reference recordings loaded so far.\n";
		return NULL;
	case CLI_GENERATE:
		return NULL;
	}

	if (a->argc != 4) {
		return CLI_SHOWUSAGE;
	}

	ast_cli(a->fd, "%-20s %-36s %8s %8s %8s\n", "NAME", "PATH", "STORED", "WINDOW", "WINDOWS");

	it = ao2_iterator_init(templates, 0);
	while ((t = ao2_iterator_next(&it))) {
		ast_cli(a->fd, "%-20s %-36s %6dms %6dms %8d\n", t->name, t->path,
			t->len * SIG_HOP_MS, t->win * SIG_HOP_MS, t->nwin);
		ao2_ref(t, -1);
		n++;
	}
	ao2_iterator_destroy(&it);

	ast_cli(a->fd, "\n%d template%s loaded\n", n, n == 1 ? "" : "s");

	return CLI_SUCCESS;
}

static struct ast_cli_entry cli_signature[] = {
	AST_CLI_DEFINE(handle_cli_signature_test, "Score recordings against a reference recording."),
	AST_CLI_DEFINE(handle_cli_signature_show, "List loaded reference recordings."),
};

int amd_signature_init(void)
{
	sig_init_coeffs();

	templates = ao2_container_alloc_hash(AO2_ALLOC_OPT_LOCK_MUTEX, 0, 7,
		sig_template_hash, NULL, sig_template_cmp);
	if (!templates) {
		return -1;
	}

	ast_cli_register_multiple(cli_signature, ARRAY_LEN(cli_signature));

	return 0;
}

void amd_signature_cleanup(void)
{
	ast_cli_unregister_multiple(cli_signature, ARRAY_LEN(cli_signature));
	ao2_cleanup(templates);
	templates = NULL;

	ast_mutex_lock(&config_lock);
	ast_free(dfltTemplates);
	dfltTemplates = NULL;
	ast_mutex_unlock(&config_lock);
}
