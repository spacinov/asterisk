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
 */

#ifndef _AMD_SIGNATURE_H
#define _AMD_SIGNATURE_H

struct ast_channel;
struct ast_config;
struct ast_frame;

/*! A matcher for one call. */
struct amd_signature;

/*!
 * \brief Set up the filterbank, the reference caches and the CLI commands.
 *
 * \retval 0 on success, -1 on failure
 */
int amd_signature_init(void);

/*! \brief Undo amd_signature_init(). */
void amd_signature_cleanup(void);

/*!
 * \brief Take the defaults from the [signature] section of amd.conf.
 *
 * Without the section, or unless it sets enabled, amd_signature_start() does
 * not start a matcher. The reference caches are dropped.
 */
void amd_signature_load_config(struct ast_config *cfg);

/*!
 * \brief Drop the reference caches, so that the next call rescans the
 * reference directories.
 *
 * To be called on every reload, whether or not amd.conf changed: references
 * are files of their own, added and removed without touching amd.conf.
 */
void amd_signature_flush(void);

/*!
 * \brief Start matching a call against the references of its language.
 *
 * These are the .sig files in the amd/<language>/ directory under the
 * configuration directory, the language being the channel's.
 *
 * \retval a matcher, to be released with amd_signature_free()
 * \retval NULL if matching is not enabled, or the language has no references
 */
struct amd_signature *amd_signature_start(struct ast_channel *chan);

/*!
 * \brief Feed one frame of signed linear audio to the matcher.
 *
 * Frames other than signed linear voice are ignored.
 *
 * \retval 1 once a template has matched, 0 otherwise
 */
int amd_signature_feed(struct amd_signature *s, struct ast_frame *f);

/*! \brief The name of the reference which matched, or "" if none has. */
const char *amd_signature_name(struct amd_signature *s);

/*! \brief The highest similarity seen so far, from 0 to 100. */
float amd_signature_score(struct amd_signature *s);

/*! \brief Release a matcher. NULL is allowed. */
void amd_signature_free(struct amd_signature *s);

#endif /* _AMD_SIGNATURE_H */
