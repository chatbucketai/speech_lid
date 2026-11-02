/*
 * Copyright 2026 Pramiteeh
 *
 * lid9 — spoken language identification, C API.
 *
 * Identifies which of 9 languages is being spoken in a chunk of audio:
 * english, hindi, telugu, tamil, kannada, malayalam, marathi, bengali,
 * gujarati.
 *
 * Feature extraction (STFT, mel filterbank, log, normalisation) is compiled
 * into the ONNX graph, so this API takes raw PCM samples. There is no feature
 * configuration to get wrong and no way for the caller's front end to drift
 * from what the model was trained with.
 *
 * Typical use:
 *
 *     Lid9 *lid = lid9_create(NULL, 4);          // NULL -> embedded model
 *     Lid9Result r;
 *     if (lid9_identify(lid, samples, n, 16000, &r) == LID9_OK) {
 *         printf("%s (%.3f)\n",
 *                lid9_language_name(lid, r.language_index), r.confidence);
 *     }
 *     lid9_destroy(lid);
 *
 * Threading: a Lid9 handle is safe to call lid9_identify() on from multiple
 * threads concurrently. Creation and destruction are not thread-safe with
 * respect to in-flight calls. Create one handle at startup and share it —
 * construction costs roughly 100x a single inference.
 */

#ifndef LID9_H
#define LID9_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define LID9_API __declspec(dllexport)
#else
#  define LID9_API __attribute__((visibility("default")))
#endif

/* Upper bound on languages, so Lid9Result needs no heap allocation. The
 * loaded model reports its real count via lid9_num_languages(). */
#define LID9_MAX_LANGUAGES 32

typedef struct Lid9 Lid9;

typedef enum {
    LID9_OK = 0,
    LID9_ERR_INVALID_ARG = -1,   /* NULL pointer or nonsensical size/rate */
    LID9_ERR_MODEL_LOAD = -2,    /* file missing, corrupt, or not a lid9 model */
    LID9_ERR_AUDIO_TOO_SHORT = -3, /* below LID9_MIN_SECONDS of audio */
    LID9_ERR_INFERENCE = -4,     /* onnxruntime failed during Run() */
    LID9_ERR_OUT_OF_MEMORY = -5,
    LID9_ERR_NO_EMBEDDED_MODEL = -6 /* built without an embedded model, and
                                     * lid9_create() was given a NULL path */
} Lid9Status;

/* Shorter clips than this cannot produce a meaningful result: the STFT needs a
 * full analysis frame and the res2net stack needs a few frames of context. */
#define LID9_MIN_SECONDS 0.3f

typedef struct {
    /* Index of the winning language; pass to lid9_language_name(). */
    int   language_index;
    /* Probability of the winning language, in [0, 1]. */
    float confidence;
    /* Probability of every language, indexed the same way as
     * lid9_language_name(). Entries beyond lid9_num_languages() are 0. */
    float scores[LID9_MAX_LANGUAGES];
    /* Seconds of audio actually classified, after truncation to the model's
     * analysis window. */
    float duration_seconds;
} Lid9Result;

/*
 * Create a recogniser.
 *
 *   model_path  Path to a .onnx file, or NULL to use the model embedded in
 *               this library at build time. Passing NULL when the library was
 *               built without an embedded model returns LID9_ERR_NO_EMBEDDED_MODEL
 *               via lid9_last_status().
 *   num_threads Intra-op threads. 1-2 is usually right on mobile; more threads
 *               on a short clip mostly buys scheduling overhead.
 *
 * Returns NULL on failure; call lid9_last_status() for the reason.
 */
LID9_API Lid9 *lid9_create(const char *model_path, int num_threads);

/* Create from a model already in memory (e.g. mapped from an Android asset).
 * The buffer is copied, so the caller may free it on return. */
LID9_API Lid9 *lid9_create_from_memory(const void *model_data,
                                       size_t model_size,
                                       int num_threads);

LID9_API void lid9_destroy(Lid9 *lid);

/*
 * Identify the language of one clip.
 *
 *   samples      Mono PCM, float, nominally in [-1, 1].
 *   num_samples  Length of `samples`.
 *   sample_rate  Rate of `samples`. Anything other than the model's rate
 *                (16000) is linearly resampled internally.
 *   out          Filled in on LID9_OK.
 *
 * Audio longer than the analysis window (10 s by default) is truncated;
 * accuracy saturates well before that and the cost is linear in length.
 */
LID9_API Lid9Status lid9_identify(Lid9 *lid,
                                  const float *samples,
                                  size_t num_samples,
                                  int sample_rate,
                                  Lid9Result *out);

/* Same, for 16-bit signed PCM — the format Android's AudioRecord produces.
 * Saves the caller a conversion pass. */
LID9_API Lid9Status lid9_identify_i16(Lid9 *lid,
                                      const short *samples,
                                      size_t num_samples,
                                      int sample_rate,
                                      Lid9Result *out);

/* Number of languages this model predicts. */
LID9_API int lid9_num_languages(const Lid9 *lid);

/* Name of language `index`, or NULL if out of range. The returned pointer is
 * owned by the handle and stays valid until lid9_destroy(). */
LID9_API const char *lid9_language_name(const Lid9 *lid, int index);

/* Seconds of audio the model looks at; longer input is truncated. */
LID9_API float lid9_window_seconds(const Lid9 *lid);

/* Status of the last failed lid9_create*() on this thread. */
LID9_API Lid9Status lid9_last_status(void);

/* Human-readable form of a status code. Never NULL. */
LID9_API const char *lid9_status_string(Lid9Status status);

/* Library version, e.g. "1.0.0". */
LID9_API const char *lid9_version(void);

/* Whether this build has a model compiled into it. */
LID9_API int lid9_has_embedded_model(void);

#ifdef __cplusplus
}
#endif

#endif /* LID9_H */
