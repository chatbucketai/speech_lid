package ai.pramiteeh.lid9;

import androidx.annotation.Nullable;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Spoken language identification for 9 languages: english, hindi, telugu,
 * tamil, kannada, malayalam, marathi, bengali, gujarati.
 *
 * <p>Feature extraction runs inside the model, so this takes raw PCM straight
 * from {@code AudioRecord} with no front-end configuration to get wrong.
 *
 * <pre>{@code
 * // Once, at app or service startup — creation costs ~100x one inference.
 * LanguageId lid = LanguageId.fromEmbeddedModel(2);
 *
 * short[] pcm = ...;                       // 16 kHz mono from AudioRecord
 * LanguageId.Result r = lid.identify(pcm, 16000);
 * Log.i("lid", r.language + " " + r.confidence);
 *
 * lid.close();                             // when you are done with it
 * }</pre>
 *
 * <p>{@link #identify} is safe to call from multiple threads on one instance.
 * {@link #close()} is not — do not close while calls are in flight.
 */
public final class LanguageId implements Closeable {

    static {
        // Load in dependency order: liblid9_jni -> liblid9 -> libonnxruntime.
        //
        // This ordering is required, not just tidy. Before API 23 the Android
        // linker does not resolve a library's transitive DT_NEEDED entries from
        // the APK's native directory, so loading "lid9" first throws
        // UnsatisfiedLinkError on API 21-22 because libonnxruntime.so has not
        // been mapped yet. Loading each dependency explicitly, deepest first,
        // is correct on every API level.
        System.loadLibrary("onnxruntime");
        System.loadLibrary("lid9");
        System.loadLibrary("lid9_jni");
    }

    /** Minimum audio the model will accept. */
    public static final float MIN_SECONDS = 0.3f;

    private long handle;
    private final List<String> languages;

    /** Outcome of one identification. */
    public static final class Result {
        /** Winning language name, e.g. {@code "hindi"}. */
        public final String language;
        /** Its index into {@link #getLanguages()}. */
        public final int index;
        /** Probability of {@link #language}, in [0, 1]. */
        public final float confidence;
        /** Probability of every language, parallel to {@link #getLanguages()}. */
        public final float[] scores;

        Result(String language, int index, float confidence, float[] scores) {
            this.language = language;
            this.index = index;
            this.confidence = confidence;
            this.scores = scores;
        }

        @Override
        public String toString() {
            return language + " (" + confidence + ")";
        }
    }

    /** Raised when the native layer refuses the input or fails to load. */
    public static final class LidException extends RuntimeException {
        public final int status;

        LidException(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    private LanguageId(long handle) {
        this.handle = handle;
        int n = nativeNumLanguages(handle);
        List<String> names = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            names.add(nativeLanguageName(handle, i));
        }
        this.languages = Collections.unmodifiableList(names);
    }

    /**
     * Use the model compiled into the library.
     *
     * @param numThreads 1-2 is usually right on a phone; more threads on a
     *                   short clip mostly buys scheduling overhead.
     */
    public static LanguageId fromEmbeddedModel(int numThreads) {
        long h = nativeCreate(null, numThreads);
        if (h == 0) throw failure();
        return new LanguageId(h);
    }

    /** Load a model from the filesystem. */
    public static LanguageId fromFile(String modelPath, int numThreads) {
        long h = nativeCreate(modelPath, numThreads);
        if (h == 0) throw failure();
        return new LanguageId(h);
    }

    /**
     * Load a model held in memory — e.g. read out of {@code assets/} with
     * {@code AssetManager.open()}. The bytes are copied natively, so the array
     * can be released on return.
     */
    public static LanguageId fromBytes(byte[] model, int numThreads) {
        long h = nativeCreateFromBytes(model, numThreads);
        if (h == 0) throw failure();
        return new LanguageId(h);
    }

    private static LidException failure() {
        int status = nativeLastStatus();
        return new LidException(status, nativeStatusString(status));
    }

    /** Languages this model predicts, in score order. */
    public List<String> getLanguages() {
        return languages;
    }

    /**
     * Identify the language of 16-bit PCM — the format {@code AudioRecord}
     * produces, so no conversion is needed on the Java side.
     *
     * @param sampleRate rate of {@code pcm}; anything other than 16000 is
     *                   resampled natively.
     * @throws LidException if the clip is under {@link #MIN_SECONDS} or
     *                      inference fails.
     */
    public Result identify(short[] pcm, int sampleRate) {
        checkOpen();
        float[] scores = new float[languages.size()];
        int r = nativeIdentifyShort(handle, pcm, sampleRate, scores);
        return toResult(r, scores);
    }

    /** As {@link #identify(short[], int)}, for float samples in [-1, 1]. */
    public Result identify(float[] pcm, int sampleRate) {
        checkOpen();
        float[] scores = new float[languages.size()];
        int r = nativeIdentifyFloat(handle, pcm, sampleRate, scores);
        return toResult(r, scores);
    }

    /**
     * As {@link #identify(short[], int)}, but returns {@code null} instead of a
     * guess when nothing clears {@code minConfidence}. Use it where "unknown"
     * is a better answer than a wrong one.
     */
    @Nullable
    public Result identifyOrNull(short[] pcm, int sampleRate, float minConfidence) {
        Result r = identify(pcm, sampleRate);
        return r.confidence >= minConfidence ? r : null;
    }

    private Result toResult(int index, float[] scores) {
        if (index < 0) {
            throw new LidException(index, nativeStatusString(index));
        }
        return new Result(languages.get(index), index, scores[index], scores);
    }

    private void checkOpen() {
        if (handle == 0) throw new IllegalStateException("LanguageId is closed");
    }

    /**
     * Release the native session. An ORT session holds tens of MB of arenas,
     * so this is deliberately explicit rather than left to the collector.
     */
    @Override
    public void close() {
        if (handle != 0) {
            nativeDestroy(handle);
            handle = 0;
        }
    }

    // --- native ---------------------------------------------------------
    // Bound in JNI_OnLoad via RegisterNatives, so the .so carries no mangled
    // symbols naming this class.

    private static native long nativeCreate(String modelPath, int numThreads);
    private static native long nativeCreateFromBytes(byte[] model, int numThreads);
    private static native void nativeDestroy(long handle);
    private static native int nativeNumLanguages(long handle);
    private static native String nativeLanguageName(long handle, int index);
    private static native int nativeIdentifyShort(long handle, short[] pcm,
                                                  int sampleRate, float[] outScores);
    private static native int nativeIdentifyFloat(long handle, float[] pcm,
                                                  int sampleRate, float[] outScores);
    private static native String nativeStatusString(int status);
    private static native int nativeLastStatus();
}
