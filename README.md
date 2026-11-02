# lid9 1.0.0

Spoken language identification for nine languages:

**english · hindi · telugu · tamil · kannada · malayalam · marathi · bengali · gujarati**

Audio in, language name out. Feature extraction is compiled into the model, so
both the Python and Android APIs take raw audio — there is no front end to
configure and nothing that can drift out of sync.

## Contents

```
python/          pip-installable package, model bundled inside it
android/
  jniLibs/<abi>/ liblid9.so, liblid9_jni.so, libonnxruntime.so
  java/          ai/pramiteeh/lid9/LanguageId.java
native/include/  lid9.h — C API
NOTICE           attribution you are required to ship
MANIFEST.txt     ABIs, build date
```

ABIs included: arm64-v8a armeabi-v7a x86_64 

---

## Python

```bash
pip install ./python
```

Pulls in `onnxruntime` and `numpy`. Add `soundfile` to read audio files;
it is not needed if you pass numpy arrays.

Command line:

```bash
lid9 --audio sample.wav
# telugu

lid9 --audio sample.wav --top-k 3
lid9 --audio a.wav b.wav --json
```

Library:

```python
from lid9 import LanguageID

lid = LanguageID()              # construct ONCE — see note below
r = lid.identify("sample.wav")

r.language      # 'telugu'
r.confidence    # 0.9997
r.top_k         # [('telugu', 0.9997), ('malayalam', 0.0002), ...]
```

The model ships inside the package, so the default constructor works from any
working directory.

`identify()` accepts a path **or a numpy array** — arrays skip `soundfile`
entirely, which matters when audio arrives from a socket or an existing decode
pipeline:

```python
lid.identify(samples)                     # 1-D float32 @ 16 kHz
lid.identify(samples, sample_rate=8000)   # resampled internally
lid.identify(stereo_array)                # downmixed automatically
```

Return `None` instead of a guess when nothing is confident enough:

```python
r = lid.identify(audio, min_confidence=0.7)
if r.language is None:
    ...                                   # r.confidence shows how close it got
```

**Construct `LanguageID` once.** Session creation costs roughly 100x a single
inference, so building one per request will dominate your latency. Make it a
module-level singleton. `identify()` is thread-safe on one instance;
16 concurrent calls through a single instance are verified to agree.

Clips under 0.3 s raise `AudioTooShort`; a missing model raises
`ModelNotFound`. Both are typed so a service can catch them individually.

---

## Android

Copy the two trees into your app module:

```bash
cp -r android/jniLibs  app/src/main/jniLibs
cp -r android/java/ai  app/src/main/java/
```

```java
// Once per process — construction is expensive.
public class LidHolder {
    private static LanguageId instance;
    public static synchronized LanguageId get() {
        if (instance == null) instance = LanguageId.fromEmbeddedModel(2);
        return instance;
    }
}

short[] pcm = new short[16000 * 5];            // 5 s @ 16 kHz
audioRecord.read(pcm, 0, pcm.length);

LanguageId.Result r = LidHolder.get().identify(pcm, 16000);
Log.i("lid", r.language + " " + r.confidence);
```

`identify(short[], int)` takes `AudioRecord` output directly — no conversion
needed. `identifyOrNull(pcm, rate, 0.7f)` returns `null` below the threshold.

`identify()` is thread-safe on one instance. `close()` is **not** — do not
close while calls are in flight.

**Ship one ABI per device.** Use
[ABI splits](https://developer.android.com/build/configure-apk-splits) or an App
Bundle, and drop `armeabi-v7a`/`x86_64` unless you need them. Including all
ABIs in a single APK multiplies the payload.

---

## C

```c
#include "lid9.h"

Lid9 *lid = lid9_create(NULL, 2);          // NULL -> embedded model
Lid9Result r;
if (lid9_identify(lid, samples, n, 16000, &r) == LID9_OK) {
    printf("%s %.3f\n", lid9_language_name(lid, r.language_index), r.confidence);
}
lid9_destroy(lid);
```

`lid9_identify_i16()` takes 16-bit PCM. Other sample rates are resampled
internally; audio longer than 10 s is truncated. Full API in
`native/include/lid9.h`. Link against `liblid9.so` with
`libonnxruntime.so` alongside it.


---

## Accuracy

The pipeline is verified end to end: Python, C and Android agree, and ONNX
matches the reference implementation.

**Accuracy across the nine languages has not been evaluated on labelled data.**
Two things to test before depending on it:

- The four Dravidian languages (Tamil, Telugu, Kannada, Malayalam) separate
  easily. The hard cluster is Indo-Aryan — Hindi, Marathi, **Gujarati**,
  Bengali. Gujarati and Marathi are both Western/Central Indo-Aryan and are the
  pair most likely to dominate a confusion matrix.
- The upstream model is trained on scraped YouTube audio; its model card reports
  worse performance on female speakers, foreign accents and children's speech.
