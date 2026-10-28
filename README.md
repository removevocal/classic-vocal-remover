# Classic Vocal Remover

**Remove (or isolate) center-panned vocals from a stereo song with the classic phase-cancellation trick, in pure Python.**

No GPU, no model downloads, no internet. One small file you can read in five minutes. It's useful for quick karaoke tracks, for teaching how stereo mixing works, and as a baseline to compare AI stem separators against.

![python](https://img.shields.io/badge/python-3.9%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![tests](https://img.shields.io/badge/tests-pytest-orange)

---

## How it works

Most pop, rock and hip-hop mixes place the lead vocal dead center, so it is (almost) identical in the left and right channels. In mid/side terms:

```
mid  = (L + R) / 2   -> everything in the center: vocal, kick, bass, snare
side = (L - R) / 2   -> everything panned away from the center: guitars, keys, reverb
```

Throwing away the *mid* signal removes the vocal. The problem is that it also throws away the kick drum and bass. This tool fixes that:

| Step | What happens |
|---|---|
| 1. Split | Convert L/R into mid and side |
| 2. Select | Optionally keep only the vocal band of the mid (`--band 200-6000`) |
| 3. Cancel | Subtract the selected center from the mix (`--strength`) |
| 4. Restore low end | Add everything below `--bass-cutoff` back, so kick and bass survive |
| 5. Protect | Peak-normalise so the result never clips |

`--mode isolate` does the opposite: it keeps the center and drops the sides, which gives you a rough acapella estimate.

## Install

```bash
git clone https://github.com/removevocal/classic-vocal-remover.git
cd classic-vocal-remover
pip install -r requirements.txt
```

Needs Python 3.9+ and works on Linux, macOS and Windows. Input can be WAV, FLAC, OGG or AIFF (anything `libsndfile` reads). Convert MP3s first with `ffmpeg -i song.mp3 song.wav`.

## Usage

```bash
# Karaoke / instrumental version -> output/song_instrumental.wav
python vocal_remover.py song.wav

# Several files at once, into a custom folder, as FLAC
python vocal_remover.py *.wav -o karaoke --format flac

# Gentler: only cancel 200 Hz-6 kHz and keep the low end below 150 Hz
python vocal_remover.py song.wav --band 200-6000 --bass-cutoff 150

# Rough acapella (center isolation) -> output/song_center.wav
python vocal_remover.py song.wav --mode isolate
```

### Options

| Flag | Default | Description |
|---|---|---|
| `-m, --mode` | `remove` | `remove` = instrumental/karaoke, `isolate` = center (vocal) estimate |
| `--bass-cutoff` | `120` | Keep everything below this frequency (Hz). `0` disables |
| `--band LOW-HIGH` | off | Only cancel the center inside this band, e.g. `200-6000` |
| `--strength` | `1.0` | `0` to `1`, how much of the center to cancel |
| `--format` | `wav` | `wav` (24-bit) or `flac` |
| `-o, --out-dir` | `output` | Output folder |
| `--no-normalize` | off | Skip peak protection |

### Use it as a library

```python
import soundfile as sf
from vocal_remover import Options, process

audio, sr = sf.read("song.wav", always_2d=True)
instrumental = process(audio, sr, Options(bass_cutoff=120, band=(200, 6000)))
sf.write("song_instrumental.wav", instrumental, sr)
```

## Limitations (please read)

Phase cancellation is a 1970s technique, and it only works when the mix cooperates:

- **Off-center or double-tracked vocals** survive, because they are not identical in both channels.
- **Stereo reverb and delay** on the vocal stay behind as a "ghost" vocal.
- **Center-panned instruments** such as snare, solo instruments and mono synths get removed too.
- **Mono files** cannot be processed, because there is no stereo image to cancel.
- **Isolate mode** is only an estimate. It contains everything panned center, not just the voice.

### When you need clean stems

If you need a properly clean acapella or instrumental, or separate **drums, bass and other** stems, use a modern AI source-separation model instead. Our hosted [AI vocal remover](https://remove-vocal.com/) runs in the browser and handles both. It splits songs and videos (MP3, WAV, FLAC, MP4, MOV and more) into 2, 4 or 6 tracks, lets you preview every stem, and exports WAV or MP3. Neural separation handles off-center vocals, reverb and dense mixes that this script cannot.

This repo is still handy as a free, offline, explainable baseline, and it's a nice way to *hear* what "center" means in a mix.

## Run the tests

```bash
pip install pytest
pytest -q
```

The tests synthesise signals (a centered "vocal", a panned "guitar" and a centered "bass") and check that the vocal is cancelled while the guitar and bass survive.

## Related tools

- [`stem-mixer`](https://github.com/removevocal/stem-mixer): remix separated stems into karaoke and practice tracks
- [`video-audio-swap`](https://github.com/removevocal/video-audio-swap): extract a video's soundtrack and put the instrumental back in
- [Remove Vocal](https://remove-vocal.com/): AI vocal remover and stem splitter in the browser

## Contributing

Issues and pull requests are welcome. Ideas: STFT-domain center extraction, a GUI, a VST wrapper.

Questions or feedback can also go to **[support@remove-vocal.com](mailto:support@remove-vocal.com)**.

## License

MIT. See [LICENSE](LICENSE). Only process audio you have the rights to use.

---

Made by the team behind [Remove Vocal](https://remove-vocal.com/), an online AI vocal remover and stem splitter.
