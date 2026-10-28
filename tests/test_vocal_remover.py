import numpy as np
import pytest

from vocal_remover import Options, process

SR = 44100


def tone(freq, seconds=1.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_center_is_removed_and_sides_survive():
    vocal = tone(440)            # centered
    guitar = tone(1000, amp=0.3)  # hard left
    mix = np.stack([vocal + guitar, vocal], axis=1)
    out = process(mix, SR, Options(bass_cutoff=0, normalize=False))
    # The centered 440 Hz tone should be gone, the left-only tone should remain.
    spectrum = np.abs(np.fft.rfft(out[:, 0]))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    v = spectrum[np.argmin(np.abs(freqs - 440))]
    g = spectrum[np.argmin(np.abs(freqs - 1000))]
    assert v < g * 0.01


def test_bass_is_preserved():
    bass = tone(60)
    vocal = tone(440)
    mix = np.stack([bass + vocal, bass + vocal], axis=1)
    out = process(mix, SR, Options(bass_cutoff=150, normalize=False))
    spectrum = np.abs(np.fft.rfft(out[:, 0]))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    b = spectrum[np.argmin(np.abs(freqs - 60))]
    v = spectrum[np.argmin(np.abs(freqs - 440))]
    assert b > 10 * v


def test_isolate_mode_returns_center():
    vocal = tone(440)
    side = tone(1500, amp=0.3)
    mix = np.stack([vocal + side, vocal - side], axis=1)
    out = process(mix, SR, Options(mode="isolate", bass_cutoff=0, normalize=False))
    assert np.allclose(out[:, 0], vocal, atol=1e-5)


def test_output_never_clips():
    loud = np.stack([tone(200, amp=1.0), -tone(200, amp=1.0)], axis=1)
    out = process(loud, SR, Options(bass_cutoff=0))
    assert np.max(np.abs(out)) <= 0.99 + 1e-6


def test_mono_input_rejected():
    with pytest.raises(ValueError):
        process(tone(440)[:, None], SR)
