import numpy as np

from dlsj.kinesthetic.sounds import CUES, SAMPLE_RATE, Cues, synth


def test_start_and_stop_cues_differ():
    start, stop = synth(CUES["start"]), synth(CUES["stop"])
    assert start.dtype == np.int16 and stop.dtype == np.int16
    assert len(start) != len(stop) or not np.array_equal(start, stop)
    # stop cue is lower-pitched: count zero crossings per second
    def pitch(x):
        return np.count_nonzero(np.diff(np.sign(x.astype(float))) != 0) / (len(x) / SAMPLE_RATE) / 2
    assert pitch(stop) < pitch(start)


def test_synth_length_and_silence():
    x = synth([(440.0, 0.1), (0.0, 0.05)])
    assert len(x) == int(SAMPLE_RATE * 0.1) + int(SAMPLE_RATE * 0.05)
    assert np.all(x[-int(SAMPLE_RATE * 0.05):] == 0)


def test_disabled_cues_do_nothing(capsys):
    c = Cues(enabled=False)
    c.play("start")
    assert capsys.readouterr().out == ""
