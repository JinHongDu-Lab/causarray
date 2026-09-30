"""The import-time warning for Python translated by Rosetta 2."""
import platform
import sys
import warnings

import pytest

from causarray import _platform


def test_no_warning_off_macos_or_on_native_arm(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert not _platform.running_under_rosetta()
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(platform, 'machine', lambda: 'arm64')
    assert not _platform.running_under_rosetta()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _platform.warn_if_emulated()


def test_warns_when_translated(monkeypatch):
    monkeypatch.setattr(_platform, 'running_under_rosetta', lambda: True)
    with pytest.warns(RuntimeWarning, match='Rosetta 2'):
        _platform.warn_if_emulated()


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS only')
def test_detection_matches_this_process():
    """On a Mac the check agrees with the interpreter's own architecture."""
    if platform.machine() == 'arm64':
        assert not _platform.running_under_rosetta()
    else:
        assert isinstance(_platform.running_under_rosetta(), bool)
