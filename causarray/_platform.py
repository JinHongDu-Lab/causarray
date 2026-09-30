"""Warn when causarray runs on an emulated CPU architecture.

On Apple Silicon, a Python built for Intel (x86_64) runs under Rosetta 2. It
works, but NumPy, Numba and the GLM fits run translated and are several times
slower: on a 5,000-cell subsample of a Perturb-seq screen, ``LFC`` took 159 s
under Rosetta against 32-57 s natively. pip installs causarray into whichever
Python it is given, so the check has to happen at run time.
"""
import platform
import sys
import warnings

__all__ = ['running_under_rosetta', 'warn_if_emulated']


def running_under_rosetta():
    """Return True if this process is an x86_64 build translated by Rosetta 2."""
    if sys.platform != 'darwin' or platform.machine() != 'x86_64':
        return False
    try:
        import ctypes
        libc = ctypes.CDLL('/usr/lib/libSystem.dylib')
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(b'sysctl.proc_translated', ctypes.byref(value),
                             ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return False  # the key is absent on Intel Macs
        return value.value == 1
    except Exception:
        return False


def warn_if_emulated():
    """Issue a ``RuntimeWarning`` once if causarray runs under Rosetta 2."""
    if running_under_rosetta():
        warnings.warn(
            'causarray is running in an x86_64 (Intel) Python translated by Rosetta 2 '
            'on an Apple Silicon Mac; its numerical code will run several times slower. '
            'Install a native arm64 Python, e.g. '
            '`CONDA_SUBDIR=osx-arm64 conda create -n causarray python=3.12`, '
            'and check with `python -c "import platform; print(platform.machine())"` '
            '(expect arm64).',
            RuntimeWarning, stacklevel=3)
