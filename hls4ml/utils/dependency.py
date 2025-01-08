import sys
from functools import wraps
from importlib.metadata import metadata
from inspect import ismethod

extra_requires: dict[str, list[str]] = {}
subpackage = None
for k, v in metadata('hls4ml')._headers:  # type: ignore
    if k != 'Requires-Dist':
        continue
    if '; extra == ' not in v:
        continue

    req, pkg = v.split('; extra == ')
    pkg = pkg.strip('"')

    extra_requires.setdefault(pkg, []).append(req)


def requires(pkg: str):
    """Mark a function or method as requiring a package to be installed.
    'name': requires hls4ml[name] to be installed.
    '_name': requires name to be installed.

    Parameters
    ----------
    pkg : str
        The package to require.
    """

    def deco(f):
        if ismethod(f):
            qualifier = f"Method {f.__self__.__class__.__name__}.{f.__name__}"
        else:
            qualifier = f"Function {f.__name__}"

        if not pkg.startswith("_"):
            reqs = ", ".join(extra_requires[pkg])
            msg = f"{qualifier} requires {reqs}, but package {{ename}} is missing"
            "Please consider install it with `pip install hls4ml[{pkg}]` for full functionality with {pkg}."
        else:
            msg = f"{qualifier} requires {pkg[1:]}, but package {{ename}} is missing."
            "Consider install it with `pip install {pkg}`."

        @wraps(f)
        def inner(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except ImportError as e:
                print(msg.format(ename=e.name), file=sys.stderr)
                raise e

        return inner

    return deco
