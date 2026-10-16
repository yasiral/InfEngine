"""M13 item 3: product-image compatibility gate tests.

The gate must fail closed: supported version + hook => success; unsupported
version => non-zero; missing hook => non-zero. The Dockerfile must invoke it
without ``|| true``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
COMPAT = REPO / "contrib" / "vllm" / "compat_check.py"
DOCKERFILE = REPO / "contrib" / "vllm" / "Dockerfile.infengine"


def _load_compat():
    spec = importlib.util.spec_from_file_location("infengine_compat_check", COMPAT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_vllm(monkeypatch, version, with_hook=True):
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    vllm = types.ModuleType("vllm")
    vllm.__version__ = version
    v1 = types.ModuleType("vllm.v1")
    engine = types.ModuleType("vllm.v1.engine")
    core = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        pass

    if with_hook:
        def set_runtime_scheduler_config(self, overrides):
            return {}
        EngineCore.set_runtime_scheduler_config = set_runtime_scheduler_config
    core.EngineCore = EngineCore
    vllm.v1 = v1
    v1.engine = engine
    engine.core = core
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core)


def test_supported_version_with_hook_passes(monkeypatch):
    _install_fake_vllm(monkeypatch, "0.28.0", with_hook=True)
    assert _load_compat().check_installed() == 0


def test_unsupported_version_fails(monkeypatch):
    _install_fake_vllm(monkeypatch, "0.99.0", with_hook=True)
    assert _load_compat().check_installed() != 0


def test_missing_hook_fails(monkeypatch):
    _install_fake_vllm(monkeypatch, "0.28.0", with_hook=False)
    assert _load_compat().check_installed() != 0


def test_vllm_not_installed_fails(monkeypatch):
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ImportError("no vllm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _load_compat().check_installed() != 0


def test_dockerfile_gate_is_not_weakened():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "compat_check.py" in text
    assert "compat_check.py || true" not in text
