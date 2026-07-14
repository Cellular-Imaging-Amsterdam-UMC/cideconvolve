from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gui_pins_exact_cu132_torch():
    text = (ROOT / "requirements_gui.txt").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cu132" in text
    assert "torch==2.13.0+cu132" in text


def test_docker_parameterizes_exact_torch_wheel():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements_docker.txt").read_text(encoding="utf-8")
    assert "ARG PYTORCH_VERSION=2.13.0" in dockerfile
    assert "ARG PYTORCH_CUDA=cu132" in dockerfile
    assert 'torch==${PYTORCH_VERSION}+${PYTORCH_CUDA}' in dockerfile
    assert "torch" not in [line.strip().split("=")[0] for line in requirements.splitlines() if line and not line.startswith("#")]


def test_headless_build_creates_cu130_fallback_tags():
    text = (ROOT / "builddocker.cmd").read_text(encoding="utf-8")
    assert "PYTORCH_CUDA=cu132" in text
    assert "PYTORCH_CUDA=cu130" in text
    assert "%VERSION%-cu130" in text
    assert "latest-cu130" in text


def test_container_includes_cuda_smoke_test():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY docker/cuda_smoke.py /app/cuda_smoke.py" in dockerfile
    assert (ROOT / "docker" / "cuda_smoke.py").is_file()
