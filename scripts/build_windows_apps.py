import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
STAGE_DIR = ROOT / ".pyinstaller-bundle-data"

COMMON_HIDDEN_IMPORTS = (
    "mivolo",
    "mivolo.model.create_timm_model",
    "mivolo.model.cross_bottleneck_attn",
    "mivolo.model.mivolo_model",
)

COMMON_METADATA = (
    "accelerate",
    "emotiefflib",
    "huggingface_hub",
    "onnxruntime",
    "openvino",
    "safetensors",
    "timm",
    "torch",
    "torchvision",
    "transformers",
)

COMMON_COLLECT_ALL = (
    "accelerate",
    "mediapipe",
    "mivolo",
    "safetensors",
    "timm",
    "torch",
    "torchvision",
    "transformers",
)

APP_TARGETS = (
    ("AI Yuz Tanima", "app_real.py"),
    ("AI Yuz Tanima Demo", "app_demo.py"),
)


def _add_data_arg(source: Path, destination: str) -> str:
    return f"{source}{os.pathsep}{destination}"


def _prepare_stage_dir() -> Path:
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    staged_models_dir = STAGE_DIR / "models"
    staged_models_dir.mkdir(parents=True, exist_ok=True)

    face_model = ROOT / "models" / "face_landmarker.task"
    if face_model.exists():
        shutil.copy2(face_model, staged_models_dir / "face_landmarker.task")

    mivolo_cache = ROOT / "models" / "mivolo_v2"
    if mivolo_cache.exists():
        shutil.copytree(
            mivolo_cache,
            STAGE_DIR / "mivolo_v2",
            symlinks=False,
            ignore=shutil.ignore_patterns(".locks"),
        )

    return STAGE_DIR


def _write_runtime_readme(app_dir: Path, app_name: str) -> None:
    lines = [
        f"{app_name} - Windows Paketi",
        "",
        "1. Bu klasoru oldugu gibi baska bilgisayara tasiyin.",
        "2. Gonderirken tum klasoru ZIP yapip yollayin.",
        "3. Karsi tarafta ZIP'i tamamen cikarin.",
        f"4. {app_name}.exe dosyasini cift tiklayip acin.",
        "",
        "Notlar:",
        "- Sadece .exe dosyasini tek basina tasimayin; yan dosyalarla birlikte kalmali.",
        "- SmartScreen uyari verirse More info > Run anyway ile acabilirsiniz.",
        "- Telefon kamerasi sanal webcam olarak baglandiysa Windows Camera uygulamasinda once test edin.",
        "- Eğer mivolo_v2 klasoru pakete dahil degilse ilk acilista internet gerekebilir.",
    ]
    (app_dir / "PAKET_NOTLARI.txt").write_text("\n".join(lines), encoding="utf-8")


def _zip_distribution(app_name: str) -> Path:
    source_dir = DIST_DIR / app_name
    archive_base = DIST_DIR / f"{app_name}-windows"
    archive_path = archive_base.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(archive_base), "zip", root_dir=DIST_DIR, base_dir=app_name)
    return archive_path


def _build_app(name: str, entry_script: str) -> Path:
    stage_dir = _prepare_stage_dir()
    data_args = [
        ("assets", ROOT / "assets"),
        ("models", stage_dir / "models" / "face_landmarker.task"),
        ("mivolo_v2", stage_dir / "mivolo_v2"),
    ]
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        name,
        "--paths",
        str(ROOT),
        "--paths",
        str(ROOT / "src"),
        "--collect-submodules",
        "mivolo",
    ]

    for destination, source in data_args:
        if source.exists():
            command.extend(["--add-data", _add_data_arg(source, destination)])

    for hidden_import in COMMON_HIDDEN_IMPORTS:
        command.extend(["--hidden-import", hidden_import])

    for package_name in COMMON_METADATA:
        command.extend(["--copy-metadata", package_name])

    for package_name in COMMON_COLLECT_ALL:
        command.extend(["--collect-all", package_name])

    command.append(str(ROOT / entry_script))
    subprocess.run(command, check=True, cwd=ROOT)

    app_dir = DIST_DIR / name
    _write_runtime_readme(app_dir, name)
    return _zip_distribution(name)


def main() -> None:
    DIST_DIR.mkdir(exist_ok=True)
    for name, entry_script in APP_TARGETS:
        print(f"Building {name}.exe ...")
        archive_path = _build_app(name=name, entry_script=entry_script)
        print(archive_path)

    print("\nBuild complete:")
    for name, _entry_script in APP_TARGETS:
        print(DIST_DIR / name)
        print(DIST_DIR / f"{name}-windows.zip")


if __name__ == "__main__":
    main()
