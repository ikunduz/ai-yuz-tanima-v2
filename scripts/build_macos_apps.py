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
    "mediapipe",
)

APP_TARGETS = (
    ("AI Yuz Tanima", "app_real.py", "com.ikunduz.aiyuztanima"),
    ("AI Yuz Tanima Demo", "app_demo.py", "com.ikunduz.aiyuztanima.demo"),
)
CAMERA_USAGE_DESCRIPTION = (
    "Bu uygulama yuz takibi ve cocuk challenge modlari icin kamerayi kullanir."
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


def _configure_bundle_metadata(app_path: Path) -> None:
    info_plist = app_path / "Contents" / "Info.plist"
    subprocess.run(
        [
            "plutil",
            "-replace",
            "NSCameraUsageDescription",
            "-string",
            CAMERA_USAGE_DESCRIPTION,
            str(info_plist),
        ],
        check=True,
    )
    subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(app_path)],
        check=True,
    )


def _build_app(name: str, entry_script: str, bundle_id: str) -> None:
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
        "--osx-bundle-identifier",
        bundle_id,
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
    _configure_bundle_metadata(DIST_DIR / f"{name}.app")


def main() -> None:
    DIST_DIR.mkdir(exist_ok=True)
    for name, entry_script, bundle_id in APP_TARGETS:
        print(f"Building {name}.app ...")
        _build_app(name=name, entry_script=entry_script, bundle_id=bundle_id)

    print("\nBuild complete:")
    for name, _entry_script, _bundle_id in APP_TARGETS:
        print(DIST_DIR / f"{name}.app")


if __name__ == "__main__":
    main()
