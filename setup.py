import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildRustBinaries(build_py):
    def run(self):
        self._build_rust()
        super().run()

    def _build_rust(self):
        version = self._read_version()
        version_dashes = version.replace(".", "-")
        binary_name = f"onnx_generic_circuit_{version_dashes}"

        binaries_dir = Path("python/core/binaries")
        binaries_dir.mkdir(parents=True, exist_ok=True)

        if not shutil.which("cargo"):
            print(
                "WARNING: cargo not found — skipping Rust binary builds",
                file=sys.stderr,
            )
            return

        self._cargo_build(
            binary_name,
            binaries_dir,
            source_dir=Path("target/release"),
        )
        expander_manifest = Path("Expander/Cargo.toml")
        if expander_manifest.exists():
            self._cargo_build(
                "expander-exec",
                binaries_dir,
                manifest_path=str(expander_manifest),
                source_dir=Path("Expander/target/release"),
            )

    def _read_version(self):
        text = Path("pyproject.toml").read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if not match:
            raise RuntimeError("Could not read version from pyproject.toml")
        return match.group(1)

    def _cargo_build(self, binary_name, target_dir, source_dir, manifest_path=None):
        cmd = ["cargo", "build", "--release", "--bin", binary_name]
        if manifest_path:
            cmd.extend(["--manifest-path", manifest_path])

        print(f"Building {binary_name}...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(
                f"WARNING: Failed to build {binary_name}",
                file=sys.stderr,
            )
            return

        src = source_dir / binary_name
        if src.exists():
            dst = target_dir / binary_name
            shutil.copy2(str(src), str(dst))
            dst.chmod(0o755)
            print(f"Copied {binary_name} to {dst}")
        else:
            print(
                f"WARNING: Built binary not found at {src}",
                file=sys.stderr,
            )


setup(cmdclass={"build_py": BuildRustBinaries})
