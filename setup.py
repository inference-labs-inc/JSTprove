import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

LINUX_BASELINE_PREFIXES = (
    "libc.so",
    "libm.so",
    "libpthread.so",
    "libdl.so",
    "librt.so",
    "libstdc++.so",
    "libgcc_s.so",
    "linux-vdso.so",
    "ld-linux",
    "libnsl.so",
    "libutil.so",
    "libresolv.so",
    "libcrypt.so",
)


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

        if not (binaries_dir / binary_name).exists():
            self._cargo_build(
                binary_name,
                binaries_dir,
                source_dir=Path("target/release"),
            )

        if not (binaries_dir / "expander-exec").exists():
            expander_manifest = Path("Expander/Cargo.toml")
            if expander_manifest.exists():
                self._cargo_build(
                    "expander-exec",
                    binaries_dir,
                    manifest_path=str(expander_manifest),
                    source_dir=Path("Expander/target/release"),
                )

        self._bundle_mpi_libs(binaries_dir)

    def _read_version(self):
        text = Path("pyproject.toml").read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if not match:
            raise RuntimeError("Could not read version from pyproject.toml")
        return match.group(1)

    def _cargo_build(self, binary_name, target_dir, source_dir, manifest_path=None):
        if not shutil.which("cargo"):
            print(
                f"WARNING: cargo not found, skipping build of {binary_name}",
                file=sys.stderr,
            )
            return

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

    def _bundle_mpi_libs(self, binaries_dir):
        lib_dir = Path("python/core/lib")
        lib_dir.mkdir(parents=True, exist_ok=True)
        (lib_dir / "__init__.py").touch()

        if any(lib_dir.glob("libmpi*")):
            print("MPI libraries already bundled, skipping")
            return

        if sys.platform == "darwin":
            self._bundle_macos(binaries_dir, lib_dir)
        elif sys.platform == "linux":
            self._bundle_linux(binaries_dir, lib_dir)

    def _bundle_macos(self, binaries_dir, lib_dir):
        try:
            mpi_prefix = Path(
                subprocess.run(
                    ["brew", "--prefix", "open-mpi"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip(),
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                "WARNING: brew/open-mpi not found, skipping MPI bundling",
                file=sys.stderr,
            )
            return

        mpi_dylibs = sorted((mpi_prefix / "lib").glob("libmpi.*.dylib"))
        if not mpi_dylibs:
            print(
                f"WARNING: no libmpi.*.dylib found in {mpi_prefix / 'lib'}",
                file=sys.stderr,
            )
            return

        shutil.copy2(str(mpi_dylibs[0]), str(lib_dir / mpi_dylibs[0].name))

        changed = True
        while changed:
            changed = False
            for lib in list(lib_dir.glob("*.dylib")):
                result = subprocess.run(
                    ["otool", "-L", str(lib)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                for line in result.stdout.splitlines()[1:]:
                    dep_path = line.strip().split()[0]
                    if dep_path.startswith(("/usr/lib", "/System", "@")):
                        continue
                    dep_name = Path(dep_path).name
                    if (lib_dir / dep_name).exists():
                        continue
                    src = Path(dep_path)
                    if not src.exists():
                        continue
                    shutil.copy2(str(src), str(lib_dir / dep_name))
                    changed = True

        all_names = {f.name for f in lib_dir.glob("*.dylib")}

        for lib in lib_dir.glob("*.dylib"):
            subprocess.run(
                ["install_name_tool", "-id", f"@loader_path/{lib.name}", str(lib)],
                check=False,
            )
            result = subprocess.run(
                ["otool", "-L", str(lib)],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines()[1:]:
                old_path = line.strip().split()[0]
                old_name = Path(old_path).name
                if old_name in all_names and not old_path.startswith("@"):
                    subprocess.run(
                        [
                            "install_name_tool",
                            "-change",
                            old_path,
                            f"@loader_path/{old_name}",
                            str(lib),
                        ],
                        check=False,
                    )
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(lib)],
                check=False,
            )

        for binary in binaries_dir.iterdir():
            if binary.name == "__init__.py" or not binary.is_file():
                continue
            result = subprocess.run(
                ["otool", "-L", str(binary)],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines()[1:]:
                old_path = line.strip().split()[0]
                old_name = Path(old_path).name
                if old_name in all_names and not old_path.startswith("@"):
                    subprocess.run(
                        [
                            "install_name_tool",
                            "-change",
                            old_path,
                            f"@rpath/{old_name}",
                            str(binary),
                        ],
                        check=False,
                    )
            subprocess.run(
                ["install_name_tool", "-add_rpath", "@loader_path/../lib", str(binary)],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(binary)],
                check=False,
            )

    def _bundle_linux(self, binaries_dir, lib_dir):
        if not shutil.which("patchelf"):
            print("WARNING: patchelf not found, skipping MPI bundling", file=sys.stderr)
            return

        mpi_lib_dir = self._find_mpi_lib_dir()
        if mpi_lib_dir is None:
            print(
                "WARNING: Could not find MPI libraries, skipping bundling",
                file=sys.stderr,
            )
            return

        for pattern in ("libmpi.so*", "libopen-pal.so*", "libopen-rte.so*"):
            for lib in mpi_lib_dir.glob(pattern):
                real = lib.resolve()
                soname = (
                    subprocess.run(
                        ["patchelf", "--print-soname", str(real)],
                        capture_output=True,
                        text=True,
                        check=False,
                    ).stdout.strip()
                    or lib.name
                )
                if not (lib_dir / soname).exists():
                    shutil.copy2(str(real), str(lib_dir / soname))

        changed = True
        while changed:
            changed = False
            for lib in list(lib_dir.glob("*.so*")):
                result = subprocess.run(
                    ["patchelf", "--print-needed", str(lib)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                for needed in result.stdout.strip().splitlines():
                    needed = needed.strip()
                    if not needed or needed.startswith(LINUX_BASELINE_PREFIXES):
                        continue
                    if (lib_dir / needed).exists():
                        continue
                    found = self._find_lib(needed, [mpi_lib_dir])
                    if found:
                        shutil.copy2(str(found.resolve()), str(lib_dir / needed))
                        changed = True

        for lib in lib_dir.glob("*.so*"):
            subprocess.run(
                ["patchelf", "--set-rpath", "$ORIGIN", str(lib)],
                check=False,
            )

        for binary in binaries_dir.iterdir():
            if binary.name == "__init__.py" or not binary.is_file():
                continue
            subprocess.run(
                ["patchelf", "--set-rpath", "$ORIGIN/../lib", str(binary)],
                check=False,
            )

    def _find_mpi_lib_dir(self):
        for p in (
            "/usr/lib64/openmpi/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu/openmpi/lib",
        ):
            path = Path(p)
            if any(path.glob("libmpi.so*")):
                return path

        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "libmpi.so" in line:
                return Path(line.strip().split()[-1]).parent
        return None

    def _find_lib(self, name, extra_dirs):
        for d in extra_dirs:
            candidate = d / name
            if candidate.exists():
                return candidate

        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if name in line:
                path = Path(line.strip().split()[-1])
                if path.exists():
                    return path
        return None


setup(cmdclass={"build_py": BuildRustBinaries})
