# Copyright 2026 Syed Basim Ali
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_py import build_py
from Cython.Build import cythonize
import Cython.Tempita as tempita
import numpy as np

src_dir = Path("src/numcircbuf")
core_file_name = "core"
sys.path.insert(0, str(src_dir.resolve()))

debug_build = os.getenv("DEBUG_BUILD", "0") == "1"
test_build = os.getenv("TEST_BUILD", "0") == "1"
coverage_build = os.getenv("COVERAGE_BUILD", "0") == "1"
save_asm = os.getenv("SAVE_ASM", "0") == "1"
annotate_cython = os.getenv("CYTHON_ANNOTATE", "0") == "1"
only_test_api = os.getenv("ONLY_TEST_API", "0") == "1"

is_test = only_test_api or test_build or debug_build or coverage_build


class FilteredBuildPy(build_py):
    """Intercept the Python module discovery and remove targeted modules."""

    _filtered_build_py_excluded_modules = ("_build_helpers",)

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = [
            (pkg, mod, filepath)
            for (pkg, mod, filepath) in modules
            if mod not in self._filtered_build_py_excluded_modules
        ]
        return filtered_modules


def insert_autogen_notice(content: str, notice: str) -> str:
    lines = content.splitlines()

    for i, line in enumerate(lines):
        if line.strip() == "":
            return "\n".join(lines[:i] + ["", notice, ""] + lines[i + 1 :])

    return notice + "\n\n" + content


def process_tempita(file_path: Path) -> Path:
    """Reads a .in file, processes the template, and writes the output."""
    assert file_path.suffix == ".in", f"{file_path} must end with .in"
    out_file = file_path.with_suffix("")  # without .in

    with open(file_path, "r", encoding="utf-8") as f:
        tmpl = f.read()

    content = tempita.sub(tmpl)
    notice = "\n".join(
        [
            "# " + "-" * 77,
            "# THIS FILE IS AUTO-GENERATED AT BUILD TIME.",
            "# " + "-" * 77,
        ]
    )
    content = insert_autogen_notice(content, notice)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)

    return out_file


extra_link_args: list[str] = []
if sys.platform == "win32":  # MSVC optimization
    if debug_build or coverage_build:  # No optimization, debug_build symbols
        extra_compile_args = ["/Od", "/Z7"]
    else:
        extra_compile_args = ["/O2"]

    if save_asm:
        extra_compile_args.append("/FAs")

else:  # GCC/Clang optimization
    if debug_build or coverage_build:
        extra_compile_args = ["-O0", "-g"]
        extra_link_args = ["-g"]
    else:
        extra_compile_args = ["-O3"]

    if save_asm:
        extra_compile_args.extend(["-save-temps", "-fverbose-asm"])

compiler_directives = {
    "emit_code_comments": coverage_build,
    "linetrace": coverage_build,
    "binding": coverage_build,
}
define_macros: list[tuple[str, str]] = []
if coverage_build:
    define_macros.extend([("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")])

extensions = []
exclude_pkg_data = ["core.cpp", "_test_cython_api.cpp"]

if not only_test_api:
    process_tempita(src_dir / f"{core_file_name}.pxd.in")
    pyx_file = process_tempita(src_dir / f"{core_file_name}.pyx.in")
    extensions.append(
        Extension(
            "numcircbuf.core",
            sources=[str(pyx_file)],
            depends=[str(src_dir / "kernels.hpp")],
            include_dirs=[np.get_include()],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )

if is_test:
    test_cython_api_pyx_file = process_tempita(src_dir / "_test_cython_api.pyx.in")
    ext_name = (
        "numcircbuf_test_cython_api" if only_test_api else "numcircbuf._test_cython_api"
    )
    extensions.append(
        Extension(
            ext_name,
            sources=[str(test_cython_api_pyx_file)],
            include_dirs=[np.get_include(), str(src_dir.parent)],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )
else:
    exclude_pkg_data.append("_test_cython_api.pyx")

setup(
    name="NumCircBuf",
    version="1.1.1",
    description="High-performance numerical circular buffers for Python, featuring O(1) statistical accumulators.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Syed Basim Ali",
    author_email="basim.ali.contact@gmail.com",
    url="https://github.com/basimali-ai/NumCircBuf",
    project_urls={
        "Source Code": "https://github.com/basimali-ai/NumCircBuf",
    },
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.0.0,<3.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.2",
            "pytest-mock>=3.10",
            "coverage>=7.13",
            "cython>=3.0,<4.0",  # for cython coverage
        ],
        "dev": [
            "cython>=3.0,<4.0",
            "build>=0.10",
            "pytest>=7.2",
            "pytest-mock>=3.10",
            "coverage>=7.13",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.3",
        ],
    },
    cmdclass={
        "build_py": FilteredBuildPy,
    },
    packages=find_packages(),
    package_data={
        "numcircbuf": ["*.pyi", "*.pxd", "*.pyx", ".hpp"],
    },
    exclude_package_data={
        "numcircbuf": exclude_pkg_data,
    },
    include_package_data=True,
    ext_modules=cythonize(
        extensions,
        include_path=[np.get_include()],
        compiler_directives=compiler_directives,
        annotate=annotate_cython,
    ),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "ring-buffer",
        "circular-buffer",
        "audio-processing",
        "signal-processing",
        "real-time",
        "performance",
        "numpy",
        "cython",
        "numerical",
        "high-performance",
        "numerical-circular-buffers",
    ],
    license="Apache-2.0",
    zip_safe=False,
)
