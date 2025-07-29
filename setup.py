"""Setup file for compiling and building Cython/C-extensions."""
import sys
import argparse
from setuptools import Extension, setup
import numpy

parser = argparse.ArgumentParser(
    prog="python setup.py",
    formatter_class=argparse.RawTextHelpFormatter,
    description="Build skais-mapper extensions.\n"
    "Use `python setup.py build_ext --inplace`\n"
    "for building Cython extensions from their corresponding C files or\n"
    "`python setup.py build_ext --inplace --use_cython`\nfor building extensions"
    " from the cython files directly.",
)
parser.add_argument(
    "--use_cython",
    "--cython",
    action="store_true",
    help="Build extensions from their cython files instead of the C files.",
)
parser.add_argument(
    "-a", "--report", action="store_true", help="Generate annotated HTML compilation report."
)
parser.add_argument("build_c", nargs="?", default="", help="Compile cython files and exit.")

args, unk = parser.parse_known_args()
USE_CYTHON = args.use_cython
BUILD_C = args.build_c == "build_c"

ext = ".pyx" if USE_CYTHON or BUILD_C else ".c"
extensions = [
    Extension(
        name="skais_mapper.raytrace",
        sources=["skais_mapper/raytrace" + ext],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

if USE_CYTHON or BUILD_C:
    from Cython.Build import cythonize

    extensions = cythonize(extensions, language_level=3, annotate=True)
if BUILD_C:
    exit(0)

for c in ["--cython", "--use_cython", "-a", "--report", "build_c"]:
    if c in sys.argv:
        sys.argv.remove(c)

setup(ext_modules=extensions)
