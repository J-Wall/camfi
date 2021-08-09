"""Camera-based Analysis and Monitoring of Flying Insects

See:
https://github.com/J-Wall/camfi
"""

import codecs
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()


def read(rel_path):
    here = pathlib.Path(__file__).parent.resolve()
    with codecs.open(here.joinpath(rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="camfi",
    version=get_version("camfi/_version.py"),
    description="Camera-based Analysis and Monitoring of Flying Insects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/J-Wall/camfi",
    author="Jesse Wallace",
    author_email="jesse.wallace@anu.edu.au",
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a list of additional keywords, separated
    # by commas, to be used to assist searching for the distribution in a
    # larger catalog.
    keywords="ecology, monitoring, insects, wingbeat frequency",  # Optional
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    # package_dir={'': 'src'},  # Optional
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    packages=find_packages(),
    # py_modules=["camfi"],
    #
    # packages=find_packages(where='src'),  # Required
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.9, <4",
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "bces",
        "exif",
        "numpy",
        "pandas",
        "pydantic",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "Shapely",
        "skyfield",
        "strictyaml",
        "torch",
        "torchvision",
        "tqdm",
    ],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    entry_points={  # Optional
        "console_scripts": [
            "camfi=camfi.cli:main",
        ],
    },
)
