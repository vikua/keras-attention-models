import os
import re
import sys
from setuptools import setup, find_packages


PY_VER = sys.version_info

if not PY_VER >= (3, 6):
    raise RuntimeError("Only Python 3.6 and later")


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


install_requires = ["Keras==2.3.0", "tensorflow==1.15.2"]
extras_require = {}


def read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(os.path.dirname(__file__), "mung", "__init__.py")
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        else:
            msg = "Cannot find version in mung/__init__.py"
            raise RuntimeError(msg)


classifiers = [
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Operating System :: POSIX",
    "Development Status :: 2 - Pre-Alpha",
]


setup(
    name="keras-attention-models",
    version=read_version(),
    description=("keras attention models"),
    long_description=read("README.md"),
    install_requires=install_requires,
    classifiers=classifiers,
    platforms=["POSIX"],
    author="Viktor Kovryzhkin",
    author_email="vik.kovrizhkin@gmail.com",
    url="https://github.com/vikua/keras-attention-models",
    download_url="",
    license="Apache 2",
    packages=find_packages(),
    extras_require=extras_require,
    keywords=[
        "machine learning",
        "deep learning",
        "seq2seq",
        "attention",
        "luong",
        "bahdanau",
    ],
    zip_safe=True,
    include_package_data=True,
)
