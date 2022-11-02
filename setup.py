import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optsim",
    version="0.0.1",
    author="F. Soldevila",
    author_email="soltfern@gmail.com",
    description="Tools for optical simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cbasedlf/optsim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-image",
        "pillow",
		"scipy",
    ],
)