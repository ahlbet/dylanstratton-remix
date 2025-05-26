from setuptools import setup, find_packages

setup(
    name="model",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # add your requirements here
    ],
    entry_points={
        "console_scripts": [
            "train=scripts.train:main",
            "generate=scripts.generate_audio:main",
            "preprocess=scripts.preprocess_audio:main",
        ],
    },
)
