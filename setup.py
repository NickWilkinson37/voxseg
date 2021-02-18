from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="voxseg",
    version="0.1.0",
    author="Nick Wilkinson",
    author_email="nwilkinsonza@gmail.com",
    description="Python voice activity detection package",
    long_description=readme,
    url="https://github.com/NickWilkinson37/voxseg",
    packages=find_packages(exclude=("tests",)),
    license=license,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.8',
    install_requires=[
        'wheel',
        'pandas', 
        'scipy', 
        'tables',
        'python_speech_features',
        'tensorflow',
    ],
)