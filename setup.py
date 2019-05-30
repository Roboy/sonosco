from setuptools import setup, find_packages

setup(
    name="sonosco",
    description="Framework for training automatic speech recognition systems.",
    author="The Roboy Gang",
    packages=["sonosco"],
    include_package_data=True,
    dependency_links=['http://github.com/pytorch/audio/tarball/master#egg=torchaudio-0.2']
)
