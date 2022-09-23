from setuptools import setup, find_packages


with open("requirements.txt") as f:
    dependencies = f.read().splitlines()

setup(
    name="dsi24-lib",
    version="0.0.1",
    packages=find_packages("dsi24_lib"),
    install_requires=dependencies,
)
