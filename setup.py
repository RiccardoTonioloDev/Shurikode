from setuptools import setup, find_packages

setup(
    name="shurikode",
    sum="A 2D, highly redundant code.",
    author="Riccardo Toniolo",
    author_email="ssctonioloriccardo@gmail.com",
    version="0.1.0",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "shurikode.ml_model": ["ResNet50.pth.tar"],
    },
    install_requires=["Pillow", "numpy", "torch", "torchvision"],
)
