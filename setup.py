from setuptools import setup

setup(name='spectral-tree-inference',
    version='0.1',
    description='Spectral methods for fitting Latent Tree models',
    url='https://github.com/NoahAmsel/spectral-tree-inference',
    author='Ariel Jaffe, Noah Amsel, Yariv Aizenbud, Mamie Wang, Amber Hu',
    author_email='yariv.aizenbud@yale.edu',
    license='GPLv3',
    packages=['spectraltree'],
    install_requires=[
        'dendropy',
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite="tests")
