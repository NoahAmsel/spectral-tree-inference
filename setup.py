from setuptools import setup
from sphinx.setup_command import BuildDoc

name = 'spectral-tree-inference'
version = '0.1'
release = '0.1.0'
setup(
    name=name,
    version=version,
    description='Spectral methods for fitting Latent Tree models',
    url='https://github.com/NoahAmsel/spectral-tree-inference',
    author='Yariv Aizenbud, Noah Amsel, Ariel Jaffe, Amber Hu, Mamie Wang',
    author_email='yariv.aizenbud@yale.edu',
    license='GPLv3',
    packages=['spectraltree'],
    install_requires=[
        'dendropy',
        'numpy',
        'oct2py',
        'pandas',
        'python-igraph',
        'scipy',
        'scikit-learn',
        'seaborn',
        'sphinx',
        'sphinx-rtd-theme'
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite="tests",
    cmdclass={'build_sphinx': BuildDoc},
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs'),
            'build_dir': ('setup.py', 'docs/_build')}})
