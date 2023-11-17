from setuptools import setup, find_packages

setup(
    name='phrases_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gensim',
        'sklearn',
        'numpy',
        'scikit-learn',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'phrases_pipeline=phrases_pipeline.__main__:main',
        ],
    }
)