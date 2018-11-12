from setuptools import setup, find_packages

setup(
    name='pandas-anaphora',
    author='Gregory Werbin',
    author_email='outthere@me.gregwerbin.com',
    description='Anaphoric functions for columns in Pandas data frames',
    url='https://github.com/gwerbin/pandas-anaphora',
    license='MIT',
    version='0.1b0',
    packages=find_packages('.', exclude=['test', 'test.*', 'tests', 'tests.*']),
    python_requires='>= 3.4',
    install_requires=['setuptools', 'pandas>=0.20', 'toolz>=0.9'],  # cytoolz>=0.9
    include_package_data=True,
    project_urls={
        'Bug Tracker': 'https://github.com/gwerbin/pandas-anaphora/issues',
        'Documentation': 'https://github.com/gwerbin/pandas-anaphora/blob/master/README.md',
        'Source Code': 'https://github.com/gwerbin/pandas-anaphora',
    }
)
