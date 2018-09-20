from setuptools import setup, find_packages

# Load Version
exec(open('./prior_pipeline/_version.py', 'r').read())

setup(name='prior_pipeline',
    description='Data analysis pipelines for CCG',
    version=__version__,
    author="GDF",
    license='GPL',
    author_email = 'gdubourg@ccg.ai',
    packages = find_packages(),
    install_requires = ['hyperopt==0.1',
        'pandas==0.23.3',
        'sklearn==0.0',
        'numpy==1.14.5',
        'boto3==1.5.24'
    ],
    include_package_data = True,
    zip_safe = False)
