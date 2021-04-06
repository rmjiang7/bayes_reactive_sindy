from setuptools import setup, find_packages

setup(
        name = 'rsindy', 
        version  = '0.01',
        packages = find_packages(),
        package_data={'rsindy' : ['models/*.stan']},
        include_package_data=True
)
