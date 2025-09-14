from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [reg.replace("\n", "") for reg in requirements]

        if HYPEN_E_DOT is requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



setup(
    name='wind-turbine-ml',
    version='0.0.1',
    author='Osama-Baakeem',
    author_email='osama.baakeem2@gmail.com',
    description='End-to-end ML system for wind turbine power prediction',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)