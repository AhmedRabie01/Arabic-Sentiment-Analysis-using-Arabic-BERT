from setuptools import find_packages,setup
from typing import List

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

def get_requirements()->List[str]:
    """
    This function will return list of requirements
    """
    requirement_list:List[str] = []

    """
    Write a code to read requirements.txt file and append each requirements in requirement_list variable.
    """
    install_reqs = parse_requirements('requirements.txt')

    # reqs is a list of requirement
    # e.g. ['django==1.5.1', 'mezzanine==1.4.6']
    reqs = [str(ir) for ir in install_reqs]
    return requirement_list

setup(
    name="Sentiment",
    version="0.0.1",
    author="Ahmed",
    author_email="ahmedrabie10100@gmail.com",
    packages = find_packages(),
    include_package_data = True,
    install_requires=get_requirements(),#["pymongo==4.2.0"],
)