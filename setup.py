#!/usr/bin/env python
# -*- coding:utf-8 -*-


from setuptools import setup, find_packages
import getopt
import sys
import os

if __name__ == "__main__":

    # set version
    try:
        argv = sys.argv[1:]
        opts, args = getopt.getopt(argv, 'h', ['help','version='])
        version = None
        for opt, arg in opts:
            if opt in ('-h','--help'):
                print('python setup.py --version <version>')
                sys.exit()
            if opt in ('--version'):
                version = arg
        assert version is not None
    except getopt.GetoptError:
        print('before setup:\n  pipreqs <package>\n  python setup.py sdist\n')
        print('run with:\n  setup.py --version <version>\n')
        sys.exit(2)

    # upload
    try:
        setup(
            name="torchmods",
            version=str(version),
            keywords=("cv", "pytorch", "auxiliary"),
            description="mods for torch & cv",
            long_description="experimental mods for pytorch & cv research",
            license="MIT Licence",

            url="https://github.com/klrc/torchmods",
            author="klrc",
            author_email="sh@mail.ecust.edu.com",

            packages=find_packages(),
            include_package_data=True,
            platforms=["all"],
            install_requires=["torch", "torchvision", "paramiko"]
        )
        if os.system(f'twine upload dist/labvision-{version}.tar.gz') != 0:
            print('before setup:\n  pipreqs <package>\n  python setup.py sdist\n')
            sys.exit(2)
        print('to install:\n  pip install torchmods -U -i https://pypi.org/simple')
    except Exception as e:
        raise e