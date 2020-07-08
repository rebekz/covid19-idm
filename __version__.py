from __future__ import print_function
from builtins import map
import os
# https://semver.org/
MAJOR_VER = 0
MINOR_VER = 1
PATCH_VER = 0
BUILD_NUMBER = os.environ.get('BUILD_NUMBER', "SNAPSHOT")

VERSION = (MAJOR_VER, MINOR_VER, PATCH_VER)

__version__ = '.'.join(map(str, VERSION))

if __name__ == '__main__':
    print(f"{__version__}-{BUILD_NUMBER}")
