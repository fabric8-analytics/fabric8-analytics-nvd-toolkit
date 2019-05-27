#!/bin/bash

set -ex

. cico_setup.sh

check_python_version() {
    python3 tools/check_python_version.py 3 6
}

check_python_version
./check-docstyle.sh
