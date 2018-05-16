#!/bin/bash

set -ex

prep() {
    yum -y update
    yum -y install epel-release
    yum -y install python35 python35-virtualenv which
}

prep
./run-linter.sh
