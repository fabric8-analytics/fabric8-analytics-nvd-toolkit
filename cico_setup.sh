#!/bin/bash -ex

prep() {
    yum -y update
    yum -y install epel-release
    yum -y install python36
}

prep
