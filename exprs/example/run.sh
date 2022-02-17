#!/bin/bash
# This script is to call the package module.


CWD=$(dirname $(readlink -f $0))
SEG_MODULE_DIR=${CWD}/../../
export PYTHONPATH=${SEG_MODULE_DIR}:$PYTHONPATH

cd ${CWD}

python3 -m seg --cfg_file=config.yaml