#!/bin/bash
set -e
DIR=`dirname "$0"`
python $DIR/threshold.py $@
