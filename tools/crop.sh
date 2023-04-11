#!/bin/bash
set -e
DIR=`dirname "$0"`
python $DIR/crop.py $@
