#!/bin/bash
set -e
echo "make-ometiff.sh: starting"
DATAFILE=$1
TMPDIR=$2
VIZFILE=$3
echo "bioformats2raw $1 $2"
echo "raw2ometiff $2 $3"
bioformats2raw $1 $2
echo "Finished running bioformats2raw"
raw2ometiff $2 $3
echo "Finished running raw2ometiff"
echo "make-ometiff.sh: finished"