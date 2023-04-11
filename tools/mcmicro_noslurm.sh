#!/bin/bash
set -e

if [[ -z "$NXF_HOME" ]]; then
    echo "You must set NXF_HOME! See github docs for more info!" 1>&2
    exit 1
fi

DATA=$1
CFG=$2
JVMMEM=$3
JVM_OPTS="-Xms500M -Xmx${JVMMEM}G"
ALIGN_CHANNEL=$4
GIT_HASH="6266d7a3624407997846505f883d9be4612ec496"

export PATH="/opt/singularity/current/bin:$PATH"

export NXF_EXECUTOR=local
export SINGULARITY_CACHEDIR=${NXF_HOME}
export NXF_SINGULARITY_CACHEDIR=${NXF_HOME}
export NXF_OPTS=$JVM_OPTS

CMD="nextflow -log $DATA/logs/nextflow.log run labsyspharm/mcmicro -r $GIT_HASH --in $DATA -w $DATA/mcmicro_workdir -c $CFG -profile singularity --start-at illumination --stop-at registration --ashlar-opts '--flip-y -m 30 --filter-sigma 1.0 --pyramid --align-channel $ALIGN_CHANNEL'"
echo $CMD

nextflow -log $DATA/logs/nextflow.log run labsyspharm/mcmicro -r $GIT_HASH --in $DATA -w $DATA/mcmicro_workdir -c $CFG -profile singularity --start-at illumination --stop-at registration -params-file $DATA/ashlar-opts.yml
