#!/bin/bash
set -e

if [[ -z "$NXF_HOME" ]]; then
    echo "You must set NXF_HOME! See github docs for more info!" 1>&2
    exit 1
fi

export PATH="/opt/singularity/current/bin:$PATH"
export SINGULARITY_CACHEDIR=${NXF_HOME}
export NXF_SINGULARITY_CACHEDIR=${NXF_HOME}
export NXF_OPTS="-Xms500M -Xmx64G"
export NXF_EXECUTOR=slurm
GIT_HASH="6266d7a3624407997846505f883d9be4612ec496"

nextflow pull labsyspharm/mcmicro
nextflow run labsyspharm/mcmicro/exemplar.nf --name exemplar-001 --path .
nextflow run labsyspharm/mcmicro -r $GIT_HASH --in exemplar-001 -profile singularity --start-at illumination --stop-at registration

rm -rf exemplar-001/ work/ .nextflow*
