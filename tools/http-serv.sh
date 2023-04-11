#!/bin/bash

DATAPATH=$1

node=$(hostname -s)
port=$(shuf -i8000-9999 -n1)

echo "Node: ${node}"
echo "Port: ${port}"
echo
echo "ssh ${USER}@exahead1.ohsu.edu -L ${port}:${node}:${port}"
echo
echo "http://127.0.0.1:${port}"
echo
echo "https://avivator.gehlenborglab.org/?image_url=http://127.0.0.1:${port}/"

srun http-server --cors='*' --port "$port" "$DATAPATH"
