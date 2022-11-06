#!/usr/bin/env bash

rsync -e 'ssh -q' \
  -av --progress -h \
  dtu-hpc:~/dev/dtu-speechsep/data/lightning_logs/version_$1 data/lightning_logs/
