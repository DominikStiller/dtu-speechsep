#!/usr/bin/env bash

scp -r scp -r dtu-hpc:~/dev/dtu-speechsep/data/lightning_logs/version_$1 data/lightning_logs/
