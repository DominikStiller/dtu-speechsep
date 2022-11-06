#!/usr/bin/env bash

rsync -e 'ssh -q' \
  -av \
  --exclude __pycache__ --exclude *.pyc \
  speechsep dtu-hpc:~/dev/dtu-speechsep
