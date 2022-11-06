#!/usr/bin/env bash

rsync -e 'ssh -q' \
  -av \
  --exclude __pycache__ --exclude *.pyc \
  speechsep bin dtu-hpc:~/dev/dtu-speechsep
