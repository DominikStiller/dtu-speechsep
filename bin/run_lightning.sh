#!/usr/bin/env bash

if [ -d "venv" ]; then
   source venv/bin/activate
fi

PYTHONPATH=. python speechsep/lightning.py
