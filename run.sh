#!/bin/bash
set -e
python -m pip install -r requirements.txt
python -m kedro run
