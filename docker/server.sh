#!/usr/bin/env bash

PYTHONPATH=${PYTHONPATH}:/base
exec start-notebook.sh --NotebookApp.token=''
