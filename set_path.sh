#!/bin/sh

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

echo $SHELL_FOLDER

export PYTHONPATH=$SHELL_FOLDER