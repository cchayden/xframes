# Notes for running xframes out of the source directory.

## Virtual Environment
Activate the virtual env, if this is where xframes is going to be installed.
Otherwise it will install into the global environment, which will probably require sudo.

    mkdir tmp
    cd tmp
    virtualenv venv
    source venv/bin/activate

## For now, install dependencies

    pip install numpy python-distutils

## Install xframes

    pip install xframes

## To run xframes application
Set up pyspark -- replace with where you have unpacked spark.

    export SPARK_HOME=~/tools/spark
    export PYTHONPATH=${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.4-src.zip

## Config
You probably want to set up spark config.
If you do not, you will see a lot of debug output on stdout.
There is a sample config in config/conf.

    cp -r ~/workspaces/xframes/xframes/conf .
    export SPARK_CONF_DIR=`pwd`/conf

## Run test
    python test.py

