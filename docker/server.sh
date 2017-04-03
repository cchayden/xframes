#!/usr/bin/env bash


cd /base/docs && make html
cp -r _build/html/* /usr/share/nginx/html
service nginx start

cd /notebooks
ipython notebook --no-browser --port 8888 --ip=* --matplotlib=inline
