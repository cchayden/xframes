Build and Run XFrames Docker
============================

These commands run Spark and XFrames in a docker,
which you can access with Jupyter.

From the xframes main directory:

    cd docker

Build
-----
    docker-compose buld

Run
---
    docker-compose up

Stop Docker
-----------
    docker-compose stop

Enter Docker
------------
    docker-compose exec xframes /bin/bash
    
Using XFrames Docker
====================

Run notebook
------------
Browse to localhost:8888.

You can test by creatng a (python2) notebook, and then
entering the test program in the first call.

    from xframes import XFrame

    xf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
    print xf

View spark console
------------------
Browse to localhost:4040.

This shows the Spark console