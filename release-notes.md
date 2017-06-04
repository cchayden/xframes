# Release To PyPi
This documents the manual steps having to do with releasing a
new version of xframes to pypi.

## Run all unit tests
Proceed only if they all pass.
You might also want to test hdfs (which required hdfs docker).

    cd xframes/test
    ./runtests
    
## Check in and push
Push to the master branch.

## Prepare for distribution

    ./make-dist

## Register xframes with pypi

    python setup.py register -r pypi

## Upload a new version
Edit xframes/version.py and increment the version appropriately.

    python setup.py sdist upload -r pypi


## Setting up .pypirc
This supplies information for PyPi.

Put the following in ~/.pypirc

    [distutils]
    index-servers =
      pypi
      pypitest

    [pypi]
    repository: https://pypi.python.org/pypi
    username: cchayden
    password: ******

    [pypitest]
    repository: https://testpypi.python.org/pypi
    username: cchayden
    password: ******

## Using the PyPi Test Server

Since version numbers can never be reused, each time a new
version is uploaded that number can never be used again.
To experiment with issues involving upload and download you can
use the test server.  Although it enforces the same rules about version
numbers, they do not share with the regular server.

First, make an account with the server at pypitest.
Second, change `-r pypi` to `-r pypitest` above.

    python setup.py sdist upload -r pypitest


Finally, on pip install, use `-i https://testpypi.python.org/pypi`
