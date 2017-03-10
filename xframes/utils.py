"""
This module defines top level utility functions for XFrames.
"""

import urllib
import urllib2
import os
import re
from zipfile import ZipFile
import bz2 as _bz2
import tarfile
import ConfigParser
import itertools
import shutil
import random
from sys import stderr
import logging

from pyspark import StorageLevel

from xframes.spark_context import CommonSparkContext
from xframes.xobject import XObject
from xframes import fileio

# Not a number singleton
nan = float('nan')


# noinspection PyUnresolvedReferences
def get_credentials():
    """
    Returns the values stored in the AWS credential environment variables.
    Returns the value stored in the AWS_ACCESS_KEY_ID environment variable and
    the value stored in the AWS_SECRET_ACCESS_KEY environment variable.

    Returns
    -------
    out : tuple [string]
        The first string of the tuple is the value of the AWS_ACCESS_KEY_ID
        environment variable. The second string of the tuple is the value of the
        AWS_SECRET_ACCESS_KEY environment variable.


    Examples
    --------
    >>> xframes.util.get_credentials()
    ('RBZH792CTQPP7T435BGQ', '7x2hMqplWsLpU/qQCN6xAPKcmWo46TlPJXYTvKcv')
    """

    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        raise KeyError('No access key found. Please set the environment variable AWS_ACCESS_KEY_ID.')
    if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
        raise KeyError('No secret key found. Please set the environment variable AWS_SECRET_ACCESS_KEY.')
    return os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY']


def make_internal_url_simple(url):
    """
    Takes a user input url string and translates into url relative to the server process.
    - URL to a local location begins with "local://" or has no "*://" modifier.
      If the server is local, returns the absolute path of the url.
      For example: "local:///tmp/foo" -> "/tmp/foo" and "./foo" -> os.path.abspath("./foo").
      If the server is not local, raise NotImplementedError.
    - URL to a server location begins with "remote://".
      Returns the absolute path after the "remote://" modifier.
      For example: "remote:///tmp/foo" -> "/tmp/foo".
    - URL to a s3 location begins with "s3://":
      Returns the s3 URL with credentials filled in using xframes.aws.get_aws_credential().
      For example: "s3://mybucket/foo" -> "s3://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY:mybucket/foo".
    - URL to other remote locations, e.g. http://, will remain as is.
    - Expands ~ to $HOME

    Parameters
    ----------
    url : str
        A URL (as described above).

    Returns
    -------
    out : str
        Translated url.

    Raises
    ------
    ValueError
        If a bad url is provided.
    """
    if not url:
        raise ValueError('Invalid url: {}'.format(url))

    # Try to split the url into (protocol, path).
    urlsplit = url.split("://")
    if len(urlsplit) == 2:
        protocol, path = urlsplit
        if not path:
            raise ValueError('Invalid url: {}'.format(url))
        if protocol in ['http', 'https']:
            # protocol is a remote url not on server, just return
            return url
        elif protocol == 'hdfs':
            if not fileio.has_hdfs():
                raise ValueError('HDFS URL is not supported because Hadoop not found. '
                                 'Please make hadoop available from PATH or set the environment variable '
                                 'HADOOP_HOME and try again.')
            else:
                return url
        elif protocol == 's3':
            if len(path.split(":")) == 3:
                # s3 url already contains secret key/id pairs, just return
                return url
            else:
                # s3 url does not contain secret key/id pair, query the environment variables
                # k, v = get_credentials()
                # return 's3n://' + k + ':' + v + '@' + path
                return 's3n://' + path
        elif protocol == 'remote':
            # url for files on the server
            path_on_server = path
        elif protocol == 'local' or protocol == 'file':
            # url for files on local client, check if we are connecting to local server
            #
            # get spark context, get master, see if it starts with local
            sc = CommonSparkContext.spark_context()
            if sc.master.startswith('local'):
                path_on_server = path
            else:
                raise ValueError('Cannot use local URL when connecting to a remote server.')
        else:
            raise ValueError('Invalid url protocol {}. Supported url protocols are: '
                             'remote://, local://, file:// s3://, https:// and hdfs://'.format(protocol))
    elif len(urlsplit) == 1:
        # expand ~ to $HOME
        url = os.path.expanduser(url)
        # url for files on local client, check if we are connecting to local server
        if True:
            path_on_server = url
        else:
            raise ValueError('Cannot use local URL when connecting to a remote server.')
    else:
        raise ValueError('Invalid url: {}.'.format(url))

    if path_on_server:
        return os.path.abspath(os.path.expanduser(path_on_server))
    else:
        raise ValueError('Invalid url: {}.'.format(url))


def make_internal_url(compound_url):
    """
    Create a list of urls by processing each list element with make_internal_url_simple.

    Parameters
    ----------
    compound_url : str
        A comma-separated list of urls.  Each url is described in `make_internal_url_simple`.

    Returns
    -------
    out : str
        Translated compound url.

Raises
    ------
    ValueError
        If a bad url is provided.
    """
    return ','.join([make_internal_url_simple(url) for url in compound_url.split(',')])


def download_dataset(url_str, extract=True, force=False, output_dir="."):
    """Download a remote dataset and extract the contents.

    Parameters
    ----------

    url_str : string
        The URL to download from

    extract : bool
        If true, tries to extract compressed file (zip/gz/bz2)

    force : bool
        If true, forces to retry the download even if the downloaded file already exists.

    output_dir : string
        The directory to dump the file. Defaults to current directory.
    """
    fname = output_dir + "/" + url_str.split("/")[-1]
    # download the file from the web
    if not os.path.isfile(fname) or force:
        logging.info("Downloading file from: {}".format(url_str))
        urllib.urlretrieve(url_str, fname)
        if extract and fname[-3:] == "zip":
            logging.info("Decompressing zip archive {}".format(fname))
            ZipFile(fname).extractall(output_dir)
        elif extract and fname[-6:] == ".tar.gz":
            logging.info("Decompressing tar.gz archive {}".format(fname))
            tarfile.TarFile(fname).extractall(output_dir)
        elif extract and fname[-7:] == ".tar.bz2":
            logging.info("Decompressing tar.bz2 archive {}".format(fname))
            tarfile.TarFile(fname).extractall(output_dir)
        elif extract and fname[-3:] == "bz2":
            logging.info("Decompressing bz2 archive: {}".format(fname))
            outfile = open(fname.split(".bz2")[0], "w")
            logging.info("Output file: {}".format(outfile))
            for line in _bz2.BZ2File(fname, "r"):
                outfile.write(line)
            outfile.close()
    else:
        logging.info("File is already downloaded {}".format(fname))


XFRAMES_CURRENT_VERSION_URL = "http://chayden.net/files/xframes_current_version"


def get_newest_version(timeout=5, url=XFRAMES_CURRENT_VERSION_URL):
    """
    Returns the version of XFrames currently available from github.
    Will raise an exception if we are unable to reach the version server.

    Parameters
    ----------
    timeout: int, optional
        How many seconds to wait for the remote server to respond

    url: string, optional
        The URL to go to to check the current version.
    """
    request = urllib2.urlopen(url=url, timeout=timeout)
    version = request.read()
    return version


# noinspection PyBroadException,PyIncorrectDocstring
def perform_version_check(configfile=(os.path.join(os.path.expanduser("~"), ".xframes", "config")),
                          url=XFRAMES_CURRENT_VERSION_URL):
    """
    Checks if currently running version of XFrames is less than the version
    available from github. Prints a message if the version server
    is reachable, and the current version is out of date. Does nothing
    otherwise.

    If the configfile contains a key "skip_version_check" in the Product
    section with non-zero value, this function does nothing.

    Also returns True if a message is printed, and returns False otherwise.
    """
    skip_version_check = False
    try:
        if os.path.isfile(configfile):
            config = ConfigParser.ConfigParser()
            config.read(configfile)
            section = 'Product'
            key = 'skip_version_check'
            skip_version_check = config.getboolean(section, key)
            logging.debug("skip_version_check={}".format(skip_version_check))
    except Exception:
        # eat all errors
        pass

    # skip version check set. Quit
    if not skip_version_check:
        try:
            latest_version = get_newest_version(timeout=1, url=url).strip()
            if parse_version(latest_version) > parse_version(XObject.version()):
                msg = ("A newer version of XPatterns XFrames (v{}) is available! "
                       "Your current version is v{}.\n"
                       "You can use pip to upgrade the xframes package. "
                       "For more information see http://chayden.net/xframes/upgrade.").format(
                           latest_version, XObject.version())
                print >>stderr, msg
                return True
        except:
            # eat all errors
            pass
    return False


# noinspection PyUnusedLocal
def parse_version(version):
    # TODO compute version, once we decide what it looks like
    return 0


def is_directory_archive(path):
    """
    Utiilty function that returns True if the path provided is a directory that has an 
    SFrame or SGraph in it.

    SFrames are written to disk as a directory archive, this function identifies if a given
    directory is an archive for an SFrame.

    Parameters
    ----------
    path : string
        Directory to evaluate.

    Returns
    -------
    True if path provided is an archive location, False otherwise
    """
    if path is None:
        return False

    if not os.path.isdir(path):
        return False

    ini_path = os.path.join(path, 'dir_archive.ini')

    if not os.path.exists(ini_path):
        return False

    if os.path.isfile(ini_path):
        return True

    return False


def get_archive_type(path):
    """
    Returns the contents type for the provided archive path.

    Parameters
    ----------
    path : string
        Directory to evaluate.

    Returns
    -------
    Returns a string of: sframe, sgraph, raises TypeError for anything else
    """
    if not is_directory_archive(path):
        raise TypeError('Unable to determine the type of archive at path: {}'.format(path))

    try:
        ini_path = os.path.join(path, 'dir_archive.ini')
        parser = ConfigParser.SafeConfigParser()
        parser.read(ini_path)

        contents = parser.get('metadata', 'contents')
        return contents
    except Exception as e:
        raise TypeError('Unable to determine type of archive for path: {}'.format(path), e)


GLOB_RE = re.compile("""[*?]""")


def split_path_elements(url):
    parts = os.path.split(url)
    m = GLOB_RE.search(parts[-1])
    if m:
        return parts[0], parts[1]
    else:
        return url, ""


def validate_feature_types(dataset, features, valid_feature_types):
    if features is not None:
        if not hasattr(features, '__iter__'):
            raise TypeError("Input 'features' must be an iterable type.")

        if not all([isinstance(x, str) for x in features]):
            raise TypeError("Input 'features' must contain only strings.")

    # Extract the features and labels
    if features is None:
        features = dataset.column_names()

    col_type_map = {
        col_name: col_type for (col_name, col_type) in
        zip(dataset.column_names(), dataset.column_types())}

    valid_features = []
    for col_name in features:
        if col_type_map.get(col_name) in valid_feature_types:
            valid_features.append(col_name)

    if len(valid_features) == 0:
        if features is not None:
            raise TypeError("No valid feature types specified")
        else:
            raise TypeError("The dataset does not contain any valid feature types")

    return features


def crossproduct(d):
    """
    Create an SFrame containing the crossproduct of all provided options.

    Parameters
    ----------
    d : dict
        Each key is the name of an option, and each value is a list
        of the possible values for that option.

    Returns
    -------
    out : SFrame
        There will be a column for each key in the provided dictionary,
        and a row for each unique combination of all values.

    Example
    -------
    settings = {'argument_1':[0, 1],
                'argument_2':['a', 'b', 'c']}
    print crossproduct(settings)
    +------------+------------+
    | argument_2 | argument_1 |
    +------------+------------+
    |     a      |     0      |
    |     a      |     1      |
    |     b      |     0      |
    |     b      |     1      |
    |     c      |     0      |
    |     c      |     1      |
    +------------+------------+
    [6 rows x 2 columns]
    """

    from xframes import XArray
    d = [zip(d.keys(), x) for x in itertools.product(*d.values())]
    sa = [{k: v for (k, v) in x} for x in d]
    return XArray(sa).unpack(column_name_prefix='')


def delete_file_or_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        os.remove(path)

# Used to merge dictionaries when combining properties.
def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# This is used in lambdas to build row dictionaries
def build_row(names, row, use_columns=None, use_columns_index=None):
    if use_columns:
        names = [name for name in names if name in use_columns]
        row = [row[i] for i in use_columns_index]
    return dict(zip(names, row))


# Random seed
def distribute_seed(rdd, seed):
    # noinspection PyUnusedLocal
    def set_seed(iterator):
        random.seed(seed)
        yield seed

    rdd.mapPartitions(set_seed)

# RDD helpers

# noinspection PyUnresolvedReferences
def cache(rdd):
    rdd.persist(StorageLevel.MEMORY_ONLY)


def uncache(rdd):
    rdd.unpersist()


# noinspection PyUnresolvedReferences
def persist(rdd):
    rdd.persist(StorageLevel.MEMORY_AND_DISK)


def unpersist(rdd):
    rdd.unpersist()


