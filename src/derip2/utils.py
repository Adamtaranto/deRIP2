import os
import sys


def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def dochecks(usrOutDir):
    """Make outDir if does not exist else set to current dir."""
    if usrOutDir:
        absOutDir = os.path.abspath(usrOutDir)
        if not os.path.isdir(absOutDir):
            log("Creating output directory: %s" % (absOutDir))
            os.makedirs(absOutDir)
        outDir = usrOutDir
    else:
        log("Setting output directory: %s" % os.getcwd())
        outDir = os.getcwd()
    return outDir


def isfile(path):
    """
    Test for existence of input file.
    """
    if not os.path.isfile(path):
        log("Input file not found: %s" % path)
        sys.exit(1)
    else:
        return os.path.abspath(path)
