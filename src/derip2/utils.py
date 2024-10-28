import os
import sys
import logging


def dochecks(usrOutDir):
    """Make outDir if does not exist else set to current dir."""
    if usrOutDir:
        absOutDir = os.path.abspath(usrOutDir)
        if not os.path.isdir(absOutDir):
            logging.info(f"Creating output directory: {absOutDir}")
            os.makedirs(absOutDir)
        outDir = usrOutDir
    else:
        logging.info(f"Setting output directory: {os.getcwd()}")
        outDir = os.getcwd()
    return outDir


def isfile(path):
    """
    Test for existence of input file.
    """
    if not os.path.isfile(path):
        logging.error(f"Input file not found: {path}")
        sys.exit(1)
    else:
        return os.path.abspath(path)
