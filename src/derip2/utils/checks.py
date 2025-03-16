"""
Utility module for file and directory validation checks.

This module provides functions to verify the existence of files and directories,
create output directories as needed, and validate input paths.
"""

import logging
import os
import sys
from typing import Optional, Tuple


def dochecks(
    usrOutDir: Optional[str] = None, usrLogfile: Optional[str] = None
) -> Tuple[str, str]:
    """
    Validate and create output directory if needed.

    This function ensures a valid output directory exists for writing results.
    If a directory is specified, it will be created if it doesn't exist.
    If no directory is specified, the current working directory will be used.

    Parameters
    ----------
    usrOutDir : str, optional
        Path to the desired output directory (default: None).
    usrLogfile : str, optional
        Path to the desired log file (default: None).

    Returns
    -------
    Tuple[str, str]
        A tuple containing the path to the validated output directory and the path to the log file.

    Notes
    -----
    The function will convert relative paths to absolute paths.
    """
    # If an output directory was specified
    if usrOutDir:
        # Convert to absolute path for consistency
        absOutDir = os.path.abspath(usrOutDir)

        # Create the directory if it doesn't exist
        if not os.path.isdir(absOutDir):
            print(f'Creating output directory: {absOutDir}')
            os.makedirs(absOutDir)

        # Use the specified directory
        outDir = usrOutDir
    # If no output directory was specified
    else:
        # Use the current working directory
        print(f'Setting output directory: {os.getcwd()}')
        outDir = os.getcwd()

    # If a logfile was specified but not an output directory
    if usrLogfile and not usrOutDir:
        # Convert to absolute path for consistency
        logfile_path = os.path.abspath(usrLogfile)
    # If a logfile was specified and an output directory
    elif usrLogfile and usrOutDir:
        # If the logfile is not path and just a filename add logfile to outDir
        if not os.path.dirname(usrLogfile):
            logfile_path = os.path.join(outDir, os.path.basename(usrLogfile))
        else:
            logfile_path = usrLogfile

    # If no logfile was specified
    if not usrLogfile:
        # Use the default logfile path
        logfile_path = os.path.join(outDir, 'derip2.log')

    return outDir, logfile_path


def isfile(path: str) -> str:
    """
    Verify a file exists and return its absolute path.

    This function checks if a specified file exists. If it does,
    returns the absolute path to the file. If it doesn't, logs
    an error message and terminates program execution.

    Parameters
    ----------
    path : str
        Path to the file to check.

    Returns
    -------
    str
        Absolute path to the existing file.

    Raises
    ------
    SystemExit
        If the file does not exist
    """
    # Check if the file exists
    if not os.path.isfile(path):
        # Log error and terminate execution if not found
        logging.error(f'Input file not found: {path}')
        sys.exit(1)
    # File exists
    else:
        # Return the absolute path to the file
        return os.path.abspath(path)
