import logging
import os
import sys
from typing import Optional


def dochecks(usrOutDir: Optional[str] = None) -> str:
    """
    Validate and create output directory if needed.

    This function ensures a valid output directory exists for writing results.
    If a directory is specified, it will be created if it doesn't exist.
    If no directory is specified, the current working directory will be used.

    Parameters
    ----------
    usrOutDir : str, optional
        Path to the desired output directory (default: None)

    Returns
    -------
    str
        Path to the validated output directory

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
            logging.info(f'Creating output directory: {absOutDir}')
            os.makedirs(absOutDir)

        # Use the specified directory
        outDir = usrOutDir
    # If no output directory was specified
    else:
        # Use the current working directory
        logging.info(f'Setting output directory: {os.getcwd()}')
        outDir = os.getcwd()

    return outDir


def isfile(path: str) -> str:
    """
    Verify a file exists and return its absolute path.

    This function checks if a specified file exists. If it does,
    returns the absolute path to the file. If it doesn't, logs
    an error message and terminates program execution.

    Parameters
    ----------
    path : str
        Path to the file to check

    Returns
    -------
    str
        Absolute path to the existing file

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
