import multiprocessing
import traceback
import os
import sys
import glob
import random

import numpy as np
import pickle
import trimesh
from multiprocessing import Pool

# Shortcut to multiprocessing's logger
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise ValueError()

        # It was fine, give a normal answer
        return result


def run_multiprocessing(func, i, n_processors):
    """
    Wrapper for multiprocessing. Logs exceptions in case of bugs.
    """
    with Pool(processes=n_processors) as pool:
        return pool.map(LogExceptions(func), i)