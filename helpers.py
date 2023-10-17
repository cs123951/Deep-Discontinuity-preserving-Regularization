import sys


def str_to_class(name, classname):
    return getattr(sys.modules[name], classname)
