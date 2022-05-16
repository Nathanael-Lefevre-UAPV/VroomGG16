import sys

import colorama
from colorama import Fore
from colorama import Style

colorama.init()


class PrinterReplace:
    def __init__(self):
        self.first_line = True

    def print(self, *args):
        if self.first_line:
            for arg in args:
                sys.stdout.write(arg)
            self.first_line = False
        else:
            sys.stdout.write('\r')
            for arg in args:
                sys.stdout.write(arg)


def colorprint(color, *args, end='\n'):
    print(color + Style.BRIGHT, end="")
    print(*args, end=end)
    print(Style.RESET_ALL, end="")


def blueprint(*args, end='\n'):
    colorprint(Fore.BLUE, *args, end=end)


def cyanprint(*args, end='\n'):
    colorprint(Fore.CYAN, *args, end=end)


def magentaprint(*args, end='\n'):
    colorprint(Fore.MAGENTA, *args, end=end)


def greenprint(*args, end='\n'):
    colorprint(Fore.GREEN, *args, end=end)


def yellowprint(*args, end='\n'):
    colorprint(Fore.YELLOW, *args, end=end)


def redprint(*args, end='\n'):
    colorprint(Fore.RED, *args, end=end)


if __name__ == "__main__":
    blueprint("hello", 42)
    redprint("hello", 32)