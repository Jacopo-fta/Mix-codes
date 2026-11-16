import sys


def cprintf(cond, fmt="", *args):
    """
    CPRINTF conditional fprintf (MATLAB-compatible).

    INPUTS:
        cond: logical condition
        fmt: format string
        *args: arguments formatted with % operator
    """
    if cond:
        if args:
            # Handle case where args is passed as a single tuple or multiple positional args
            if len(args) == 1 and isinstance(args[0], tuple):
                sys.stdout.write(fmt % args[0])
            else:
                sys.stdout.write(fmt % args)
        else:
            sys.stdout.write(str(fmt))
        sys.stdout.flush()


if __name__ == "__main__":
    cprintf(True, "Test cprintf: %d + %d = %d\n", 2, 3, 5)
