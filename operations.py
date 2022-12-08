def or_(a, b):
    return a or b


def and_(a, b):
    return a and b


def nor(a, b):
    return not (a or b)


def nand(a, b):
    return not (a and b)


def xor(a, b):
    return a != b
