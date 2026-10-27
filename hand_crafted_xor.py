"""Milestone 1.2: XOR written by hand, no learning involved."""


def xor_gate(a, b):
    return (a * (1 - b)) + (b * (1 - a))


def main():
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for a, b in inputs:
        print(f"{a} XOR {b} = {xor_gate(a, b)}")


if __name__ == "__main__":
    main()
