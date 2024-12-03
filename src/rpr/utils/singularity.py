import numpy as np


def singularity(x, y, fi, R, r):
    return all(
        [
            np.sin(fi) != 0,
            np.pow(x, 2) + np.pow(y, 2)
            != np.pow(R, 2) - 2 * R * r * np.cos(fi) + np.pow(r, 2),
        ]
    )


if __name__ == "__main__":
    print(singularity(x=60, y=0, fi=15, R=100, r=25))
