import argparse

import matplotlib.pyplot as plt
import numpy as np


def main(x_max, n_points, linestyle, linecolor):
    x = np.linspace(0, x_max, n_points)
    y = x * x
    fig, ax = plt.subplots()
    ax.plot(x, y, linestyle=linestyle, color=linecolor)
    fig.savefig("plot.png", format="png")

    print(f"{np.amax(x)}^2 = {np.amax(y)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for plotting.')
    parser.add_argument('length', type=float, help='domain length from 0..length')
    parser.add_argument('n_points', type=int, help='How many data points on the domain')
    parser.add_argument('--linestyle', type=str, default=None)
    parser.add_argument('--linecolor', type=str, default="k")
    args = parser.parse_args()
    main(args.length, args.n_points, args.linestyle, args.linecolor)
