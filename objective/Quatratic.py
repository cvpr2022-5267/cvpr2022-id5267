# Implementation of one-dimensional Quadratic function with parameters a, b, c.
#
# f(x) = a^2 * (x + b)^2 + c
#

import numpy as np


class QuadraticFunction:
    def __init__(
        self, output_noise, n_data_per_task, prng,
    ):
        self.bounds = {"a": [0.5, 1.5], "b": [-0.9, 0.9], "c": [-1.0, 1.0]}
        self.search_space = {"X": [-1.0, 1.0]}
        self.output_noise = output_noise
        self.n_data_per_task = n_data_per_task
        self.prng = prng

    def get_meta_data(self):
        """Return the benchmark's metadata.

        Returns:
        --------
        meta_data: numpy.array
            The benchmark's meta_data
        """

        metadata = dict()

        for task, n_data in enumerate(self.n_data_per_task):
            # sample search space
            X = self.prng.uniform(
                low=self.search_space["X"][0],
                high=self.search_space["X"][1],
                size=n_data,
            ).reshape(-1, 1)

            # sample parameters
            a = self.prng.uniform(
                low=self.bounds["a"][0], high=self.bounds["a"][1], size=1
            )
            b = self.prng.uniform(
                low=self.bounds["b"][0], high=self.bounds["b"][1], size=1
            )
            c = self.prng.uniform(
                low=self.bounds["c"][0], high=self.bounds["c"][1], size=1
            )

            # compute y and add noise
            Y = self.__call__(X, a, b, c)
            if self.output_noise:
                y_noise = self.prng.normal(size=Y.shape, scale=self.output_noise)
                Y += y_noise

            # pack metadata in dict
            metadata[task] = {
                "X": X,
                "y": Y,
            }

        return metadata

    def __call__(self, x, a, b, c):
        """Evaluate the quadratic function at the specified points.

        Parameters:
        -----------
        x: numpy.array, shape = (n_samples, 1)
            Numerical representation of the points.
        a, b, c: floats
            The parameters for the quadratic function.
                a^2 * (x+b)^2 + c

        Returns:
        --------
        y: numpy.array
            Observed value at the query points
        """
        y = (np.power(a * (x + b), 2) + c).reshape(-1, 1)
        return y


if __name__ == "__main__":
    # instantiate benchmarks
    benchmark_m = QuadraticFunction(
        output_noise=0.01, n_data_per_task=[128] * 16, prng=np.random.RandomState(1234),
    )
    # draw metadata
    metadata = benchmark_m.get_meta_data()
