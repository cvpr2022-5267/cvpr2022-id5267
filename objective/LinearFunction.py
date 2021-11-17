# Implementation of one-dimensional LinearFunction function with parameters a.
#
# f(x) = ax
#

import numpy as np


class LinearFunction:
    def __init__(
        self, output_noise, n_data_per_task, prng, search_space={"X": [-1.0, 1.0]}, bounds={"a": [0.5, 1.5]}
    ):
        self.bounds = bounds
        self.search_space = search_space
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

            # compute y and add noise
            Y = self.__call__(X, a)
            if self.output_noise:
                y_noise = self.prng.normal(size=Y.shape, scale=self.output_noise)
                Y += y_noise

            # pack metadata in dict
            metadata[task] = {
                "X": X,
                "y": Y,
            }

        return metadata

    def __call__(self, x, a):
        """Evaluate the quadratic function at the specified points.

        Parameters:
        -----------
        x: numpy.array, shape = (n_samples, 1)
            Numerical representation of the points.
        a: floats
            The parameters for the quadratic function.
                a * x

        Returns:
        --------
        y: numpy.array
            Observed value at the query points
        """
        y = (x * a).reshape(-1, 1)
        return y


if __name__ == "__main__":
    # instantiate benchmarks
    benchmark_m = LinearFunction(
        output_noise=0.01, n_data_per_task=[128] * 16, prng=np.random.RandomState(1234),
    )
    # draw metadata
    metadata = benchmark_m.get_meta_data()
