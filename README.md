# RAtio SelectOR

RASOR is a package comprising algorithms for selecting suitable (in the best case _optimal_) isotopic ratios for Bayesian inference with nuclear reprocessing waste.

## Algorithms

The algorithms in `rasor` function according to a basic scheme:

1. Start with a specific sets if isotopic ratios
2. Compute the likelihood on a test point in the parameter space
3. Determine metrics for the likelihood, e.g., standard deviation of the marginal distributions
4. (Optionally,) repeat for different test points all over the parameter space
5. Repeat with different sets of isotopic ratios
6. Compare metrics to select the best set of isotopic ratios.

There are two conceptually different algorithms (and many options for further variation) for executing the above scheme: _brute force_ and _genetic evolution_.

The _brute-force_ algorithm starts with a list of candidate ratios and then trys all possible combinations of ratios given a fixed ratio set size.
E.g. it repeats the above steps for all possible pairs (or tripletts or quadrupletts or ...) of isotopic ratios that can be created with the candidate ratios.
Then, the best set (pair, triplet or whatever) is selected for each test point.
Optionally, the second (and third or fourth ...) sets at each test point can be included as well.

The _genetic-evolution_ algorithm starts with a random subset of ratio candidates and iteratively changes the ratios according to certain rules (called _mutations_).
This process continues for a specific number of iterations and in each iteration the best performing (according to the fitness function) ratio sets are kept and the rest discarded or modified via a mutation.

## Pre-selection

Before running the above algorithms, the list of candidate ratios needs to be selected.
Several criteria can be applied to remove undesirable elements and nuclides from the potential candidates.

## Using this package

Three scripts are made accessible via the command line:

- `run_pre_selection`
- `run_ratio_selection` (_brute-force_ and _genetic-evolution_ is specified via the input file)
- `run_post_processing` (only for _brute-force_ output at the moment)

### Example inputs

To run the brute-force algorithm the input file should look something like this...

```json
{
    "Algorithm": "brute_force",
    "Problem": {
        "ratios": [
            "As-75/As-84*",
            "Ba-134/Ba-136",
            "Ba-134/Ba-137",
            "Ba-134/Ba-138",
            "Ba-136/Ba-137",
            "Ba-136/Ba-138",
            ...
            "Zr-92/Zr-93",
            "Zr-92/Zr-94",
            "Zr-92/Zr-96",
            "Zr-93/Zr-94",
            "Zr-93/Zr-96",
            "Zr-94/Zr-96"
        ],
        "limits": [
            [
                0.1,
                0.03,
                0.0,
                0.72
            ],
            [
                8.491796875,
                0.99810546875,
                9980.46875,
                1.49923828125
            ]
        ]
    },
    "Metric": {
        "method": "sobol",
        "bin_number": 100,
        "m": 16
    },
    "Likelihood": {
        "uncertainty": {
            "uncertainty_model": "constant",
            "value": 0.1
        },
        "surrogates": {
            "x": "/path/to/interpolators/x_grid.npy",
            "y": "/path/to/interpolators/y_grid.npy",
            "grid_size": [
                25,
                25,
                25,
                25
            ]
        }
    },
    "Test_points": {
        "method": "sobol",
        "m": 6,
        "seed": 12345,
        "limits": [
            [
                0.1,
                0.03,
                0.0,
                0.72
            ],
            [
                8.491796875,
                0.99810546875,
                9980.46875,
                1.49923828125
            ]
        ]
    }
}
```

To run _genetic-evolution_, replace "brute_force" with the following:

```json
{
    "genetic_evolution": {
        "init_size": 1000,
        "init_length": 10,
        "max_iter": 200,
        "mutations": {
            "gene_swap": 0.2,
            "cross_over": 0.2,
            "deletion": 0.1,
            "addition": 0.1
        },
        "rng_seed": 12345
    }
}
```

In any case, the algorithms require input and output data for a regular grid interpolator.

Regarding the metric, the marginal distributions can be approximated either via a grid-based or a Sobol-sequence-based algorithm.
The Sobol sequence-based algorithm scales better than the grid-based algorithm, but is usually slower in lower dimensions (2 or lower).
The reason is that the Sobol sequence needs about 2^m evaluations, where m should be larger than 16.