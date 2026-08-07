# benchmark-redundancy

How much of a benchmark is doing work, and how much is repeating itself.

The question is whether a reported ranking between two methods survives when the
benchmark is subsampled, and whether the redundancy among items is large enough
that the comparison is decided by a handful of them.

## Layout

    redundancy_study.py     redundancy across benchmark items
    subset_robustness.py    does the ranking survive subsampling
    make_figures_numpy.py   figures, numpy only, no plotting stack required

## Running

    python redundancy_study.py
    python subset_robustness.py
    python make_figures_numpy.py
