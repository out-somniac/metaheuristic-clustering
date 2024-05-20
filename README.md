## Description
This library contains implementation of algorithms adapted from the following metaheuristics:
 - Gravitational Search Algorithm (Rashedi et al., [2009](https://doi.org/10.1016/j.ins.2009.03.004))
 - Whale Optimization Algorithm (Mirjalili et al., [2016](https://doi.org/10.1016/j.advengsoft.2016.01.008))

 In addition this library contains tools for visualisation and running tests and comparisons on the `iris` dataset. 

## Structure
The `model` module contains all the implementations of algorithms and metrics as well as the solution representation structs.
The `plot` module contains all the utility functions required for visualising crossections of the dataset and the results of our implementations.
The `utility` module contains normalization functions and other generic utility functions. 

## Build instructions
Simply run `cargo build --release`. 🦀😎

## Authors
- [Jan Smółka](https://github.com/integraledelebesgue)
- [Krzysztof Pęczek](https://github.com/out-somniac)
- [Igor Urbanik](https://github.com/Radinyn)
- [Szymon Bednorz](https://github.com/dsonyy)
- [Hieronim Koc](https://github.com/Panyloi)
