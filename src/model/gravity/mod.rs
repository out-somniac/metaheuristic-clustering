use std::{cmp::Ordering, error::Error, iter};

use ndarray::{s, Array, Array1, Array2, Array3, ArrayView};

use super::solution::{Discrete, Fuzzy};
use crate::Data;

fn compute_masses(fitness: &Vec<f64>, best: f64, worst: f64) -> Array1<f64> {
    let masses = fitness
        .iter()
        .map(|f| (f - worst) / (best - worst))
        .collect::<Vec<_>>();

    let masses_sum: f64 = masses.iter().sum();

    let normalized_masses = masses
        .iter()
        .map(|weight| weight / masses_sum)
        .collect::<Vec<_>>();

    Array::from_vec(normalized_masses)
}

fn fit(
    data: &Data,
    n_samples: usize,
    n_classes: usize,
    agents_total: usize,
    max_iterations: usize,
    initial_gravity: f64,
    gravity_decay: f64,
) -> Result<Discrete, Box<dyn Error>> {
    // Dimensionality of the data
    let n_cols = data.records.ncols();

    // Generate initial population
    let agents: Vec<Fuzzy> = (0..agents_total)
        .map(|_| Fuzzy::random(n_samples, n_classes))
        .collect();

    for time in 0..max_iterations {
        // Compute fitness values
        let fitness = agents
            .iter()
            .map(|agent| agent.fitness(&data))
            .collect::<Vec<_>>();

        // Gravitational constant
        let gravity = initial_gravity * (1.0 / time as f64).powf(gravity_decay);

        // Compute best and worst fitness for iteration
        let best = fitness.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst = fitness.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Compute masses
        let masses = compute_masses(&fitness, best, worst);

        // Compute the total force working on each agent
        let mut total_forces: Array2<f64> = Array2::<f64>::zeros((agents_total, n_cols));
        for (i, &mass_i) in masses.iter().enumerate() {
            for (j, &mass_j) in masses.iter().enumerate() {
                if i == j {
                    continue;
                }

                let x_i = data.records.row(i);
                let x_j = data.records.row(j);
                let difference = &x_j - &x_i;
                let distance = difference.dot(&difference).sqrt();
                let force = gravity * mass_i * mass_j * difference / distance;

                for feature in 0..n_cols {
                    unsafe { *total_forces.uget_mut((i, feature)) += *force.uget(feature) }
                }
            }
        }

        let accelerations = total_forces / masses;
    }

    todo!();
}
