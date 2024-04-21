use std::error::Error;

use itertools::iproduct;
use ndarray::{Array, Array1, Array2, Array3};

use super::solution::Fuzzy;
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

fn total_forces(
    n_samples: usize,
    gravity: f64,
    params: &GSAParameters,
    masses: &Array1<f64>,
    agents: &Vec<Fuzzy>,
) -> Array3<f64> {
    let mut total_forces = Array3::<f64>::zeros((params.agents_total, n_samples, params.n_classes));

    let mass_enumerator = iproduct!(masses.iter().enumerate(), masses.iter().enumerate())
        .filter(|((i, _), (j, _))| i != j);

    for ((i, &mass_i), (j, &mass_j)) in mass_enumerator {
        let x_i: &Array2<f64> = &agents[i].distribution;
        let x_j: &Array2<f64> = &agents[j].distribution;
        let difference = x_j - x_i;
        let distance = (x_i * x_j).sum().sqrt();
        let force = gravity * mass_i * mass_j * difference / distance;

        // FixMe: Nie Za taką Polskę walczyłem :(
        for sample in 0..n_samples {
            for class in 0..params.n_classes {
                unsafe {
                    *total_forces.uget_mut((i, sample, class)) += *force.uget((sample, class));
                }
            }
        }
    }

    return total_forces;
}

pub struct GSAParameters {
    pub n_classes: usize,
    pub agents_total: usize,
    pub max_iterations: usize,
    pub initial_gravity: f64,
    pub gravity_decay: f64,
}

fn fit(data: &Data, params: GSAParameters) -> Result<Fuzzy, Box<dyn Error>> {
    let n_samples = data.records.nrows();

    let mut agents: Vec<Fuzzy> = (0..params.agents_total)
        .map(|_| Fuzzy::random(n_samples, params.n_classes))
        .collect();

    let mut velocities: Array3<f64> =
        Array3::<f64>::zeros((params.agents_total, n_samples, params.n_classes));

    for time in 0..params.max_iterations {
        let fitness = agents
            .iter()
            .map(|agent| agent.fitness(&data))
            .collect::<Vec<_>>();

        let gravity = params.initial_gravity * (1.0 / time as f64).powf(params.gravity_decay);

        let best = fitness.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst = fitness.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let masses = compute_masses(&fitness, best, worst);
        let forces = total_forces(n_samples, gravity, &params, &masses, &agents);
        let accelerations = forces / masses;
        velocities += &accelerations;

        for agent in &mut agents {
            agent.distribution += &velocities
        }
    }

    todo!()
}
