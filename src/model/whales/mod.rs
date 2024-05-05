use super::solution::Fuzzy;
use crate::Data;
use rand::Rng;

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;

pub struct WOAParameters {
    pub n_classes: usize,
    pub agents_total: usize,
    pub max_iterations: usize,
    pub spiral_constant: f64,
}

fn agents_fitness(agents: &Vec<Fuzzy>, data: &Data) -> Vec<f64> {
    agents
        .iter()
        .map(|agent| agent.fitness(&data))
        .collect::<Vec<_>>()
}

pub fn fit(data: &Data, params: WOAParameters) -> Result<Fuzzy, Box<dyn Error>> {
    let n_samples = data.records.nrows();

    let mut agents: Vec<Fuzzy> = (0..params.agents_total)
        .map(|_| Fuzzy::random(n_samples, params.n_classes))
        .collect();

    let mut rng = rand::thread_rng();

    for time in 1..=params.max_iterations {
        let a = 2.0 * params.max_iterations as f64 / time as f64;
        let r_1 = Array2::random((n_samples, params.n_classes), Uniform::new(0.0, 1.0));
        let r_2 = Array2::random((n_samples, params.n_classes), Uniform::new(0.0, 1.0));

        let A: Array2<f64> = a * (2.0 * &r_1 - 1.0);
        let C: Array2<f64> = 2.0 * &r_2;

        for agent in &agents {
            if rng.gen_range(0.0..1.0) > 0.5 {
                if (&A * &A).sum() < 1.0 {
                    todo!("Update the position of the current search agent by the equation (1)");
                } else {
                    todo!("Select a random search agent (X_rand)
                                Update the position of the current search agent by the equation (3)");
                }
            } else {
                todo!("");
            }
        }
    }

    todo!("I was lazy");
}
