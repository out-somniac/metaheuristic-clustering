use derive_error::Error as StdError;

#[allow(unused_imports)]
use itertools::{iproduct, Itertools};

use ndarray::{s, Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_stats::DeviationExt;
use rand::{distributions::Uniform, random};

use super::solution::Fuzzy;
use crate::{utility::{array::Norm, normalization::Normalize, order::Ordered}, Data};

#[derive(Debug, Clone, Copy)]
pub enum Distance {
    Cosine,
    L2,
    LInf
}


// TODO: Consider passing default closures as a hyperparameter
#[derive(Debug, Clone, Copy)]
pub enum Normalization {
    Logistic,
    MinMax,
    ReLU
}

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub n_classes: usize,
    pub n_agents: usize,
    pub max_iterations: usize,
    pub initial_gravity: f64,
    pub gravity_decay: f64,
    pub distance: Distance,
    pub normalization: Normalization
}

#[derive(Debug, Clone, StdError)]
pub enum Error {  // TODO: Review these variants
    UndefinedOrder,
    IterationProcessDivergent
}

pub type Result<T> = core::result::Result<T, Error>;

// TODO: Considet moving into a `Builder` trait
impl Parameters {
    pub fn new(n_classes: usize) -> Parameters {
        Parameters {
            n_classes,
            n_agents: 10,
            max_iterations: 500,
            initial_gravity: 1.0,
            gravity_decay: 0.01,
            distance: Distance::Cosine,
            normalization: Normalization::MinMax
        }
    }

    pub fn agents(mut self, n: usize) -> Parameters {
        self.n_agents = n;
        self
    }

    pub fn iterations(mut self, n: usize) -> Parameters {
        self.max_iterations = n;
        self
    }

    pub fn gravity(mut self, initial_value: f64, decay: f64) -> Parameters {
        self.initial_gravity = initial_value;
        self.gravity_decay = decay;
        self
    }

    pub fn metric(mut self, method: Distance) -> Parameters {
        self.distance = method;
        self
    }

    pub fn normalization(mut self, method: Normalization) -> Parameters {
        self.normalization = method;
        self
    }

    pub fn fit(self, data: &Data) -> Result<Fuzzy> {
        fit(data, self)
    }
}

// TODO: Move into the `Parameters` struct
const TOLERANCE: f64 = 1e-16;

fn fitness(agents: &Vec<Fuzzy>, data: &Data) -> Vec<f64> {
    agents
        .iter()
        .map(|agent| agent.fitness(&data))
        .collect::<Vec<_>>()
}

fn masses(fitness: &Vec<f64>) -> Array1<f64> {
    let (worst, best) = fitness.min_max().unwrap();
    let range = best - worst;

    let masses = fitness
        .iter()
        .map(|f| (f - worst) / range)
        .collect::<Vec<_>>();

    let total_mass: f64 = masses.iter().sum();

    let masses = masses
        .iter()
        .map(|mass| mass / total_mass + TOLERANCE)
        .collect::<Vec<_>>();

    Array::from_vec(masses)
}

#[inline(always)]
fn cosine_distance(x: &Array2<f64>, y: &Array2<f64>) -> f64 {
    let x_norm = x.l2();
    let y_norm = y.l2();
    let inner = (x * y).sum();

    let similarity = inner / (x_norm * y_norm).sqrt();

    (1.0 - similarity) / 2.0
}

fn total_forces(
    n_samples: usize,
    n_classes: usize,
    n_agents: usize,
    gravity: f64,
    masses: &Array1<f64>,
    agents: &Vec<Fuzzy>,
    distance: Distance
) -> Array3<f64> {
    let mut total_forces = Array3::<f64>::zeros((n_agents, n_samples, n_classes));

    let mass_enumerator = iproduct!(
        masses.iter().enumerate(),
        masses.iter().enumerate()
    ).filter(|((i, _), (j, _))| i != j);

    for ((i, &mass_i), (j, &mass_j)) in mass_enumerator {
        let x_i: &Array2<f64> = &agents[i].distribution;
        let x_j: &Array2<f64> = &agents[j].distribution;

        let difference = x_j - x_i;

        // TODO: Consider passing a default closure
        let distance = match distance {
            Distance::Cosine => cosine_distance(x_i, x_j),
            Distance::L2 => x_i.l2_dist(x_j).unwrap(),
            Distance::LInf => x_i.linf_dist(x_j).unwrap()
        };

        let force = gravity * mass_i * mass_j * difference / distance;

        let random_factor = random::<f64>();

        total_forces
            .slice_mut(s![i, .., ..])
            .scaled_add(random_factor, &force);
    }

    total_forces
}

#[inline(always)]
fn gravity(initial: f64, decay: f64, time: f64, max_time: f64) -> f64 {
    initial * (-decay * time / max_time).exp()
}

pub fn fit(data: &Data, params: Parameters) -> Result<Fuzzy> {
    let n_samples = data.records.nrows();

    let Parameters {
        n_classes,
        n_agents,
        max_iterations,
        initial_gravity,
        gravity_decay,
        distance,
        normalization
    } = params;

    let mut agents: Vec<Fuzzy> = (0..n_agents)
        .map(|_| Fuzzy::random(n_samples, n_classes))
        .collect();

    let mut velocities: Array3<f64> =
        Array3::<f64>::zeros((n_agents, n_samples, n_classes));

    let max_time = max_iterations as f64;

    for time in 0..max_iterations {
        let gravity = gravity(
            initial_gravity,
            gravity_decay,
            time as f64,
            max_time
        );

        let fitness = fitness(&agents, &data);

        let masses = masses(&fitness);

        let mut forces = total_forces(
            n_samples,
            n_classes,
            n_agents,
            gravity,
            &masses,
            &agents,
            distance
        );

        let masses = masses
            .into_shape((n_agents, 1, 1))
            .unwrap();

        forces /= &masses;

        let randomizer = Array3::random((n_agents, 1, 1), Uniform::new(0.0, 1.0));

        velocities *= &randomizer;
        velocities += &forces;

        // println!("Min-Max velocity: {:?}", velocities.clone().into_iter().collect_vec().min_max());

        for (i, agent) in agents.iter_mut().enumerate() {
            agent.distribution += &velocities.slice(s![i, .., ..]);

            match normalization {
                Normalization::ReLU => agent.distribution.relu_inplace(),
                Normalization::MinMax => agent.distribution.minmax_inplace(),
                Normalization::Logistic => agent.distribution.logistic_inplace()
            };
        }
    }

    let fitness = fitness(&agents, &data);
    let best = fitness.argmax().unwrap();

    Result::Ok(agents[best].clone())
}
