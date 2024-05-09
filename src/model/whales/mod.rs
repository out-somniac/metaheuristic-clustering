
use super::solution::Fuzzy;
use crate::Data;
use rand::distributions::uniform::SampleRange;
use rand::rngs::ThreadRng;
use rand::{distributions::Distribution, RngCore};
use rand::Rng;

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::ops::Range;
use std::{f64::consts, error::Error};

pub struct WOAParameters {
    pub n_classes: usize,
    pub n_agents: usize,
    pub max_iterations: usize,
    pub spiral_constant: f64,
}

fn best_agent_index(agents: &Vec<Fuzzy>, data: &Data) -> usize {
    let mut best_index = 0;
    let mut best_fitness = f64::NEG_INFINITY;
    for (i, agent) in agents.iter().enumerate() {
        let fitness = agent.fitness(data);
        best_index = if fitness > best_fitness {
            best_fitness = fitness;
            i
        } else {
            best_index
        }
    }
    best_index
}

fn rand_other_than(value: usize, range: Range<usize>, rng: &mut ThreadRng) -> usize {
    loop {
        let j = rng.gen_range(range.clone());
        if j != value {
            break j;
        }
    }
}

pub fn fit(data: &Data, params: WOAParameters) -> Result<Fuzzy, Box<dyn Error>> {
    let n_samples = data.records.nrows();
    
    let WOAParameters { n_classes, n_agents, max_iterations, spiral_constant } = params;

    let n_dimensions = n_samples * n_classes;

    let mut agents: Vec<Fuzzy> = (0..n_agents)
        .map(|_| Fuzzy::random(n_samples, params.n_classes))
        .collect();

    let mut rng = rand::thread_rng();

    for time in 0..max_iterations {
        let decay_factor = 2.0 - 2.0 * time as f64 / max_iterations as f64;
        let decay = decay_factor * Array2::random(
            (n_samples, n_classes),
            Uniform::new(-1.0, 1.0)
        );

        let randomizer = Array2::random(
            (n_samples, n_classes),
            Uniform::new(0.0, 2.0)
        );

        // println!("{:#?}", decay_factor);

        let agents_direct = &agents as *const Vec<Fuzzy>;

        let best_agent_index = best_agent_index(&agents, data);
        let best_agent = unsafe { &(*agents_direct)[best_agent_index] }; // Allows data race (algorithm sometimes proceeds more dynamically)
        // let best_agent = agents[best_agent_index].clone(); // Prevents data race

        // println!("{:#?}", best_agent.distribution);

        for (i, agent) in agents.iter_mut().enumerate() {
            if rng.gen_range(0.0..1.0) > 0.5 {
                if decay_factor < 1.0 {
                    // Encircling prey
                    let displacement = &randomizer * &best_agent.distribution - &agent.distribution;
                    agent.distribution = &best_agent.distribution - &decay * &displacement;
                } else {
                    // Exploration phase
                    let rand_agent_index = rand_other_than(i, 0..n_agents, &mut rng);
                    let rand_agent = unsafe { &(*agents_direct)[rand_agent_index] };
                    let displacement = &randomizer * &rand_agent.distribution - &agent.distribution;
                    agent.distribution = &rand_agent.distribution - &decay * &displacement;
                }
            } else {
                // Exploitation phase
                let spiral_displacement = Uniform::new(0.0, 1.0).sample(&mut rng);
                let spiral_phase = 2.0 * consts::PI * spiral_displacement;

                let dim1 = rng.gen_range(0..n_dimensions);
                let dim2 = rand_other_than(dim1, 0..n_dimensions, &mut rng);

                let x_index = (dim1 % n_samples, dim1 % n_classes);
                let y_index = (dim2 % n_samples, dim2 % n_classes);

                let exp_factor = (spiral_constant * spiral_displacement).exp();

                unsafe {
                    let best_x = best_agent.distribution.uget(x_index);
                    let best_y = best_agent.distribution.uget(x_index);

                    let x = agent.distribution.uget_mut(x_index);
                    let x_displacement = best_x - *x;

                    *x = x_displacement 
                        * exp_factor
                        * spiral_phase.cos()
                        + best_x;

                    let y = agent.distribution.uget_mut(y_index);
                    let y_displacement = best_y - *y;

                    *y = y_displacement 
                        * exp_factor
                        * spiral_phase.sin()
                        + best_y;
                }

                // let displacement = &best_agent.distribution - &agent.distribution;

                // agent.distribution = &displacement
                //     * (spiral_constant * spiral_phase).exp()
                //     * (2.0 * consts::PI * spiral_phase).cos()
                //     + &best_agent.distribution;
            }
        }

        // println!("");
    }

    let best_agent_index = best_agent_index(&agents, data);
    Ok(agents[best_agent_index].clone())
}
