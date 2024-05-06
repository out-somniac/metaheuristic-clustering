
use super::solution::Fuzzy;
use crate::Data;
use rand::distributions::Distribution;
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

pub fn fit(data: &Data, params: WOAParameters) -> Result<Fuzzy, Box<dyn Error>> {
    let n_samples = data.records.nrows();
    let n_agents = params.agents_total;

    let mut agents: Vec<Fuzzy> = (0..n_agents)
        .map(|_| Fuzzy::random(n_samples, params.n_classes))
        .collect();

    let mut rng = rand::thread_rng();

    for time in 1..=params.max_iterations {
        let a = 2.0 - 2.0 * time as f64 / params.max_iterations as f64;

        let A = a *  Array2::random((n_samples, params.n_classes), Uniform::new(-1.0, 1.0));
        let C = Array2::random((n_samples, params.n_classes), Uniform::new(0.0, 2.0));
    
        
        // let A: Array2<f64> = a * (2.0 * &r_1 - 1.0);
        // let C: Array2<f64> = 2.0 * &r_2;

        let best_agent_index = best_agent_index(&agents, data);
        let best_agent = agents[best_agent_index].clone();

        // println!("{:#?}", best_agent.distribution);

        println!("{}", a);

        let agents_ref = &agents as *const Vec<Fuzzy>;

        for (i, agent) in agents.iter_mut().enumerate() {
            if rng.gen_range(0.0..1.0) > 0.5 {
                if a < 1.0 {
                    // Encircling prey
                    println!("Encircling");
                    let dupa123 = &C * &best_agent.distribution - &agent.distribution;
                    agent.distribution = &best_agent.distribution - &A * &dupa123;
                } else {
                    // Exploration phase
                    println!("Exploration");
                    let rand_agent_index = loop {
                        let j = rng.gen_range(0..params.agents_total);
                        if j != i {
                            break j;
                        }
                    };

                    let rand_agent = unsafe { &(*agents_ref)[rand_agent_index] };

                    // let rand_agent = agents[rng.gen_range(0..params.agents_total)].clone();
                    let dupa123 = &C * &rand_agent.distribution - &agent.distribution;
                    agent.distribution = &rand_agent.distribution - &A * &dupa123;
                }
            } else {
                println!("Exploitation");
                // Exploitation phase
                let dupa123 = &best_agent.distribution - &agent.distribution;
                let l = Uniform::new(0.0, 1.0).sample(&mut rng);

                agent.distribution = &dupa123
                    * (params.spiral_constant * l).exp()
                    * (2.0 * 3.14159265358979323846264338327950288_f64 * l).cos()
                    + &best_agent.distribution;
            }
        }

        println!("");
    }

    let best_agent_index = best_agent_index(&agents, data);
    Ok(agents[best_agent_index].clone())
}
