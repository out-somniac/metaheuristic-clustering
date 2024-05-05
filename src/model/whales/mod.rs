pub struct GSAParameters {}

pub fn fit(data: &Data, params: GSAParameters) -> Result<Fuzzy, Box<dyn Error>> {
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

            let best_agent_index = best_agent_index(&agents, data);
            let best_agent = agents[best_agent_index];

            for agent in &mut agents {
                if rng.gen_range(0.0..1.0) > 0.5 {
                    if (&A * &A).sum() < 1.0 {
                        // Encircling prey
                        let dupa123 = &C * &best_agent.distribution - &agent.distribution;
                        agent.distribution = &best_agent.distribution - &A * &dupa123;
                    } else {
                        // Exploration phase
                        let rand_agent = agents[rng.gen_range(0..params.agents_total)];
                        let dupa123 = &C * &rand_agent.distribution - &agent.distribution;
                        agent.distribution = &rand_agent.distribution - &A * &dupa123;
                    }
                } else {
                    // Exploitation phase
                    let dupa123 = &best_agent.distribution - &agent.distribution;
                    agent.distribution = &dupa123
                        * (params.spiral_constant * time as f64).exp()
                        * (2.0 * 3.14159265358979323846264338327950288_f64 * time as f64).cos()
                        + best_agent.distribution;
                }
            }
        }

        todo!("I was lazy");
    }
}
