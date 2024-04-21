use std::error::Error;

use linfa::{
    self,
    prelude::{Fit, Predict},
    Dataset,
};
use linfa_clustering::{self, KMeans};
use linfa_nn::distance::L2Dist;
use rand::thread_rng;

use crate::Data;

use super::solution::Discrete;

fn fit(
    data: &Data,
    n_clusters: usize,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Discrete, Box<dyn Error>> {
    let dataset = Dataset::from(data.records.to_owned());
    let rng = thread_rng();

    let model = KMeans::params_with(n_clusters, rng, L2Dist)
        .max_n_iterations(max_iterations as u64)
        .tolerance(tolerance);

    let model = model.fit(&dataset)?;

    // Ok(Discrete::from(model.predict(&dataset)))
    todo!()
}
