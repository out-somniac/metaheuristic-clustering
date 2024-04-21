use std::error::Error;
use clusterization::model::solution::{Discrete, Fuzzy};
use itertools::Itertools;
use linfa::dataset::Records;
use linfa_datasets::iris;
use plotly::ImageFormat;

#[allow(unused_imports)]
use clusterization::{Data, plot::{cluster_map, prediction_map, scatter_matrix}};

const DEST: &'static str = "images/prediction.png";


fn main() -> Result<(), Box<dyn Error>> {
    let data: Data = iris();

    // let plot = scatter_matrix::plot(data, "Iris")?;

    // let plot = cluster_map::plot(data, 0, 1, "Iris")?;

    let n_samples = data.nsamples();
    let n_classes = data
        .targets()
        .iter()
        .unique()
        .collect::<Vec<_>>()
        .len();
    
    let population = Fuzzy::random(n_samples, n_classes);
    let prediction: Discrete = population.try_into()?;

    let plot = prediction_map::plot(
        data,
        prediction.to_vec(),
        0,
        1,
        "Iris"
    )?;

    plot.write_image(
        DEST,
        ImageFormat::PNG,
        640,
        420,
        1.0
    );

    Ok(())
}
