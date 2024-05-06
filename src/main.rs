use clusterization::model::{
    kmeans, metric,
    solution::{self, Discrete}, whales::{self, WOAParameters},
};
use linfa_datasets::iris;
use plotly::ImageFormat;
use std::error::Error;

#[allow(unused_imports)]
use clusterization::{
    model::gravity::{self, GSAParameters},
    plot::{cluster_map, prediction_map, scatter_matrix},
    Data,
};

const DEST: &'static str = "images/prediction.png";

fn main() -> Result<(), Box<dyn Error>> {
    let data: Data = iris();

    // let plot = scatter_matrix::plot(data, "Iris")?;

    // let plot = cluster_map::plot(data, 0, 1, "Iris")?;

    // let solution = Fuzzy::random(n_samples, n_classes);
    let params = GSAParameters {
        n_classes: 3,
        agents_total: 10,
        max_iterations: 50,
        initial_gravity: 10.0,
        gravity_decay: 0.01,
    };

    let whale_params = WOAParameters {
        n_classes: 3,
        agents_total: 15,
        max_iterations: 100,
        spiral_constant: 1.0
    };

    let whales = whales::fit(&data, whale_params)?;

    // println!("{:#?}", whales);

    // let solution = gravity::fit(&data, params)?;

    // let solution = kmeans::fit(&data, 3, 200, 1e-4);
    // println!("{:#?}", solution);

    // let fitness = solution.fitness(&data);
    // println!("Fitness: {fitness}");

    let truth = Discrete::from(&data);
    // let prediction: Discrete = solution.try_into()?;
    // let prediction: Discrete = solution?;
    let prediction = whales.to_discrete();

    let accuracy = 100.0 * metric::accuracy(&truth, &prediction).unwrap();
    println!("Accuracy: {accuracy} %");

    // let plot = prediction_map::plot(data, prediction.to_vec(), 0, 1, "Iris")?;
    let plot = cluster_map::plot(data, prediction, 0, 1, "Iris")?;

    plot.write_image("images/prediction_whales.png", ImageFormat::PNG, 640, 420, 1.0);

    Ok(())
}
