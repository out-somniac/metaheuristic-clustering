use linfa_datasets::iris;
use plotly::ImageFormat;
use std::error::Error;

#[allow(unused_imports)]
use clusterization::{
    model::{
        solution::{self, Discrete},
        metric,
        kmeans,
        gravity,
        whales
    },
    plot::{cluster_map, prediction_map, scatter_matrix},
    Data,
};

const DEST: &'static str = "images/prediction_whales.png";

fn main() -> Result<(), Box<dyn Error>> {
    let data: Data = iris();
    let truth = Discrete::from(&data);

    // GSA

    // let gravity_params = gravity::Parameters {
    //     n_classes: 3,
    //     agents_total: 25,
    //     max_iterations: 200,
    //     initial_gravity: 10.0,
    //     gravity_decay: 0.01,
    // };

    // let solution = gravity::fit(&data, gravity_params)?;
    // let prediction = solution
    //     .to_discrete()
    //     .matched_with(&truth)
    //     .unwrap();


    // WOA

    let whale_params = whales::Parameters {
        n_classes: 3,
        n_agents: 25,
        max_iterations: 2000,
        spiral_constant: 5.0
    };

    let solution = whales::fit(&data, whale_params)?;
    let prediction = solution
        .to_discrete()
        .matched_with(&truth)
        .unwrap();

    // K-means

    // let prediction = kmeans::fit(&data, 3, 200, 1e-4)?;


    // Evaluation

    let accuracy = 100.0 * metric::accuracy(&truth, &prediction).unwrap();
    println!("Accuracy: {accuracy} %");

    let plot = prediction_map::plot(data, prediction, 0, 1, "Iris")?;
    // let plot = cluster_map::plot(data, prediction, 0, 1, "Iris")?;

    plot.write_image(DEST, ImageFormat::PNG, 640, 420, 1.0);

    Ok(())
}
