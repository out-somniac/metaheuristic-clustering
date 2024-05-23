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


fn evaluate_and_save_results(data: &Data, prediction: &Discrete, truth: &Discrete, algorithm_name: &'static str) {
    let accuracy = 100.0 * metric::accuracy(truth, prediction).unwrap();

    let title = format!("Iris - {}, {:.2}% accuracy", algorithm_name, accuracy);

    let dest = format!("images/accuracy_{}.png", algorithm_name);

    prediction_map::plot(data.clone(), prediction.clone(), 0, 1, &title)
        .unwrap()
        .write_image(
            &dest,
            ImageFormat::PNG,
            640,
            420,
            1.0
        );

    let dest = format!("images/clusters_{}.png", algorithm_name);

    cluster_map::plot(data.clone(), prediction.clone(), 0, 1, &title)
        .unwrap()
        .write_image(
            &dest,
            ImageFormat::PNG,
            640,
            420,
            1.0
        );
}


fn main() -> Result<(), Box<dyn Error>> {
    let data: Data = iris();
    let truth = Discrete::new(&data);

    // GSA

    let solution = gravity::Parameters::new(3)
        .agents(10)
        .gravity(1.0, 1e-2)
        .metric(gravity::Distance::Cosine)
        .normalization(gravity::Normalization::MinMax)
        .fit(&data)?;

    // let solution = gravity::fit(&data, params)?;

    let prediction = solution
        .to_discrete()
        .matched_with(&truth)
        .unwrap();

    evaluate_and_save_results(&data, &prediction, &truth, "gravity");


    // WOA

    // let params = whales::Parameters {
    //     n_classes: 3,
    //     n_agents: 50,
    //     max_iterations: 2000,
    //     spiral_constant: 1.0,
    //     n_spiral_samples: 50
    // };

    // let solution = whales::fit(&data, params)?;
    // let prediction = solution
    //     .to_discrete()
    //     .matched_with(&truth)
    //     .unwrap();

    // evaluate_and_save_results(&data, &prediction, &truth, "whales");


    // K-means

    let prediction = kmeans::fit(&data, 3, 200, 1e-4)?
        .matched_with(&truth)
        .unwrap();

    evaluate_and_save_results(&data, &prediction, &truth, "kmeans");

    // Evaluation

    // let accuracy = 100.0 * metric::accuracy(&truth, &prediction).unwrap();
    // println!("Accuracy: {accuracy} %");

    // prediction_map::plot(data.clone(), prediction.clone(), 0, 1, "Iris")?
    //     .write_image(format!("images/accuracy_{}.png", alg), ImageFormat::PNG, 640, 420, 1.0);

    // cluster_map::plot(data.clone(), prediction.clone(), 0, 1, "Iris")?
    //     .write_image(format!("images/clusters_{}.png", alg), ImageFormat::PNG, 640, 420, 1.0);

    Ok(())
}
