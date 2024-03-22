use std::error::Error;
use linfa::dataset::Records;
use linfa_datasets::iris;
use plotly::ImageFormat;
use rand::{self, Rng};

#[allow(unused_imports)]
use clusterization::{Data, cluster_map, prediction_map, scatter_matrix};

const DEST: &'static str = "images/prediction.png";


fn main() -> Result<(), Box<dyn Error>> {
    let data: Data = iris();

    // let plot = scatter_matrix::plot(data, "Iris")?;

    // let plot = cluster_map::plot(data, 0, 1, "Iris")?;

    let n = data.nsamples();
    let prediction = (0..n)
        .map(|_| rand::thread_rng().gen_range(0..=2))
        .collect::<Vec<_>>();

    let plot = prediction_map::plot(
        data,
        prediction,
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
