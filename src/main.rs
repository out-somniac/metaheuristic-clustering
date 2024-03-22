use std::error::Error;
use linfa_datasets::iris;
use plotly::ImageFormat;

use clusterization::{Data, cluster_map::cluster_map};

const DEST: &'static str = "images/clusters.png";


fn main() -> Result<(), Box<dyn Error>> {
    let data: Data = iris();

    // let plot = scatter_matrix(data, "Iris")?;

    let plot = cluster_map(data, 0, 1, "Iris")?;

    plot.write_image(
        DEST,
        ImageFormat::PNG,
        640,
        420,
        1.0
    );

    Ok(())
}
