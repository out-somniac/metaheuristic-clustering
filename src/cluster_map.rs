use std::error::Error;
use itertools::Itertools;
use ndarray::{self, s, Ix1};

use plotly::{
    color::NamedColor, 
    common::{Font, Marker, Mode, Title},
    layout::{Axis, Legend},
    Layout, Plot, Scatter
};

use crate::Data;


pub fn cluster_map(data: Data, x: usize, y: usize, title: &str) -> Result<Plot, Box<dyn Error>> {
    let names = data.feature_names();

    let x_name = &names[x];
    let y_name = &names[y];

    let records = data.records();

    let target = data.targets();

    let groups = target
        .iter()
        .cloned()
        .unique()
        .collect_vec();

    let colors = vec![
        NamedColor::Aquamarine,
        NamedColor::Aqua,
        NamedColor::CornflowerBlue
    ];

    let labels = vec![
        "Setosa",
        "Versicolor",
        "Virginica"
    ];

    let x = records
        .slice(s![.., x])
        .into_dimensionality::<Ix1>()?
        .to_vec();

    let y = records
        .slice(s![.., y])
        .into_dimensionality::<Ix1>()?
        .to_vec();

    let target = target.to_vec();

    let mut plot = Plot::new();

    let layout = Layout::new()
        .title(Title::new(title).font(Font::new().size(25)))
        .x_axis(Axis::new().anchor("x").title(Title::new(x_name)))
        .y_axis(Axis::new().anchor("y").title(Title::new(y_name)))
        .legend(
            Legend::new()
                .title("Species".into())
                .border_color(NamedColor::Gray)
                .border_width(1)
        );

    plot.set_layout(layout);

    for group in groups {
        let x = x
            .iter()
            .cloned()
            .enumerate()
            .filter(|(i, _)| target[*i] == group)
            .map(|(_, val)| val)
            .collect_vec();

        let y = y
            .iter()
            .cloned()
            .enumerate()
            .filter(|(i, _)| target[*i] == group)
            .map(|(_, val)| val)
            .collect_vec();

        let trace = Scatter::new(x, y)
            .mode(Mode::Markers)
            .marker(
                Marker::new()
                    .color(colors[group])
                    .size(7)
            )
            .name(labels[group]);

        plot.add_trace(trace);
    }

    Ok(plot)
}