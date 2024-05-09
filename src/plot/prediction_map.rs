use std::error::Error;
use ndarray::{s, Ix1};
use plotly::{
    color::NamedColor,
    common::{Font, Title, Mode, Marker},
    layout::{Axis, Legend}, 
    Layout, Plot, Scatter
};

use crate::{model::solution::Discrete, Data, INDICATOR_COLORS, INDICATOR_LABELS};


pub fn plot(
    data: Data,
    prediction: Discrete,
    x: usize,
    y: usize,
    title: &str
) -> Result<Plot, Box<dyn Error>> {
    let names = data.feature_names();

    let x_name = &names[x];
    let y_name = &names[y];

    let records = data.records();

    let x = records
        .slice(s![.., x])
        .into_dimensionality::<Ix1>()?
        .to_vec();

    let y = records
        .slice(s![.., y])
        .into_dimensionality::<Ix1>()?
        .to_vec();

    let indicator = data
        .targets()
        .iter()
        .zip(prediction.to_vec())
        .map(|(e1, e2)| *e1 == e2);

    let mut matching_x = Vec::<f64>::new();
    let mut matching_y = Vec::<f64>::new();
    let mut non_matching_x = Vec::<f64>::new();
    let mut non_matching_y = Vec::<f64>::new();

    indicator
        .enumerate()
        .for_each(|(i, val)| match val {
            true => {
                matching_x.push(x[i]);
                matching_y.push(y[i]);
            },
            false => {
                non_matching_x.push(x[i]);
                non_matching_y.push(y[i]);
            }
        });

    let layout = Layout::new()
        .title(Title::new(title).font(Font::new().size(25)))
        .x_axis(Axis::new().anchor("x").title(Title::new(x_name)))
        .y_axis(Axis::new().anchor("y").title(Title::new(y_name)))
        .legend(
            Legend::new()
                .title("Prediction".into())
                .border_color(NamedColor::Gray)
                .border_width(1)
        );

    let mut plot = Plot::new();
    plot.set_layout(layout);

    let matching = Scatter::new(matching_x, matching_y)
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .color(INDICATOR_COLORS[1])
                .size(7)
        )
        .name(INDICATOR_LABELS[1]);

    plot.add_trace(matching);

    let non_matching = Scatter::new(non_matching_x, non_matching_y)
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .color(INDICATOR_COLORS[0])
                .size(7)
        )
        .name(INDICATOR_LABELS[0]);

    plot.add_trace(non_matching);
    
    Ok(plot)
}