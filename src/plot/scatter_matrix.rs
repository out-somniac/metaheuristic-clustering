use std::error::Error;
use ndarray::{self, s, Ix1};
use linfa::dataset::Records;

use plotly::{
    color::NamedColor, common::{Font, Line, Marker, Mode, Title}, layout::{Annotation, GridPattern, LayoutGrid, Legend}, Histogram, Layout, Plot, Scatter
};

use crate::Data;


pub fn plot(data: Data, title: &str) -> Result<Plot, Box<dyn Error>> {
    let n = data.nfeatures();
    let names = data.feature_names();

    let records = data.records();
    let target = data.targets().to_vec();

    let scatter_marker = Marker::new()
        .color_array(target.iter().map(|i| match i {
            0 => NamedColor::Aquamarine,
            1 => NamedColor::Aqua,
            2 => NamedColor::CornflowerBlue,
            _ => panic!()
        }).collect())
        .size(5);

    let mut plot = Plot::new();

    let mut layout = Layout::new()
        .title(Title::new(title).font(Font::new().size(30).family("Arial")))
        .grid(
            LayoutGrid::new()
                .rows(n)
                .columns(n)
                .pattern(GridPattern::Independent)
        )
        .legend(
            Legend::new()
                .title("Species".into())
                .border_width(1)
                .border_color(NamedColor::Black)
        );

    layout.add_annotation(Annotation::new().text("value"));

    plot.set_layout(layout);

    let mut axis = 1;

    for i in 0..n {
        let x = records
            .slice(s![.., i])
            .into_dimensionality::<Ix1>()?;

        for j in 0..i {
            let x = x.to_vec();

            let y = records
                .slice(s![.., j])
                .into_dimensionality::<Ix1>()?
                .to_vec();

            let trace = Scatter::new(y, x)
                .x_axis(format!("x{axis}"))
                .y_axis(format!("y{axis}"))
                .show_legend(false)
                .name(format!("{} vs. {}", names[i], names[j]))
                .mode(Mode::Markers)
                .marker(scatter_marker.clone());

            plot.add_trace(trace);
           
            axis += 1;
        }

        let hist = Histogram::new(x.to_vec())
            .x_axis(format!("x{axis}"))
            .y_axis(format!("y{axis}"))
            .name(&names[i])
            .show_legend(false)
            .marker(
                Marker::new()
                    .color(NamedColor::CornflowerBlue)
                    .line(
                        Line::new()
                            .color(NamedColor::Black)
                            .width(1.0)
                    )
            );

        plot.add_trace(hist);

        axis += n - i;
    }

    Ok(plot)
}