use ndarray::{self, ArrayBase, Dim, OwnedRepr};
use linfa::DatasetBase;
use plotly::color::NamedColor;

pub type Data = DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>>;

pub mod model;
pub mod plot;

pub mod utility;

pub const IRIS_LABELS: [&'static str; 3] = [
    "Setosa",
    "Versicolor",
    "Virginica"
];

pub const IRIS_COLORS: [NamedColor; 3] = [
    NamedColor::Aquamarine,
    NamedColor::Aqua,
    NamedColor::CornflowerBlue
];

pub const INDICATOR_LABELS: [&'static str; 2] = [
    "Incorrect",
    "Correct"
];

pub const INDICATOR_COLORS: [NamedColor; 2] = [
    NamedColor::Red,
    NamedColor::LimeGreen
];