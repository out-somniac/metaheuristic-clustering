use std::cmp::Ordering;
use ndarray::{Array2, Axis};

pub trait Normalize {
    fn minmax_inplace(&mut self) -> &mut Self;
    fn logistic_inplace(&mut self) -> &mut Self;
    fn relu_inplace(&mut self) -> &mut Self;
}

impl Normalize for Array2<f64> {
    fn minmax_inplace(&mut self) -> &mut Self {
        self.map_axis_mut(
            Axis(0),
            |mut ax| {
                let maximum = ax.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let minimum = ax.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                ax.mapv_inplace(|x| (x - minimum) / maximum);
            }
        );

        return self;
    }
    
    fn logistic_inplace(&mut self) -> &mut Self {
        self.map_axis_mut(
            Axis(0),
            |mut ax| {
                ax.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            }
        );

        return self;
    }

    fn relu_inplace(&mut self) -> &mut Self {
        self.map_axis_mut(
            Axis(0),
            |mut ax| {
                ax.mapv_inplace(|x| match x.total_cmp(&0.0) {
                    Ordering::Greater => x,
                    _ => 0.0
                })
            }
        );

        return self;
    }
}