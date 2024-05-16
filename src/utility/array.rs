
use ndarray::{Array, Dimension};

pub trait Norm<T> {
    fn l2(&self) -> T;
}

impl<D: Dimension> Norm<f64> for Array<f64, D> {
    fn l2(&self) -> f64 {
        (self * self).sum()
    }
}