use super::solution::Discrete;
use ndarray;

pub fn accuracy(truth: &Discrete, prediction: &Discrete) -> Result<f64, ndarray::ErrorKind> {
    let n = truth.0.dim();
    
    if n != prediction.0.dim() {
        return Err(ndarray::ErrorKind::IncompatibleShape);
    }

    let matches: usize = truth.0
        .iter()
        .zip(&prediction.0)
        .map(|(x, y)| (*x == *y) as usize)
        .sum();

    Ok(matches as f64 / n as f64)
}
