use super::solution::Discrete;
use ndarray;

pub fn accuracy(truth: &Discrete, prediction: &Discrete) -> Result<f64, ndarray::ErrorKind> {
    let n = truth.indicators.dim();
    
    if n != prediction.indicators.dim() {
        return Err(ndarray::ErrorKind::IncompatibleShape);
    }

    let matches: usize = truth.indicators
        .iter()
        .zip(&prediction.indicators)
        .map(|(x, y)| (*x == *y) as usize)
        .sum();

    Ok(matches as f64 / n as f64)
}
