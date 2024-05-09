use super::solution::Discrete;
use pathfinding::{kuhn_munkres::kuhn_munkres, matrix::Matrix};

pub fn accuracy(truth: &Discrete, prediction: &Discrete) -> Result<f64, ndarray::ErrorKind> {
    let n_classes = truth.n_classes;
    let n_samples = truth.n_samples;
    
    if n_samples != prediction.indicators.dim() {
        return Err(ndarray::ErrorKind::IncompatibleShape);
    }

    let mut cost = Matrix::new(n_classes, n_classes, 0isize);

    truth
        .indicators
        .iter()
        .zip(prediction.indicators.iter())
        .for_each(|(&t, &p)| unsafe {
            *cost.get_unchecked_mut(t * n_classes + p) += 1
        });

    let (score, _) = kuhn_munkres(&cost);

    Ok(score as f64 / n_samples as f64)
}
