use super::solution::Discrete;

pub fn accuracy(truth: &Discrete, prediction: &Discrete) -> Result<f64, ndarray::ErrorKind> {
    let n_samples = truth.n_samples;

    if n_samples != prediction.indicators.dim() {
        return Err(ndarray::ErrorKind::IncompatibleShape);
    }

    let mut matching: usize = 0;

    let prediction = prediction
        .to_owned()
        .matched_with(truth)?;

    truth
        .indicators
        .iter()
        .zip(prediction.indicators.iter())
        .for_each(|(&t, &p)| {
            if t == p {
                matching += 1
            }
        });

    Ok(matching as f64 / n_samples as f64)
}
