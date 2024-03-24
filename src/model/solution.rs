use ndarray::{Array1, Array2, Axis};
use ndarray_stats::{errors::MinMaxError, QuantileExt};

use crate::Data;

pub struct Fuzzy(pub Array2<f64>);

impl TryInto<Probabilistic> for Fuzzy {
    type Error = MinMaxError;

    fn try_into(mut self) -> Result<Probabilistic, MinMaxError> {
        for mut row in self.0.axis_iter_mut(Axis(0)) {
            let max = *row.max()?;
            row.mapv_inplace(|x| f64::exp(x - max));
            
            let sum = row.sum();
            row.mapv_inplace(|x| x / sum);
        }

        Ok(Probabilistic(self.0))
    }
}

pub struct Probabilistic(pub Array2<f64>);

impl TryInto<Discrete> for Probabilistic {
    type Error = MinMaxError;

    fn try_into(self) -> Result<Discrete, Self::Error> {
        Ok(Discrete(self.0.map_axis(
            Axis(0),
            |row| row.argmax().unwrap()
        )))
    }
}

pub struct Discrete(pub Array1<usize>);

impl Discrete {
    pub fn from(data: &Data) -> Self {
        let target = data
            .targets()
            .to_owned();

        Discrete(target)
    }
}