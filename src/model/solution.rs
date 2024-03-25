use ndarray::{Array1, Array2, Axis};
use ndarray_stats::{errors::MinMaxError, QuantileExt};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;


use crate::Data;

#[derive(Debug)]
pub struct Fuzzy(pub Array2<f64>);

impl Fuzzy {
    pub fn random(n_samples: usize, n_classes: usize) -> Self {
        Fuzzy(Array2::random(
            (n_samples, n_classes),
            Uniform::new(0.0, 1.0)
        ))
    }
}

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

impl TryInto<Discrete> for Fuzzy {
    type Error = MinMaxError;

    fn try_into(self) -> Result<Discrete, Self::Error> {
        let prob: Probabilistic = self.try_into()?;
        prob.try_into()
    }
}

#[derive(Debug)]
pub struct Probabilistic(pub Array2<f64>);

impl TryInto<Discrete> for Probabilistic {
    type Error = MinMaxError;

    fn try_into(self) -> Result<Discrete, Self::Error> {
        Ok(Discrete(self.0.map_axis(
            Axis(1),
            |row| row.argmax().unwrap()
        )))
    }
}

#[derive(Debug)]
pub struct Discrete(pub Array1<usize>);

impl Discrete {
    pub fn from(data: &Data) -> Self {
        let target = data
            .targets()
            .to_owned();

        Discrete(target)
    }

    pub fn to_vec(self) -> Vec<usize> {
        self.0.to_vec()
    }
}

impl Into<Vec<usize>> for Discrete {
    fn into(self) -> Vec<usize> {
        self.to_vec()
    }
}