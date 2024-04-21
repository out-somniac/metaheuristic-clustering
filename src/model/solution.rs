use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::{errors::MinMaxError, QuantileExt};

use crate::Data;

#[derive(Debug, Clone)]
pub struct Fuzzy {
    pub distribution: Array2<f64>,
    pub n_samples: usize,
    pub n_classes: usize,
}

impl Fuzzy {
    pub fn random(n_samples: usize, n_classes: usize) -> Self {
        let distribution = Array2::random((n_samples, n_classes), Uniform::new(0.0, 1.0));
        Fuzzy {
            distribution,
            n_samples,
            n_classes,
        }
    }

    pub fn fitness(&self, data: &Data) -> f64 {
        let indicator = self.clone().to_discrete().to_vec();
        0.0
    }

    pub fn to_prob(self) -> Probabilistic {
        self.try_into().unwrap()
    }

    pub fn to_discrete(self) -> Discrete {
        self.try_into().unwrap()
    }
}

impl TryInto<Probabilistic> for Fuzzy {
    type Error = MinMaxError;

    fn try_into(mut self) -> Result<Probabilistic, MinMaxError> {
        for mut row in self.distribution.axis_iter_mut(Axis(0)) {
            let max = *row.max()?;
            row.mapv_inplace(|x| f64::exp(x - max));

            let sum = row.sum();
            row.mapv_inplace(|x| x / sum);
        }

        Ok(Probabilistic(self.distribution))
    }
}

impl TryInto<Discrete> for Fuzzy {
    type Error = MinMaxError;

    fn try_into(self) -> Result<Discrete, Self::Error> {
        Ok(Discrete(self.distribution.map_axis(
            Axis(1),
            |row| row.argmax().unwrap(), // TODO handle error properly
        )))
    }
}

#[derive(Debug)]
pub struct Probabilistic(pub Array2<f64>);

impl Probabilistic {
    pub fn to_discrete(self) -> Discrete {
        self.try_into().unwrap()
    }
}

impl TryInto<Discrete> for Probabilistic {
    type Error = MinMaxError;

    fn try_into(self) -> Result<Discrete, Self::Error> {
        Ok(Discrete(self.0.map_axis(
            Axis(1),
            |row| row.argmax().unwrap(), // TODO handle error properly
        )))
    }
}

#[derive(Debug)]
pub struct Discrete(pub Array1<usize>);

impl Discrete {
    pub fn from(data: &Data) -> Self {
        let target = data.targets().to_owned();

        Discrete(target)
    }

    pub fn from_prediction(target: Array1<usize>) -> Self {
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
