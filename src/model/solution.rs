use linfa::dataset::{Labels, Records};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::{errors::MinMaxError, QuantileExt};
use itertools::Itertools;

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
        let samples = data.records();
        
        let n_cols = samples.ncols();
        let n_classes = self.n_classes;

        let indicator = self
            .clone()
            .to_discrete()
            .to_vec();

        // Initialize space for centroid vectors
        let mut centroids = Array2::<f64>::zeros((n_classes, n_cols));
        let mut counts = vec![0usize; n_classes];

        // Compute record sums for each cluster
        for (record, &cluster) in indicator.iter().enumerate() {
            for col in 0..n_cols {
                unsafe {
                    *centroids.uget_mut((cluster, col)) += *samples.uget((record, col));
                }
            }

            counts[cluster] += 1;
        }

        // Compute centroids (arithmetical means)
        for (class, mut col) in centroids.axis_iter_mut(Axis(1)).enumerate() {
            col.mapv_inplace(|x| x / counts[class] as f64);
        }

        let mut variances = vec![0f64; n_classes];

        // Compute variance for each cluster
        for (record, &cluster) in indicator.iter().enumerate() {
            let mut distance = 0.0;

            for col in 0..n_cols {
                unsafe {
                    distance += (
                        *samples.uget((record, col)) -
                        *centroids.uget((cluster, col))
                    ).powf(2.0);
                }
            }

            variances[cluster] += distance.sqrt();
        }

        variances
            .iter()
            .sum()
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

    fn try_into(self) -> Result<Probabilistic, MinMaxError> {
        let Fuzzy { mut distribution, n_classes, n_samples } = self;

        for mut row in distribution.axis_iter_mut(Axis(0)) {
            let max = *row.max()?;
            row.mapv_inplace(|x| f64::exp(x - max));

            let sum = row.sum();
            row.mapv_inplace(|x| x / sum);
        }

        Ok(Probabilistic { distribution, n_classes, n_samples })
    }
}

impl TryInto<Discrete> for Fuzzy {
    type Error = MinMaxError;

    fn try_into(self) -> Result<Discrete, Self::Error> {
        let Fuzzy { distribution, n_classes, n_samples } = self;

        let indicators = distribution.map_axis(
            Axis(1),
            |row| row.argmax().unwrap(), // TODO handle error properly
        );

        Ok(Discrete { indicators, n_classes, n_samples })
    }
}

#[derive(Debug)]
pub struct Probabilistic {
    pub distribution: Array2<f64>,
    pub n_samples: usize,
    pub n_classes: usize,
}

impl Probabilistic {
    pub fn to_discrete(self) -> Discrete {
        self.try_into().unwrap()
    }
}

impl TryInto<Discrete> for Probabilistic {
    type Error = MinMaxError;

    fn try_into(self) -> Result<Discrete, Self::Error> {
        let Probabilistic { distribution, n_classes, n_samples } = self;

        let indicators = distribution.map_axis(
            Axis(1),
            |row| row.argmax().unwrap(), // TODO handle error properly
        );

        Ok(Discrete { indicators, n_classes, n_samples })
    }
}

#[derive(Debug)]
pub struct Discrete {
    pub indicators: Array1<usize>,
    n_classes: usize,
    n_samples: usize
}

impl Discrete {
    pub fn from(data: &Data) -> Self {
        let indicators = data.targets().to_owned();
        let n_samples = data.nsamples();
        let n_classes = data
            .targets()
            .iter()
            .unique()
            .collect::<Vec<_>>()
            .len();

        Discrete { indicators, n_classes, n_samples }
    }

    pub fn to_vec(self) -> Vec<usize> {
        self.indicators.to_vec()
    }
}

impl Into<Vec<usize>> for Discrete {
    fn into(self) -> Vec<usize> {
        self.to_vec()
    }
}
