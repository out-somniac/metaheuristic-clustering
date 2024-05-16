use std::collections::HashMap;

use linfa::dataset::Records;
use ndarray::{Array1, Array2, ArrayBase, Axis, OwnedRepr};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::{errors::{MinMaxError, MultiInputError}, DeviationExt, QuantileExt};
use itertools::Itertools;
use pathfinding::prelude::{kuhn_munkres, Matrix};

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

    pub fn l2_distance(a: &Fuzzy, b: &Fuzzy) -> Result<f64, MultiInputError> {
        a.distribution.l2_dist(&b.distribution)
    }

    pub fn cosine_distance(a: &Fuzzy, b: &Fuzzy) -> Result<f64, MultiInputError> {
        let a = &a.distribution;
        let b = &b.distribution;

        let a_norm = a.sq_l2_dist(a)?;
        let b_norm = b.sq_l2_dist(b)?;
        let dist = a.sq_l2_dist(b)?;

        Ok(dist / (a_norm * b_norm))
    }

    pub fn fitness(&self, data: &Data) -> f64 {
        let samples = data.records();
        let n_cols = samples.ncols();

        let indicator = self
            .clone()
            .to_discrete()
            .to_vec();

        let clusters = indicator
            .iter()
            .zip(samples.axis_iter(Axis(0)))
            .into_group_map();

        let centroids: HashMap<_, _> = clusters
            .iter()
            .map(|(&cluster_num, records)| {
                let n_samples = records.len() as f64;
                let centroid = records.iter().fold(
                    Array1::zeros(n_cols),
                    |acc: ArrayBase<OwnedRepr<f64>, _>, record| acc + record
                ) / n_samples;

                (cluster_num, centroid)
            })
            .collect();

        let variance = centroids
            .iter()
            .zip(clusters)
            .map(|((_, centroid), (_, records))| records
                .iter()
                .map(|record| (record - centroid)
                    .mapv_into(|x| x.powf(2.0))
                    .sum()
                    .sqrt()
                )
                .sum::<f64>()
            )
            .sum::<f64>();

        1.0 / variance
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct Discrete {
    pub indicators: Array1<usize>,
    pub n_classes: usize,
    pub n_samples: usize
}

impl Discrete {
    pub fn new(data: &Data) -> Self {
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

    pub fn from_prediction(pred: Array1<usize>, n_classes: usize) -> Self {
        let n_samples = pred.shape()[0];
        Discrete { indicators: pred, n_classes, n_samples }
    }

    pub fn to_vec(self) -> Vec<usize> {
        self.indicators.to_vec()
    }

    pub fn matched_with(self, truth: &Discrete) -> Result<Discrete, ndarray::ErrorKind> {
        let Discrete { mut indicators, n_classes, n_samples } = self;
        
        if n_samples != truth.indicators.dim() {
            return Err(ndarray::ErrorKind::IncompatibleShape);
        }

        let mut cost = Matrix::new(n_classes, n_classes, 0isize);

        truth
            .indicators
            .iter()
            .zip(indicators.iter())
            .for_each(|(&t, &p)| unsafe {
                *cost.get_unchecked_mut(t * n_classes + p) += 1
            });

        let (_, mapping) = kuhn_munkres(&cost);

        indicators.mapv_inplace(|x| mapping[x]);

        Ok(Discrete { indicators, n_classes, n_samples })
    }
}

impl Into<Vec<usize>> for Discrete {
    fn into(self) -> Vec<usize> {
        self.to_vec()
    }
}
