use std::collections::HashMap;

use linfa::dataset::Records;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, OwnedRepr};
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

        let indicator = self
            .clone()
            .to_discrete()
            .to_vec();

        let clusters = indicator
            .iter()
            .zip(samples.axis_iter(Axis(0)))
            .into_group_map();

        
        // let mut fitn: f64 = 0.0;
        // for (idx, cluster) in clusters {
        //     for elem in cluster {
        //         fitn -= *idx as f64;
        //         // println!("{}", elem);
        //         // fitn += unsafe {elem.uget(0)};
        //         // fitn -= unsafe {elem.uget(1)};
        //         // fitn -= unsafe {elem.uget(2)};
        //     }
        // };

        // println!("{}", fitn);
        // return fitn;


        // let clusters_idx = indicator
        //     .iter()
        //     .zip(0..samples.len())
        //     .into_group_map();

        // https://www.researchgate.net/publication/341593540_Genetic_Algorithm_with_New_Fitness_Function_for_Clustering
        let k = clusters.len();

        // BC - Distance between clusters

        let mut BC: f64 = 0.0;

        for m in 0..k-1 {
            for n in m+1..k {
                let mut BC_nm2: f64 = 0.0;

                let cluster_a = clusters.get(&m).unwrap();
                let cluster_b = clusters.get(&n).unwrap();

                let pomocy = cluster_a.iter().cartesian_product(cluster_b);

                for (u, v) in pomocy {
                    let diff = u-v;
                    let skull_emoji = (diff.clone() * diff).sum();
                    BC_nm2 += skull_emoji;
                }

                BC_nm2 /= (cluster_a.len() * cluster_b.len() * n) as f64;
                BC += BC_nm2.sqrt();
            }
        };

        // WC - Distance within a cluster

        let mut WC: f64 = 0.0;

        for m in 0..k-1 {
            let mut WC_m2: f64 = 0.0;
            let cluster = clusters.get(&m).unwrap();
            let pomocy = cluster.iter().cartesian_product(cluster);

            for (u, v) in pomocy {
                let diff = u-v;
                let skull_emoji = (diff.clone() * diff).sum();
                WC_m2 += skull_emoji;
            }

            WC_m2 /= (cluster.len() * cluster.len()) as f64;
            WC += WC_m2.sqrt();
        };

        // SW - Average of the silhouette value of each observation

        // Porzuccie nadzieje ci co tutaj przychodzicie
        // :(

        let mut SW: f64 = 0.0;

        let observation_to_cluster = self
            .clone()
            .to_discrete()
            .to_vec();

        for (obs_id, cluster_id) in observation_to_cluster.iter().enumerate() {
            let current_obs_cluster = clusters.get(cluster_id).unwrap();

            // NIE TRZEBA TEGO FILTROWAC PONIEWAZ a - a == 0 :DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD 
            // let others_in_curr_cluster = current_obs_cluster_idx.iter().filter(|x| **x != obs_id);

            let wektorus = samples.slice(s![obs_id, ..]).clone();

            // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            let mut ai: f64 = 0.0;
            for other in current_obs_cluster {
                let diff = &wektorus - other;
                let dist = (diff.clone() * diff).sum();
                ai += dist;
            }
            ai /= current_obs_cluster.len() as f64;

            // BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
            let mut bi: f64 = 99999999999999999999.0;

            for (idx, cluster) in &clusters {
                if *idx == cluster_id {
                    continue;
                }

                let mut curr_bi = 0.0;
                for other in cluster {
                    let diff = &wektorus - other;
                    let dist = (diff.clone() * diff).sum();
                    curr_bi += dist;
                }
                curr_bi /= current_obs_cluster.len() as f64;

                bi = bi.min(curr_bi);
            }

            SW += (bi-ai)/bi.min(ai);
        };

        // FF

        println!("FITNESS: {}", (BC/WC)+SW);

        (BC/WC)+SW

        // let centroids: HashMap<_, _> = clusters
        //     .iter()
        //     .map(|(&cluster_num, records)| {
        //         let n_samples = records.len() as f64;
        //         let centroid = records.iter().fold(
        //             Array1::zeros(n_cols),
        //             |acc: ArrayBase<OwnedRepr<f64>, _>, record| acc + record
        //         ) / n_samples;

        //         (cluster_num, centroid)
        //     })
        //     .collect();

        // let variance = centroids
        //     .iter()
        //     .zip(clusters)
        //     .map(|((_, centroid), (_, records))| records
        //         .iter()
        //         .map(|record| (record - centroid)
        //             .mapv_into(|x| x.powf(2.0))
        //             .sum()
        //             .sqrt()
        //         )
        //         .sum::<f64>()
        //     )
        //     .sum::<f64>();

        // 1.0 / variance
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
    pub n_classes: usize,
    pub n_samples: usize
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

    pub fn from_prediction(pred: Array1<usize>, n_classes: usize) -> Self {
        let n_samples = pred.shape()[0];
        Discrete { indicators: pred, n_classes, n_samples }
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
