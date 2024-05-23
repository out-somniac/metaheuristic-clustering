use std::collections::HashMap;

use itertools::Itertools;
use ndarray::{s, Array2, ArrayView1, Axis};
use ndarray_stats::DeviationExt;

use crate::Data;
use super::solution::Fuzzy;


#[inline(always)]
fn external_deviation(clusters: &HashMap<&usize, Vec<ArrayView1<f64>>>) -> f64 {
    let mut result = 0.0;
    let k = clusters.len();

    for m in 0..k - 1 {
        for n in m + 1..k {
            let cluster_a = &clusters[&m];
            let cluster_b = &clusters[&n];
            let denominator = cluster_a.len() * cluster_b.len() * n;

            let total_l2_dist = cluster_a
                .iter()
                .cartesian_product(cluster_b)
                .map(|(u, v)| u.l2_dist(&v).unwrap())
                .sum::<f64>();

            let avg_l2_dist = total_l2_dist / denominator as f64;

            result += avg_l2_dist.sqrt();
        }
    };

    result
}

#[inline(always)]
fn internal_deviation(clusters: &HashMap<&usize, Vec<ArrayView1<f64>>>) -> f64 {
    let mut result = 0.0;
    let k = clusters.len();

    for m in 0..k - 1 {
        let cluster = &clusters[&m];
        let n = cluster.len() as f64;

        let total_l2_dist = cluster
            .iter()
            .cartesian_product(cluster)
            .map(|(u, v)| u.l2_dist(&v).unwrap())
            .sum::<f64>();

        let avg_l2_dist = total_l2_dist / (n * n);

        result += avg_l2_dist.sqrt();
    };

    result
}

#[inline(always)]
fn average_silhouette(
    samples: &Array2<f64>,
    indicator: &Vec<usize>,
    clusters: &HashMap<&usize, Vec<ArrayView1<f64>>>
) -> f64 {
    let mut result = 0.0;

    for (i, cluster_id) in indicator.iter().enumerate() {
        let point_i = samples.slice(s![i, ..]);

        let cluster = &clusters[cluster_id];
        let n = cluster.len() as f64;

        let a_i = cluster
            .iter()
            .map(|point| point_i.l2_dist(point).unwrap())
            .sum::<f64>();

        let a_i = a_i / n;

        let mut b_i = f64::INFINITY;

        for (&&j, other_cluster) in clusters {
            if j == *cluster_id {
                continue;
            }

            let n = other_cluster.len() as f64;

            let total_b_j = other_cluster
                .iter()
                .map(|point| point_i.l2_dist(point).unwrap())
                .sum::<f64>();

            let b_j = total_b_j / n;

            b_i = b_i.min(b_j);
        }

        result += (b_i - a_i) / b_i.min(a_i);
    };

    result
}

pub fn silhouette_similarity(data: &Data, solution: &Fuzzy) -> f64 {
    let samples = data.records();

    let indicator = solution
        .clone()
        .to_discrete()
        .to_vec();

    let clusters = indicator
        .iter()
        .zip(samples.axis_iter(Axis(0)))
        .into_group_map();

    let deviation_ratio = external_deviation(&clusters) / internal_deviation(&clusters);
    let silhouette = average_silhouette(samples, &indicator, &clusters);

    deviation_ratio + silhouette
}
