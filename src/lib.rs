use ndarray::{self, ArrayBase, Dim, OwnedRepr};
use linfa::DatasetBase;

pub type Data = DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>>;

pub mod cluster_map;
pub mod scatter_matrix;
