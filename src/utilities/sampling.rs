use std::ops::Range;
use num_traits::Num;

use rand::{distributions::uniform::{SampleRange, SampleUniform}, rngs::ThreadRng, Rng};

pub trait ExtendedRng<T, R> where T: SampleUniform, R: SampleRange<T> {
    fn gen_zero_to(&mut self, lim: T) -> T;
    fn gen_range_excluding(&mut self, range: R, exclude: T) -> T;
    fn gen_distinct_pair_range(&mut self, range: R) -> (T, T);
}

impl<N: Num + Clone + Copy + std::cmp::PartialOrd + std::cmp::PartialEq + SampleUniform> ExtendedRng<N, Range<N>> for ThreadRng {
    fn gen_zero_to(&mut self, lim: N) -> N {
        self.gen_range(N::zero()..lim)
    }

    fn gen_range_excluding(&mut self, range: Range<N>, exclude: N) -> N {
        loop {
            let x = self.gen_range(range.clone());
            if x != exclude {
                break x;
            }
        }
    }

    fn gen_distinct_pair_range(&mut self, range: Range<N>) -> (N, N) {
        let x = self.gen_range(range.clone());
        let y = loop {
            let y = self.gen_range(range.clone());
            if y != x {
                break x;
            }
        };

        (x, y)
    }
}
