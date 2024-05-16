use std::cmp::Ordering;
use num_traits::Num;

pub trait Ordered<T> {
    type Index;

    fn min(&self) -> Option<&T>;
    fn max(&self) -> Option<&T>;
    fn min_max(&self) -> Option<(&T, &T)>;
    fn argmin(&self) -> Option<Self::Index>;
    fn argmax(&self) -> Option<Self::Index>;
}

impl <T: Num + Clone + Copy + std::cmp::PartialOrd + std::cmp::PartialEq> Ordered<T> for Vec<T> {
    type Index = usize;

    fn min(&self) -> Option<&T> {
        let mut result = self.get(0)?;
        
        for element in self {
            match element.partial_cmp(result)? {
                Ordering::Less => { result = element; }
                _ => {}
            }
        }

        Some(result)
    }

    fn max(&self) -> Option<&T> {
        let mut result = self.get(0)?;
        
        for element in self {
            match element.partial_cmp(result)? {
                Ordering::Greater => { result = element; }
                _ => {}
            }
        }

        Some(result)
    }

    fn min_max(&self) -> Option<(&T, &T)> {
        let mut minimum = self.get(0)?;
        let mut maximum = self.get(0)?;

        for element in self {
            match element.partial_cmp(minimum)? {
                Ordering::Less => { minimum = element; },
                Ordering::Greater => match element.partial_cmp(maximum)? {
                    Ordering::Greater => { maximum = element; },
                    _ => {}
                },
                _ => {}
            }
        }

        Some((minimum, maximum))
    }

    fn argmin(&self) -> Option<usize> {
        let mut minimum = self.get(0)?;
        let mut index = 0;
        
        for (i, element) in self.iter().enumerate() {
            match element.partial_cmp(minimum)? {
                Ordering::Less => {
                    minimum = element;
                    index = i;
                }
                _ => {}
            }
        }

        Some(index)
    }

    fn argmax(&self) -> Option<usize> {
        let mut maximum = self.get(0)?;
        let mut index = 0;
        
        for (i, element) in self.iter().enumerate() {
            match element.partial_cmp(maximum)? {
                Ordering::Greater => {
                    maximum = element;
                    index = i;
                }
                _ => {}
            }
        }

        Some(index)
    }
}