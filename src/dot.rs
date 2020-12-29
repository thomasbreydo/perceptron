use std::iter::Sum;
use std::ops::Mul;

pub trait Dot<T>
where
    T: Mul + Sum<<T as Mul>::Output> + Copy,
{
    fn dot(&self, other: &[T]) -> T;
}

impl<T> Dot<T> for std::vec::Vec<T>
where
    T: Mul + Sum<<T as Mul>::Output> + Copy,
{
    fn dot(&self, other: &[T]) -> T {
        self.iter()
            .zip(other.iter())
            .map(|(&weight, &component)| weight * component)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let vec1: Vec<f64> = vec![0.1, 3.1, 4.1];
        let vec2: Vec<f64> = vec![-1.2, 4.0, 2.3];
        approx::assert_relative_eq!(vec1.dot(&vec2), 21.71);
    }
}
