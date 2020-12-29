mod perceptron;
mod sample;

pub use perceptron::Perceptron;
use pyo3::prelude::*;
pub use sample::Sample;

/// A Python module implemented in Rust.
#[pymodule]
fn perceptron(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Sample>()?;
    m.add_class::<Perceptron>()?;
    Ok(())
}
