mod perceptron;
mod sample;

use perceptron::Perceptron;
use pyo3::prelude::*;
use sample::Sample;

/// A Python module implemented in Rust.
#[pymodule]
fn perceptron(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Sample>()?;
    m.add_class::<Perceptron>()?;
    Ok(())
}
