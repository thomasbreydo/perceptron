pub mod dot;
mod exceptions;
mod perceptron;
mod sample;

pub use exceptions::NotTrainedError;
pub use perceptron::Perceptron;
use pyo3::prelude::*;
pub use sample::Sample;

/// A Python module implemented in Rust.
#[pymodule]
fn perceptron_rs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Sample>()?;
    m.add_class::<Perceptron>()?;
    m.add("NotTrainedError", py.get_type::<NotTrainedError>())?;
    Ok(())
}
