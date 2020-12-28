use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use std::fmt;

#[pyclass]
pub struct Sample {
    coords: Vec<f64>,
    label: String,
}

#[pymethods]
impl Sample {
    #[new]
    pub fn new(coords: Vec<f64>, classification: &str) -> PyResult<Self> {
        Ok(Self {
            coords,
            label: classification.to_string(),
        })
    }
}

#[pyproto]
impl PyObjectProtocol for Sample {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Sample")
            .field(&self.coords)
            .field(&self.label)
            .finish()
    }
}
