use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use std::fmt;

#[pyclass]
#[derive(Clone)]
pub struct Sample {
    feature_vector: Vec<f64>,
    label: String,
}

#[pymethods]
impl Sample {
    #[new]
    pub fn new(feature_vector: Vec<f64>, label: &str) -> PyResult<Self> {
        Ok(Self {
            feature_vector,
            label: label.to_string(),
        })
    }

    #[getter]
    pub fn get_feature_vector(&self) -> Vec<f64> {
        self.feature_vector.clone()
    }

    pub fn get_n_features(&self) -> usize {
        self.feature_vector.len()
    }

    #[getter]
    pub fn get_label(&self) -> &str {
        &self.label
    }
}

#[pyproto]
impl PyObjectProtocol for Sample {
    // ^^^^^^^^^^^^^^ if this shows [E0277], it's actually fine
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }
}

impl Sample {
    pub fn get_feature_vector_as_ref(&self) -> &Vec<f64> {
        &self.feature_vector
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Sample")
            .field(&self.feature_vector)
            .field(&self.label)
            .finish()
    }
}

impl fmt::Debug for Sample {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
