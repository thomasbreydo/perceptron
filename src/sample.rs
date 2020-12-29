use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use std::fmt;

#[pyclass]
#[derive(Clone)]
pub struct Sample {
    tensor: Vec<f64>,
    label: String,
}

#[pymethods]
impl Sample {
    #[new]
    pub fn new(tensor: Vec<f64>, label: &str) -> PyResult<Self> {
        Ok(Self {
            tensor,
            label: label.to_string(),
        })
    }

    #[getter]
    pub fn get_tensor(&self) -> Vec<f64> {
        self.tensor.clone()
    }

    pub fn get_tensor_len(&self) -> usize {
        self.tensor.len()
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
    pub fn get_tensor_as_ref(&self) -> &Vec<f64> {
        &self.tensor
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Sample")
            .field(&self.tensor)
            .field(&self.label)
            .finish()
    }
}

impl fmt::Debug for Sample {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
