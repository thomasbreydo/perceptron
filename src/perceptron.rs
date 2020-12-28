use crate::sample::Sample;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use std::collections::HashSet;
use std::fmt;

#[pyclass]
pub struct Perceptron {
    samples: Vec<Sample>,
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

#[pymethods]
impl Perceptron {
    #[new]
    fn new(samples: Vec<Sample>, learning_rate: f64) -> PyResult<Self> {
        Self::check_samples_ok(&samples)?;
        Ok(Self {
            samples: samples.clone(),
            weights: Self::create_weights(&samples),
            bias: Self::create_bias(),
            learning_rate,
        })
    }

    #[getter]
    fn get_samples(&self) -> Vec<Sample> {
        self.samples.clone()
    }

    #[getter]
    fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    #[getter]
    fn get_bias(&self) -> f64 {
        self.bias
    }

    #[getter]
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

#[pyproto]
impl PyObjectProtocol for Perceptron {
    // ^^^^^^^^^^^^^^ if this shows [E0277], it's actually fine
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }
}

impl Perceptron {
    fn create_weights(samples: &[Sample]) -> Vec<f64> {
        vec![1.0; samples[0].get_tensor().len()]
    }

    fn create_bias() -> f64 {
        0.0
    }

    /// Returns `Err(PyValueError)` if `samples` is invalid; otherwise returns `Ok(())`.
    ///
    /// Note, `samples` is invalid if the samples:
    ///
    ///     - are labeled with one of _three_ or more unique labels (_two_ or fewer is valid)
    ///
    ///     - have tensors of differing length
    fn check_samples_ok(samples: &[Sample]) -> PyResult<()> {
        Self::check_labels_ok(samples)?;
        Self::check_tensors_ok(samples)?;
        Ok(())
    }

    /// Returns `Err(PyValueError)` if the choice for labels in `samples` is not limited to two
    /// labels; otherwise returns `Ok(())`.
    fn check_labels_ok(samples: &[Sample]) -> PyResult<()> {
        let mut labels = HashSet::new();
        for sample in samples {
            labels.insert(sample.get_label());
            if labels.len() > 2 {
                return Err(PyErr::new::<PyValueError, _>(
                    "choice for labels in 'samples' must be limited to two labels",
                ));
            }
        }
        Ok(())
    }

    /// Returns `Err(PyValueError)` if the samples in `samples` have tensors of differing length;
    /// otherwise returns `Ok(())`.
    fn check_tensors_ok(samples: &[Sample]) -> PyResult<()> {
        let shape = samples[0].get_tensor().len();
        for sample in samples[1..].iter() {
            if sample.get_tensor().len() != shape {
                return Err(PyErr::new::<PyValueError, _>(
                    "all tensors in 'samples' must have the same length",
                ));
            }
        }
        Ok(())
    }
}

impl fmt::Display for Perceptron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Perceptron")
            .field(&self.samples)
            .field(&self.weights)
            .field(&self.bias)
            .field(&self.learning_rate)
            .finish()
    }
}

impl fmt::Debug for Perceptron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
