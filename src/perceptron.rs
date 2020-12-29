use crate::sample::Sample;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt;

#[pyclass]
pub struct Perceptron {
    samples: Vec<Sample>,
    learning_rate: f64,
    weights: Vec<f64>, // TODO: make options
    bias: f64,

    /// used to map the sample's labels to 0 and 1 (e.g. "red" -> 0, "blue" -> 1)
    label_to_num: HashMap<String, i8>,

    /// used to map 0 and 1 to the sample's labels (e.g. "red" -> 0, "blue" -> 1)
    num_to_label: HashMap<i8, String>,
}

#[pymethods]
impl Perceptron {
    #[new]
    fn new(samples: Vec<Sample>, learning_rate: f64) -> PyResult<Self> {
        Self::check_samples_ok(&samples)?;
        let weights = Self::create_weights(&samples);
        let bias = Self::create_bias();
        let label_to_num = Self::create_label_to_num(&samples);
        let num_to_label = label_to_num
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect::<HashMap<i8, String>>();
        Ok(Self {
            samples,
            learning_rate,
            weights,
            bias,
            label_to_num,
            num_to_label,
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

    #[setter]
    fn set_samples(&mut self, value: Vec<Sample>) {
        self.samples = value;
    }

    #[setter]
    fn set_weights(&mut self, value: Vec<f64>) {
        self.weights = value;
    }

    #[setter]
    fn set_bias(&mut self, value: f64) {
        self.bias = value;
    }

    #[setter]
    fn set_learning_rate(&mut self, value: f64) {
        self.learning_rate = value;
    }

    pub fn train(&mut self, n_epochs: usize) {
        for _ in 0..n_epochs {
            self.train_for_one_epoch();
        }
    }

    pub fn predict(&self, sample: &Sample) -> String {
        self.num_to_label
            .get(&self.predict_num(sample))
            .unwrap()
            .clone()
    }
}

impl Perceptron {
    fn create_weights(samples: &[Sample]) -> Vec<f64> {
        vec![1.0; samples[0].get_tensor().len()]
    }

    fn create_bias() -> f64 {
        0.0
    }

    fn create_label_to_num(samples: &[Sample]) -> HashMap<String, i8> {
        let mut map = HashMap::new();
        for sample in samples {
            let label = sample.get_label();
            if map.contains_key(label) {
                continue;
            }
            map.insert(label.to_string(), map.len() as i8);
            if map.len() == 2 {
                break;
            }
        }
        map
    }

    /// Returns `Err(PyValueError)` if `samples` is invalid; otherwise returns `Ok(())`.
    ///
    /// Note, `samples` is invalid if:
    ///
    ///     - there are not exactly two values of `label` across all samples
    ///
    ///     - the samples have tensors of differing length
    fn check_samples_ok(samples: &[Sample]) -> PyResult<()> {
        Self::check_labels_ok(samples)?;
        Self::check_tensors_ok(samples)?;
        Ok(())
    }

    /// Returns `Err(PyValueError)` if there are not exactly two values of `label` across all
    /// samples in `samples`; otherwise returns `Ok(())`.
    fn check_labels_ok(samples: &[Sample]) -> PyResult<()> {
        let mut labels = HashSet::new();
        for sample in samples {
            labels.insert(sample.get_label());
        }
        if labels.len() == 2 {
            return Ok(());
        }
        Err(PyErr::new::<PyValueError, _>(
            "there must be exactly two values of 'label' across all samples",
        ))
    }

    /// Returns `Err(PyValueError)` if the samples in `samples` have tensors of differing length;
    /// otherwise returns `Ok(())`.
    fn check_tensors_ok(samples: &[Sample]) -> PyResult<()> {
        let shape = samples[0].get_tensor().len();
        for sample in &samples[1..] {
            if sample.get_tensor().len() != shape {
                return Err(PyErr::new::<PyValueError, _>(
                    "all tensors in 'samples' must have the same length",
                ));
            }
        }
        Ok(())
    }
}

/// Implements helper function to interact with this perceptron's parameters (`weights`, `bias`)
impl Perceptron {
    fn predict_num(&self, sample: &Sample) -> i8 {
        let z = self.dot_weights_with(sample) + self.bias;
        if z < 0.0 {
            0
        } else {
            1
        }
    }
    fn train_for_one_epoch(&mut self) {
        for sample in &self.samples.clone() {
            self.update_params(sample);
        }
    }

    fn update_params(&mut self, sample: &Sample) {
        let weight_change_factor = self.get_weight_change_factor(sample);
        for (weight, &component) in self.weights.iter_mut().zip(sample.get_tensor().iter()) {
            *weight += weight_change_factor * component;
        }
        self.bias += weight_change_factor;
    }

    fn get_weight_change_factor(&self, sample: &Sample) -> f64 {
        let prediction = self.predict_num(sample);
        let actual = self.label_to_num[sample.get_label()];
        let multiplier = (actual - prediction) as f64;
        //  multiplier is 0.0 if prediction is correct
        //               -1.0 if prediction is too big
        //                1.0 if prediction is too small
        multiplier * self.learning_rate
    }

    fn dot_weights_with(&self, sample: &Sample) -> f64 {
        self.weights
            .iter()
            .zip(sample.get_tensor().iter())
            .map(|(&weight, &component)| weight * component)
            .sum()
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
