use crate::dot::Dot;
use crate::sample::Sample;
use crate::NotTrainedError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

#[pyclass]
#[derive(Debug)]
pub struct Perceptron {
    learning_rate: f64,
    weights: Option<Vec<f64>>,
    bias: Option<f64>,

    /// used to map the sample's labels to 0 and 1 (e.g. "red" -> 0, "blue" -> 1)
    label_to_num: Option<HashMap<String, i8>>,

    /// used to map 0 and 1 to the sample's labels (e.g. "red" -> 0, "blue" -> 1)
    num_to_label: Option<HashMap<i8, String>>,
}

#[pymethods]
impl Perceptron {
    #[new]
    fn new(learning_rate: f64) -> PyResult<Self> {
        Ok(Self {
            learning_rate,
            weights: None,
            bias: None,
            label_to_num: None,
            num_to_label: None,
        })
    }

    #[getter]
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn get_weights(&self) -> PyResult<Vec<f64>> {
        if let Some(weights) = self.weights.as_ref() {
            return Ok(weights.clone());
        }
        Err(PyErr::new::<NotTrainedError, _>(
            ".train() must be called before 'weights' can be accessed",
        ))
    }

    #[getter]
    fn get_bias(&self) -> PyResult<f64> {
        if let Some(bias) = self.bias {
            return Ok(bias);
        }
        Err(PyErr::new::<NotTrainedError, _>(
            ".train() must be called before 'bias' can be accessed",
        ))
    }

    #[setter]
    fn set_learning_rate(&mut self, value: f64) {
        self.learning_rate = value;
    }

    #[setter]
    fn set_weights(&mut self, value: Vec<f64>) {
        self.weights = Some(value);
    }

    #[setter]
    fn set_bias(&mut self, value: f64) {
        self.bias = Some(value);
    }

    #[args(samples, n_epochs, reinitialize_params = "false")]
    pub fn train(
        &mut self,
        samples: Vec<Sample>,
        n_epochs: usize,
        reinitialize_params: bool,
    ) -> PyResult<()> {
        Self::check_samples_ok(&samples)?;
        if reinitialize_params || self.weights.is_none() || self.bias.is_none() {
            self.initialize_params(&samples);
        }
        let gil = Python::acquire_gil();
        let py = gil.python();
        for _ in 0..n_epochs {
            py.check_signals()?;
            self.train_for_one_epoch(&samples)?;
        }
        Ok(())
    }

    pub fn predict(&self, sample: &Sample) -> PyResult<&String> {
        if self.num_to_label.is_none() {
            return Err(PyErr::new::<NotTrainedError, _>(
                ".train() must be called before predicting",
            ));
        }
        Ok(self
            .num_to_label
            .as_ref()
            .unwrap()
            .get(&self.predict_num(sample)?)
            .unwrap())
    }
}

impl Perceptron {
    fn initialize_params(&mut self, samples: &[Sample]) {
        self.weights = Some(Self::create_weights(&samples));
        self.bias = Some(Self::create_bias(&samples));
        self.label_to_num = Some(Self::create_label_to_num(&samples));
        self.num_to_label = Some(
            self.label_to_num
                .as_ref()
                .unwrap()
                .iter()
                .map(|(k, v)| (*v, k.clone()))
                .collect::<HashMap<i8, String>>(),
        );
    }

    /// Initializes `weights` to a `Vec` of `1.0`s of that matches the length of `samples`.  
    fn create_weights(samples: &[Sample]) -> Vec<f64> {
        vec![1.0; samples[0].get_n_features()]
    }

    /// Initializes bias to 0.0 (`_samples` is not currently used)
    fn create_bias(_samples: &[Sample]) -> f64 {
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
        let shape = samples[0].get_n_features();
        for sample in &samples[1..] {
            if sample.get_n_features() != shape {
                return Err(PyErr::new::<PyValueError, _>(
                    "all tensors in 'samples' must have the same length",
                ));
            }
        }
        Ok(())
    }
}

/// Implements helper function to interact with this perceptron-rs's parameters (`weights`, `bias`)
impl Perceptron {
    fn predict_num(&self, sample: &Sample) -> PyResult<i8> {
        if self.weights.is_none() || self.bias.is_none() {
            return Err(PyErr::new::<NotTrainedError, _>(
                ".train() must be called before predicting",
            ));
        }
        let z = self
            .weights
            .as_ref()
            .unwrap()
            .dot(sample.get_feature_vector_as_ref())
            + self.bias.unwrap();
        if z < 0.0 {
            Ok(0)
        } else {
            Ok(1)
        }
    }

    fn train_for_one_epoch(&mut self, samples: &[Sample]) -> PyResult<()> {
        for sample in samples {
            self.update_params(sample)?;
        }
        Ok(())
    }

    fn update_params(&mut self, sample: &Sample) -> PyResult<()> {
        let weight_change_factor = self.calculate_wcf(sample)?;
        for (weight, &component) in self
            .weights
            .as_mut()
            .unwrap()
            .iter_mut()
            .zip(sample.get_feature_vector_as_ref().iter())
        {
            *weight += weight_change_factor * component;
        }
        *self.bias.as_mut().unwrap() += weight_change_factor;
        Ok(())
    }

    /// Calculate the _weight change factor_ for this `sample`.
    fn calculate_wcf(&self, sample: &Sample) -> PyResult<f64> {
        let prediction = self.predict_num(sample)?;
        let actual = self.label_to_num.as_ref().unwrap()[sample.get_label()];
        let multiplier = (actual - prediction) as f64;
        //  multiplier is 0.0 if prediction is correct
        //               -1.0 if prediction is too big
        //                1.0 if prediction is too small
        Ok(multiplier * self.learning_rate)
    }
}
