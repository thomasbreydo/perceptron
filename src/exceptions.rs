pub use pyo3::create_exception;
use pyo3::exceptions::PyException;

create_exception!(perceptron, NotTrainedError, PyException);
