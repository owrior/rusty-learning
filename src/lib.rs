mod algo;

use algo::perceptron::register_perceptron;
use pyo3::prelude::*;

#[pymodule]
fn rusty_learning(_py: Python, m: &PyModule) -> PyResult<()> {
    register_perceptron(_py, m)?;
    Ok(())
}
