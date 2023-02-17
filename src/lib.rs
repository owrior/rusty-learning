use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

mod algo;
use algo::*;

#[pyclass]
struct Perceptron {
    #[pyo3(get)]
    alpha: f64,
    #[pyo3(get)]
    n_epoch: u64,
    weights: Array2<f64>,
}

#[pymethods]
impl Perceptron {
    #[new]
    pub fn new(alpha: f64, n_epoch: u64, features: usize) -> Self {
        let weights: Array2<f64> = Array2::zeros((features + 1, 1));
        Perceptron {
            alpha,
            n_epoch,
            weights,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Alpha({}), N_Epoch({}), Weights({})",
            self.alpha,
            self.n_epoch,
            self.weights.t()
        )
    }

    pub fn train<'py>(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
    ) -> PyResult<f64> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let res = perceptron::train(&x_array, &y_array, self.alpha, self.n_epoch);
        self.weights = res.0;
        Ok(res.1)
    }

    pub fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
        let x_array = x.as_array();
        let res = perceptron::predict(&self.weights, &x_array);
        res.into_pyarray(py)
    }

    pub fn set_weights(&mut self, weights: PyReadonlyArray2<f64>) -> () {
        self.weights = weights.as_array().to_owned();
    }

    pub fn get_weights<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.weights.clone().into_pyarray(py)
    }
}

#[pymodule]
fn rusty_learning(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Perceptron>()?;
    Ok(())
}
