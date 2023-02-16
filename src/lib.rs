use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyTuple};

mod algo;
use algo::*;

#[pymodule]
fn rusty_learning(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn train<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        alpha: f64,
        n_epoch: i64,
    ) -> &'py PyTuple {
        let x_array = x.as_array();
        let y_array = y.as_array();

        let res = perceptron::train(&x_array, &y_array, alpha, n_epoch);
        PyTuple::new(py, vec![res.0.into_pyarray(py), res.1.into_pyarray(py)])
    }

    #[pyfn(m)]
    fn predict<'py>(
        py: Python<'py>,
        weights: PyReadonlyArray2<f64>,
        x: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let weights_array = weights.as_array();
        let x_array = x.as_array();
        let res = perceptron::predict(&weights_array, &x_array);
        res.into_pyarray(py)
    }
    Ok(())
}
