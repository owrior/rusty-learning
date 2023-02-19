use numpy::ndarray::{s, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::ops::AddAssign;

fn accuracy(error: Array2<f64>) -> f64 {
    1.0 - error.map(|a| a.powi(2)).mean().unwrap()
}

fn unit_step_function(activation: Array2<f64>) -> Array2<f64> {
    activation.mapv(|v| if v >= 0.0 { 1.0 } else { 0.0 })
}

fn _predict(weights: &Array2<f64>, x: &ArrayView2<'_, f64>) -> Array2<f64> {
    let bias = &weights.slice(s![0, ..]);
    let _weights = &weights.slice(s![1.., ..]);
    let activation = x.dot(_weights) + bias;
    unit_step_function(activation)
}

fn _train(
    x: &ArrayView2<'_, f64>,
    y: &ArrayView2<'_, f64>,
    alpha: f64,
    n_epoch: u64,
) -> (Array2<f64>, f64) {
    let features = x.dim().1;
    let x_size = x.dim().0;
    let mut weights: Array2<f64> = Array2::zeros((features + 1, 1));
    let mut error: Array2<f64> = Array2::zeros((x_size, 1));

    for _epoch in 0..n_epoch {
        let y_hat = _predict(&weights, &x);
        error = y - y_hat;
        let update = error.map(|e| e * alpha);

        if update.map(|v| v.powi(2)).sum() == 0.0 {
            break;
        };

        weights.slice_mut(s![0, ..]).add_assign(update.sum());

        let update = update
            .into_shape((1, x_size))
            .unwrap()
            .dot(x)
            .into_shape((features, 1))
            .unwrap();

        Zip::from(&mut weights.slice_mut(s![1.., ..]))
            .and(&update)
            .for_each(|w, u| *w = *w + u);
    }
    let acc = accuracy(error);
    (weights, acc)
}

#[pyfunction]
fn train<'py>(
    py: Python<'py>,
    alpha: f64,
    n_epoch: u64,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let x_array = x.as_array();
    let y_array = y.as_array();
    let res = _train(&x_array, &y_array, alpha, n_epoch);
    Ok(res.0.into_pyarray(py))
}

#[pyfunction]
fn predict<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<f64>,
    x: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let weights_array = weights.as_array().to_owned();
    let x_array = x.as_array();
    let res = _predict(&weights_array, &x_array);
    Ok(res.into_pyarray(py))
}

pub fn register_perceptron(_py: Python, rusty_learning: &PyModule) -> PyResult<()> {
    let perceptron = PyModule::new(_py, "perceptron")?;
    perceptron.add_function(wrap_pyfunction!(predict, perceptron)?)?;
    perceptron.add_function(wrap_pyfunction!(train, perceptron)?)?;
    rusty_learning.add_submodule(perceptron)?;
    Ok(())
}
