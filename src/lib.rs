use numpy::ndarray::s;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyTuple};

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
        weights: PyReadonlyArray1<f64>,
        x: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let weights_array = weights.to_owned_array();
        let x_array = x.as_array();

        let bias = weights_array[[0]];
        let _weights = weights_array.slice_move(s![1..]);

        let res = perceptron::predict(&bias, &_weights, &x_array);
        res.into_pyarray(py)
    }
    Ok(())
}

mod perceptron {
    use std::ops::AddAssign;

    use numpy::ndarray::{arr1, s, Array1, Array2, ArrayView2, Axis};

    pub fn predict(bias: &f64, _weights: &Array1<f64>, x: &ArrayView2<'_, f64>) -> Array2<f64> {
        let features = x.dim().1;
        let weights = _weights.slice(s![..]).into_shape((features, 1)).unwrap();
        let activation = x.dot(&weights).mapv(|v| v + bias);
        // Return activation
        step_function(activation)
    }

    fn step_function(activation: Array2<f64>) -> Array2<f64> {
        activation.mapv(|v| if v >= 0.0 { 1.0 } else { 0.0 })
    }

    fn accuracy(error: Array2<f64>) -> Array1<f64> {
        let acc = vec![1.0 - error.map(|a| a.powi(2)).mean().unwrap()];
        Array1::from_vec(acc)
    }

    pub fn train(
        x: &ArrayView2<'_, f64>,
        y: &ArrayView2<'_, f64>,
        alpha: f64,
        n_epoch: i64,
    ) -> (Array1<f64>, Array1<f64>) {
        let features = x.dim().1;
        let x_size = x.dim().0;
        let mut bias: f64 = 0.0;
        let mut weights: Array1<f64> = Array1::zeros(features);
        let mut error: Array2<f64> = Array2::zeros((x_size, 1));

        for _epoch in 0..n_epoch {
            let y_hat = predict(&bias, &weights, &x);
            error = y - y_hat;

            for it in x.outer_iter().zip(error.outer_iter()) {
                let (xi, e) = it;
                let update = e[[0]] * alpha;

                bias += update;
                weights.add_assign(&xi.map(|v| v * update));
            }
        }
        let acc = accuracy(error);
        let mut weights_res = arr1(&[bias]);
        weights_res.append(Axis(0), weights.view()).unwrap();
        (weights_res, acc)
    }
}
