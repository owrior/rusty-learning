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
        let weights_array = weights.as_array();
        let x_array = x.as_array();
        let res = perceptron::predict(&weights_array, &x_array);
        res.into_pyarray(py)
    }
    Ok(())
}

mod perceptron {
    use std::ops::AddAssign;

    use numpy::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

    pub fn predict(weights: &ArrayView1<'_, f64>, x: &ArrayView2<'_, f64>) -> Array2<f64> {
        let features = x.dim().1 + 1;

        let bias = &weights.slice(s![0]);
        let _weights = &weights
            .slice(s![1..])
            .into_shape((features - 1, 1))
            .unwrap();
        let activation = x.dot(_weights) + bias;
        // Return activation
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
        let features = x.dim().1 + 1;
        let x_size = x.dim().0;
        let mut weights: Array1<f64> = Array1::zeros(features);
        let mut error: Array2<f64> = Array2::zeros((x_size, 1));

        for _epoch in 0..n_epoch {
            let y_hat = predict(&weights.view(), &x);
            error = y - y_hat;

            for it in x.outer_iter().zip(error.outer_iter()) {
                let (xi, e) = it;
                let update = e[[0]] * alpha;
                let mut weight_count = 0;
                for w in weights.iter_mut() {
                    if weight_count == 0 {
                        w.add_assign(update);
                    } else {
                        w.add_assign(xi[[weight_count - 1]] * update);
                    }
                    weight_count += 1;
                }
            }
        }
        let acc = accuracy(error);
        (weights, acc)
    }
}
