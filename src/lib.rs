use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
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

mod perceptron {
    use numpy::ndarray::{s, Array2, ArrayView2, Axis};

    pub fn predict(weights: &ArrayView2<'_, f64>, x: &ArrayView2<'_, f64>) -> Array2<f64> {
        let res1 = &weights.slice(s![0, ..]);
        let _weights = &weights.slice(s![1..;-1, ..]);
        let res2 = x.dot(_weights);
        let res = res1 + res2;
        // Return activation
        res.mapv(|v| if v >= 0.0 { 1.0 } else { 0.0 })
    }

    fn accuracy(error: Array2<f64>) -> Array2<f64> {
        let acc = vec![error.map(|a| a.powi(2)).mean().unwrap()];
        Array2::from_shape_vec((1, 1), acc).unwrap()
    }

    pub fn train(
        x: &ArrayView2<'_, f64>,
        y: &ArrayView2<'_, f64>,
        alpha: f64,
        n_epoch: i64,
    ) -> (Array2<f64>, Array2<f64>) {
        let features = x.dim().1 + 1;
        let x_size = x.dim().0;
        let mut weights: Array2<f64> = Array2::zeros((features, 1));
        let mut error: Array2<f64> = Array2::zeros((x_size, 1));

        for _epoch in 0..n_epoch {
            let res = predict(&weights.view(), &x);
            error = res - y;
            let mut weight_count = 0;
            weights = weights.map(|v| {
                let res = match weight_count {
                    0 => error.map(|e| v + alpha * e).sum(),
                    _ => error
                        .map(|e| {
                            x.map_axis(Axis(0), |xi| v + alpha * e * xi[weight_count])
                                .sum()
                        })
                        .sum(),
                };
                weight_count += 1;
                res
            });
        }
        let acc = accuracy(error);
        (weights, acc)
    }
}
