use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyTuple};

#[pymodule]
fn rusty_learning(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn train<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        alpha: f64,
        n_epoch: i64,
    ) -> &'py PyTuple {
        let x_array = x.as_array();
        let y_array = y.as_array();

        let res = perceptron::train(&x_array, &y_array, alpha, n_epoch);

        PyTuple::new(py, vec![res.0.to_object(py), res.1.to_object(py)])
    }

    #[pyfn(m)]
    fn predict<'py>(
        py: Python<'py>,
        weights: PyReadonlyArray1<f64>,
        x: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        let weights_array = weights.as_array();
        let x_array = x.as_array();
        let res = perceptron::predict(&weights_array, &x_array);
        res.into_pyarray(py)
    }
    Ok(())
}

mod perceptron {
    use numpy::ndarray::{arr1, s, Array1, ArrayView1, ArrayView2};

    pub fn predict(weights: &ArrayView1<'_, f64>, x: &ArrayView2<'_, f64>) -> Array1<f64> {
        let res1 = &weights.slice(s![0]);
        let _weights = &weights.slice(s![1..;-1]);
        let res2 = x.dot(_weights);
        let res = res1 + res2;

        // Return activation
        res.mapv(|v| if v >= 0.0 { 1.0 } else { 0.0 })
    }

    pub fn train(
        x: &ArrayView2<'_, f64>,
        y: &ArrayView1<'_, f64>,
        alpha: f64,
        n_epoch: i64,
    ) -> (Vec<f64>, f64) {
        let features = x.dim().1 + 1;
        let x_size = x.dim().0 as f64;
        let mut weights: Vec<f64> = vec![0.0; features];
        let mut error: Vec<f64>;
        let mut sq_error = 0.0;

        for _epoch in 0..n_epoch {
            let res = predict(&arr1(&weights).view(), &x);
            error = (res - y).into_raw_vec();
            sq_error = error.iter().map(|a| a.powi(2)).sum();
            weights = weights
                .into_iter()
                .map(|v| error.iter().map(|e| v + alpha * e).sum())
                .collect();
        }
        (weights, sq_error / x_size)
    }
}
