use std::ops::AddAssign;

use numpy::ndarray::{s, Array2, ArrayView2, Zip};

fn accuracy(error: Array2<f64>) -> f64 {
    1.0 - error.map(|a| a.powi(2)).mean().unwrap()
}

fn unit_step_function(activation: Array2<f64>) -> Array2<f64> {
    activation.mapv(|v| if v >= 0.0 { 1.0 } else { 0.0 })
}

pub fn predict(weights: &Array2<f64>, x: &ArrayView2<'_, f64>) -> Array2<f64> {
    let bias = &weights.slice(s![0, ..]);
    let _weights = &weights.slice(s![1.., ..]);
    let activation = x.dot(_weights) + bias;
    unit_step_function(activation)
}

pub fn train(
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
        let y_hat = predict(&weights, &x);
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
