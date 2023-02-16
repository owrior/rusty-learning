use std::ops::AddAssign;

use numpy::ndarray::{s, Array2, ArrayView2};

fn accuracy(error: Array2<f64>) -> Array2<f64> {
    let acc = vec![1.0 - error.map(|a| a.powi(2)).mean().unwrap()];
    Array2::from_shape_vec((1, 1), acc).unwrap()
}

fn unit_step_function(activation: Array2<f64>) -> Array2<f64> {
    activation.mapv(|v| if v >= 0.0 { 1.0 } else { 0.0 })
}

pub fn predict(weights: &ArrayView2<'_, f64>, x: &ArrayView2<'_, f64>) -> Array2<f64> {
    let bias = &weights.slice(s![0, ..]);
    let _weights = &weights.slice(s![1.., ..]);
    let activation = x.dot(_weights) + bias;
    unit_step_function(activation)
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
        let y_hat = predict(&weights.view(), &x);
        error = y - y_hat;

        for it in x.outer_iter().zip(error.outer_iter()) {
            let (xi, e) = it;
            let update = e[[0]] * alpha;
            let mut weight_count = 0;
            for mut w in weights.outer_iter_mut() {
                if weight_count == 0 {
                    w.slice_mut(s![0]).add_assign(update);
                } else {
                    w.slice_mut(s![0])
                        .add_assign(xi[[weight_count - 1]] * update);
                }
                weight_count += 1;
            }
        }
    }
    let acc = accuracy(error);
    (weights, acc)
}
