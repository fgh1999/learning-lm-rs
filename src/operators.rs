use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    let length = indices.size();
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, &b| a.max(b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    debug_assert!(y.shape() == x.shape());
    debug_assert!(x.shape().len() == w.shape().len() + 1 || w.size() == 1 && x.shape().len() == 1);
    debug_assert!(w
        .shape()
        .iter()
        .zip(x.shape().iter().rev())
        .all(|(&w_s, &x_s)| w_s == x_s));

    let chunk_size = w.shape().iter().product();
    let res = x.data().chunks_exact(chunk_size).map(|x_i| {
        let square_sum = x_i.iter().map(|&x_ij| x_ij * x_ij).sum::<f32>();
        let norm = (square_sum / x_i.len() as f32 + epsilon).sqrt();
        let prod = x_i.iter().zip(w.data().iter()).map(|(&x, &w)| x * w);
        prod.map(move |p| p / norm)
    });
    let y_data = unsafe { y.data_mut() };
    y_data
        .chunks_exact_mut(chunk_size)
        .zip(res)
        .for_each(|(y_i, y_i_res)| y_i.iter_mut().zip(y_i_res).for_each(|(y_ij, r)| *y_ij = r));
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    debug_assert!(y.size() == x.size());

    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();

    fn silu(x: &f32) -> f32 {
        let x = *x;
        x / (1.0 + (-x).exp())
    }
    let silu_x = x_data.iter().map(silu);
    y_data.iter_mut().zip(silu_x).for_each(|(y, x)| *y *= x);
}

// C = beta * C + alpha * A @ B^T
// Assume that A is of shape (m, k), B is of shape (n, k), C is of shape (m, n).
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    debug_assert!(c.shape().len() == 2);
    debug_assert!(a.shape().len() == 2);
    debug_assert!(b.shape().len() == 2);
    debug_assert!(a.shape()[1] == b.shape()[1]);
    debug_assert!(c.shape()[0] == a.shape()[0]);
    debug_assert!(c.shape()[1] == b.shape()[0]);
    let vec_shape = [a.shape()[1]];
    #[cfg(not(feature = "rayon"))]
    cartesian_product2(0..c.shape()[0], 0..c.shape()[1]).for_each(|(i, j)| {
        let a_vec = a.slice(
            Tensor::<f32>::index_to_offset(&[i, 0], a.shape()),
            &vec_shape,
        );
        let b_vec = b.slice(
            Tensor::<f32>::index_to_offset(&[j, 0], b.shape()),
            &vec_shape,
        );
        let prod = dot(&a_vec, &b_vec);
        unsafe {
            c.with_data_mut_at(&[i, j], |&prev| alpha * prod + beta * prev);
        }
    });

    #[cfg(feature = "rayon")]
    {
        let c_shape = c.shape().clone();
        let c_data = unsafe { c.data_mut() };
        use rayon::prelude::*;
        c_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(offset, c_val)| {
                let idx = Tensor::<f32>::offset_to_index(offset, &c_shape);
                let i = idx[0];
                let j = idx[1];
                let a_vec = a.slice(
                    Tensor::<f32>::index_to_offset(&[i, 0], a.shape()),
                    &vec_shape,
                );
                let b_vec = b.slice(
                    Tensor::<f32>::index_to_offset(&[j, 0], b.shape()),
                    &vec_shape,
                );
                let prod = dot(&a_vec, &b_vec);
                *c_val = alpha * prod + beta * *c_val;
            });
    }
}

pub fn cartesian_product2<I, J>(iter1: I, iter2: J) -> impl Iterator<Item = (I::Item, J::Item)>
where
    I: Iterator + Clone,
    J: Iterator + Clone,
    I::Item: Clone,
    J::Item: Clone,
{
    iter1.flat_map(move |item1| iter2.clone().map(move |item2| (item1.clone(), item2)))
}

// Dot product of two tensors (treated as vectors)
#[cfg(not(feature = "rayon"))]
pub fn dot<'a, T: num_traits::Num + Copy + Clone + Default + std::iter::Sum>(
    x_vec: &'a Tensor<T>,
    y_vec: &'a Tensor<T>,
) -> T {
    debug_assert!(x_vec.size() == y_vec.size());
    x_vec
        .data()
        .iter()
        .zip(y_vec.data())
        .map(|(&a, &b)| a * b)
        .sum()
}

#[cfg(feature = "rayon")]
pub fn dot(x_vec: &Tensor<f32>, y_vec: &Tensor<f32>) -> f32 {
    debug_assert!(x_vec.size() == y_vec.size());
    use rayon::prelude::*;
    (x_vec.data(), y_vec.data())
        .into_par_iter()
        .map(|(&a, &b)| a * b)
        .sum()
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, &p): (usize, &f32)) -> Self {
            Self {
                val: p,
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    //          [[1,2,3],  [[1,4],
    //  a@b^T =  [4,5,6]] x [2,5],  = [14,32]
    //                      [3,6]]    [32,77]
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_dot_product() {
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![1, 4]);
    let y = Tensor::<f32>::new(vec![2., 2., 3., 4.], &vec![1, 4]);
    assert!((dot(&x, &y) - 31.).abs() <= 1e-6);
}
