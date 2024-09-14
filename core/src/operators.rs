use std::{
    collections::HashSet,
    ops::{DivAssign, MulAssign},
};

use crate::tensor::{Tensor, TensorIndex, TensorView, WritableTensorView};
use num_traits::{Float, Num};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// get (row) vectors from a 2D table given a list of indices
pub fn gather<P: Num + Copy>(y: &mut Tensor<P>, indices: &Tensor<u32>, table: &Tensor<P>) {
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    let length = indices.size();
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = table.slice(table.to_offset(&[indices.data()[i] as usize, 0]), &[dim]);
        let mut dst = y.slice(y.to_offset(&[i, 0]), &[dim]);
        unsafe {
            dst.data_mut().copy_from_slice(src.data());
        }
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope<P: Float, D: WritableTensorView<P>>(y: &mut D, start_pos: usize, theta: impl Float) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let theta = P::from(theta).unwrap();

    for tok in 0..seq_len {
        let pos = P::from(start_pos + tok).unwrap();
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let (sin, cos) = {
                    let pow_param = P::from(i * 2).unwrap() / P::from(d).unwrap();
                    let freq = pos / theta.powf(pow_param);
                    freq.sin_cos()
                };

                let a_idx = [tok, head, i];
                let b_idx = [tok, head, i + d / 2];
                let b = *y.data_at(&b_idx);
                unsafe {
                    let a = y.with_data_mut_at(&a_idx, |a| a.mul(cos) - b * sin);
                    y.with_data_mut_at(&b_idx, |_| b * cos + a * sin);
                }
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax<P: Float + std::iter::Sum + DivAssign>(y: &mut Tensor<P>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let total_batch_num = y.size() / (seq_len * total_seq_len);

    let data = unsafe { y.data_mut() };
    for batch in 0..total_batch_num {
        let base = batch * seq_len * total_seq_len;
        for seq_idx in 0..seq_len {
            let data = &mut data[base + seq_idx * total_seq_len..][..total_seq_len];
            let boundary = total_seq_len - seq_len + seq_idx + 1;
            let (unmasked_data, masked_data) = data.split_at_mut(boundary);

            let max = unmasked_data
                .iter()
                .fold(unmasked_data.first().cloned().unwrap(), |a, b| b.max(a));

            unmasked_data.iter_mut().for_each(|j| *j = (*j - max).exp());
            let sum = unmasked_data.iter().cloned().sum::<P>();
            unmasked_data.iter_mut().for_each(|j| *j /= sum);
            masked_data.iter_mut().for_each(|j| *j = P::zero());
        }
    }
}

pub fn rms_norm<P: Float + std::iter::Sum>(
    y: &mut Tensor<P>,
    x: &Tensor<P>,
    w: &Tensor<P>,
    epsilon: impl Float,
) {
    debug_assert!(y.shape() == x.shape());
    debug_assert!(x.shape().len() == w.shape().len() + 1 || w.size() == 1 && x.shape().len() == 1);
    debug_assert!(w
        .shape()
        .iter()
        .rev()
        .zip(x.shape().iter().rev())
        .all(|(w_s, x_s)| w_s == x_s));
    let epsilon = P::from(epsilon).unwrap();

    let chunk_size = w.shape().iter().product();
    let res = x.data().chunks_exact(chunk_size).map(|x_i| {
        let norm = {
            let square_sum = x_i.iter().map(|x_ij| x_ij.powi(2)).sum::<P>();
            (square_sum / P::from(chunk_size).unwrap() + epsilon).sqrt()
        };
        x_i.iter()
            .zip(w.data())
            .map(|(&x, &w)| x * w)
            .map(move |p| p / norm)
    });
    unsafe { y.data_mut() }
        .chunks_exact_mut(chunk_size)
        .zip(res)
        .for_each(|(y_i, y_i_res)| y_i.iter_mut().zip(y_i_res).for_each(|(y_ij, r)| *y_ij = r));
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu<P: Float + MulAssign, T: TensorView<P>>(y: &mut Tensor<P>, x: &T) {
    debug_assert!(y.size() == x.size());
    let silu_x = x.data_iter().cloned().map(|x| x / (P::one() + (-x).exp()));
    unsafe { y.data_mut() }
        .iter_mut()
        .zip(silu_x)
        .for_each(|(y, x)| *y *= x);
}

pub fn repetition_penalty<P: Float>(
    logits: &mut Tensor<P>,
    repetition_penalty: impl Float,
    prompt_token_ids: &[u32],
) {
    let repetition_penalty = P::from(repetition_penalty).unwrap();
    let used_token_ids = prompt_token_ids.iter().cloned().collect::<HashSet<_>>();

    let logits_shape = logits.shape().clone();
    let need_to_reshape = logits.shape().len() > 1;
    if need_to_reshape {
        logits.reshape(&[logits.size()]);
    }
    for i in used_token_ids {
        unsafe {
            logits.with_data_mut_at(&[i as usize], |&p| {
                if p.is_sign_positive() {
                    p / repetition_penalty
                } else {
                    p * repetition_penalty
                }
            });
        }
    }
    if need_to_reshape {
        logits.reshape(&logits_shape);
    }
}

fn check_matmul_shape<
    T0: TensorView<impl Num>,
    T1: TensorView<impl Num>,
    T2: TensorView<impl Num>,
>(
    c: &T0,
    a: &T1,
    b: &T2,
) {
    debug_assert!(c.shape().len() == 2);
    debug_assert!(a.shape().len() == 2);
    debug_assert!(b.shape().len() == 2);
    debug_assert!(a.shape()[1] == b.shape()[1]);
    debug_assert!(c.shape()[0] == a.shape()[0]);
    debug_assert!(c.shape()[1] == b.shape()[0]);
}

// C = beta * C + alpha * A @ B^T
// Assume that A is of shape (m, k), B is of shape (n, k), C is of shape (m, n).
#[cfg(not(feature = "rayon"))]
pub fn matmul_transb<P: Float + std::iter::Sum, D: WritableTensorView<P>, S: TensorView<P>>(
    c: &mut D,
    beta: impl Float,
    a: &S,
    b: &S,
    alpha: impl Float,
) {
    check_matmul_shape(c, a, b);
    let vec_shape = [a.shape()[1]];
    let beta = P::from(beta).unwrap();
    let alpha = P::from(alpha).unwrap();

    cartesian_product2(0..c.shape()[0], 0..c.shape()[1]).for_each(|(i, j)| {
        let a_vec = a.slice(a.to_offset(&[i, 0]), &vec_shape);
        let b_vec = b.slice(b.to_offset(&[j, 0]), &vec_shape);
        let prod = dot(&a_vec, &b_vec);
        unsafe {
            c.with_data_mut_at(&[i, j], |&prev| alpha * prod + beta * prev);
        }
    });
}

#[cfg(feature = "rayon")]
pub fn matmul_transb<
    'a,
    P: Float + std::iter::Sum + Sync + Send + 'a,
    W: Float,
    D: WritableTensorView<P>,
    S: TensorView<P> + Sync,
>(
    c: &'a mut D,
    beta: W,
    a: &'a S,
    b: &'a S,
    alpha: W,
) where
    Vec<(usize, &'a mut P)>: rayon::iter::IntoParallelIterator<Item = (usize, &'a mut P)>,
{
    check_matmul_shape(c, a, b);
    let vec_shape = [a.shape()[1]];
    let beta = P::from(beta).unwrap();
    let alpha = P::from(alpha).unwrap();

    let c_shape = c.shape().to_owned();
    let mut c_data: Vec<(usize, &mut P)> = unsafe { c.data_iter_mut() }.enumerate().collect();
    c_data.par_iter_mut().for_each(|(offset, c_val)| {
        let idx = Tensor::<P>::offset_to_index(*offset, &c_shape);
        let a_vec = a.slice(a.to_offset(&[idx[0], 0]), &vec_shape);
        let b_vec = b.slice(b.to_offset(&[idx[1], 0]), &vec_shape);
        let prod = dot(&a_vec, &b_vec);
        **c_val = alpha * prod + beta * **c_val;
    });
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
pub fn dot<P: Num + Copy + std::iter::Sum, T0: TensorView<P>, T1: TensorView<P>>(
    x_vec: &T0,
    y_vec: &T1,
) -> P {
    debug_assert!(x_vec.size() == y_vec.size());
    x_vec
        .data_iter()
        .zip(y_vec.data_iter())
        .map(|(&a, &b)| a * b)
        .sum()
}

// Samples an index from a tensor (treated as a probability vector)
pub fn random_sample<P: Float + Copy, T: TensorView<P>>(
    x: &T,
    top_p: f32,
    top_k: u32,
    temperature: f32,
) -> u32 {
    use std::cmp::Ordering;
    assert!(x.shape().last().cloned().unwrap() == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data_iter()
            .enumerate()
            .filter(|(_, i)| i.is_normal() || i.is_zero())
            .max_by(|(_, &a), (_, &b)| {
                if a > b {
                    Ordering::Greater
                } else if a < b {
                    Ordering::Less
                } else {
                    // ignores Nan Inf ...
                    Ordering::Equal
                }
            })
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability<T: Float + Copy> {
        val: T,
        tok: u32,
    }
    impl<T: Float + Copy> Eq for Probability<T> {}
    impl<T: Float + Copy> PartialOrd for Probability<T> {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl<T: Float + Copy> Ord for Probability<T> {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            if self.val > other.val {
                Ordering::Less
            } else if self.val < other.val {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }
    impl<T: Float + Copy> From<(usize, &T)> for Probability<T> {
        #[inline]
        fn from((i, &p): (usize, &T)) -> Self {
            Self {
                val: p,
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data_iter()
        .enumerate()
        .filter(|(_, i)| i.is_normal() || i.is_zero())
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, P::one());

    // softmax & sum
    let temperature = P::from(temperature).unwrap();
    logits.iter_mut().skip(1).fold(P::one(), |prev, p| {
        p.val = prev + ((p.val - max) / temperature).exp();
        p.val
    });
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits.last().cloned().unwrap().val * P::from(top_p).unwrap();
    let plimit = P::from(rand::random::<f32>()).unwrap() * P::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

#[test]
fn test_silu() {
    macro_rules! test_silu_for_float_type {
        ($type_:ty, $tolerance:expr, $type_converter:expr) => {{
            let x_src = vec![1., 2., 3.];
            let y_src = vec![2., 3., 4.];

            let mut y = Tensor::<$type_>::new(
                y_src.into_iter().map($type_converter).collect(),
                &vec![1, 3],
            );
            let x = Tensor::<$type_>::new(
                x_src.into_iter().map($type_converter).collect(),
                &vec![1, 3],
            );
            silu(&mut y, &x);
            assert!(y.close_to(
                &Tensor::<$type_>::new(
                    vec![1.4621172, 5.2847824, 11.43089]
                        .into_iter()
                        .map($type_converter)
                        .collect(),
                    &vec![1, 3]
                ),
                $tolerance
            ));
        }};
    }
    test_silu_for_float_type!(f32, 1e-6, f32::from);
    test_silu_for_float_type!(f64, 1e-6, f64::from);
    use half::{bf16, f16};
    test_silu_for_float_type!(f16, f16::EPSILON, f16::from_f32);
    test_silu_for_float_type!(bf16, bf16::EPSILON, bf16::from_f32);
}

#[test]
fn test_rms_norm() {
    macro_rules! test_rms_norm_for_float {
        ($type_:ty, $tolerance:expr, $type_converter:expr) => {{
            let y_src = vec![0., 0., 0., 0.];
            let x_src = vec![1., 2., 3., 4.];
            let w_src = vec![1., 2.];

            let mut y =
                Tensor::<$type_>::new(y_src.into_iter().map($type_converter).collect(), &[2, 2]);
            let x =
                Tensor::<$type_>::new(x_src.into_iter().map($type_converter).collect(), &[2, 2]);
            let w = Tensor::<$type_>::new(w_src.into_iter().map($type_converter).collect(), &[2]);
            rms_norm(&mut y, &x, &w, 1e-6);
            assert!(y.close_to(
                &Tensor::<$type_>::new(
                    vec![0.6324554, 2.5298216, 0.8485281, 2.2627416]
                        .into_iter()
                        .map($type_converter)
                        .collect(),
                    &[2, 2]
                ),
                $tolerance
            ));
        }};
    }
    test_rms_norm_for_float!(f32, 1e-6, f32::from);
    test_rms_norm_for_float!(f64, 1e-6, f64::from);
    use half::{bf16, f16};
    test_rms_norm_for_float!(f16, f16::EPSILON, f16::from_f32);
    test_rms_norm_for_float!(bf16, bf16::EPSILON, bf16::from_f32);
}

#[test]
fn test_matmul_transb() {
    use std::vec::Vec;
    macro_rules! test_matmul_transb_for_float_type {
        ($type_:ty, $tolerance:expr, $type_converter:expr) => {{
            let c_src: Vec<f32> = vec![1., 2., 3., 4.];
            let a_src: Vec<f32> = vec![1., 2., 3., 4., 5., 6.];
            let b_src: Vec<f32> = vec![1., 2., 3., 4., 5., 6.];
            //          [[1,2,3],  [[1,4],
            //  a@b^T =  [4,5,6]] x [2,5],  = [14,32]
            //                      [3,6]]    [32,77]
            let mut c = Tensor::<$type_>::new(
                c_src.iter().cloned().map($type_converter).collect(),
                &vec![2, 2],
            );
            let a = Tensor::<$type_>::new(
                a_src.iter().cloned().map($type_converter).collect(),
                &vec![2, 3],
            );
            let b = Tensor::<$type_>::new(
                b_src.iter().cloned().map($type_converter).collect(),
                &vec![2, 3],
            );
            matmul_transb(&mut c, 1., &a, &b, 1.);
            let tol = <$type_>::from($tolerance);
            assert!(c.close_to(
                &Tensor::<$type_>::new(
                    vec![15., 34., 35., 81.]
                        .into_iter()
                        .map($type_converter)
                        .collect(),
                    &vec![2, 2]
                ),
                tol
            ));
        }};
    }
    test_matmul_transb_for_float_type!(f32, 1e-6, f32::from);
    test_matmul_transb_for_float_type!(f64, 1e-6, f64::from);
    use half::{bf16, f16};
    test_matmul_transb_for_float_type!(f16, f16::EPSILON, f16::from_f32);
    test_matmul_transb_for_float_type!(bf16, bf16::EPSILON, bf16::from_f32);
}

#[test]
fn test_dot_product() {
    macro_rules! test_dot_product_for_integer_type {
        ($type_:ty, $src_ty:ty) => {{
            use std::vec::Vec;
            let x: Vec<$src_ty> = vec![1, 2, 3, 4];
            let y: Vec<$src_ty> = vec![2, 2, 3, 3];
            let x = x.into_iter().map(Into::into);
            let y = y.into_iter().map(Into::into);
            let x = Tensor::<$type_>::new(x.collect(), &vec![1, 4]);
            let y = Tensor::<$type_>::new(y.collect(), &vec![1, 4]);
            assert!(dot(&x, &y) == 27);
        }};
    }
    test_dot_product_for_integer_type!(i8, i8);
    test_dot_product_for_integer_type!(i16, i8);
    test_dot_product_for_integer_type!(i32, i8);
    test_dot_product_for_integer_type!(i64, i8);
    test_dot_product_for_integer_type!(u8, u8);
    test_dot_product_for_integer_type!(u16, u8);
    test_dot_product_for_integer_type!(u32, u8);
    test_dot_product_for_integer_type!(u64, u8);

    macro_rules! test_dot_product_for_float_type {
        ($type_:ty, $tolerance:expr, $type_converter:expr) => {{
            let x = vec![1., 2., 3., 4.].into_iter().map($type_converter);
            let y = vec![2., 2., 3., 4.].into_iter().map($type_converter);
            let x = Tensor::<$type_>::new(x.collect(), &vec![1, 4]);
            let y = Tensor::<$type_>::new(y.collect(), &vec![1, 4]);
            let correct = $type_converter(31.);
            assert!((dot(&x, &y) - correct).abs() <= $tolerance);
        }};
    }
    test_dot_product_for_float_type!(f32, 1e-6, f32::from);
    test_dot_product_for_float_type!(f64, 1e-6, f64::from);
    use half::{bf16, f16};
    test_dot_product_for_float_type!(f16, f16::EPSILON, f16::from_f32);
    test_dot_product_for_float_type!(bf16, bf16::EPSILON, bf16::from_f32);
}
