use getset::Getters;
use num_traits::{Float, Num};
use std::{fmt::Debug, slice, sync::Arc};

#[derive(Getters, Clone)]
pub struct Tensor<T: Num> {
    data: Arc<Box<[T]>>,
    #[getset(get = "pub")]
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Num + Copy + Clone + Default> Tensor<T> {
    pub fn default(shape: &[usize]) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }
}

impl<T: Num + Copy + Clone> Tensor<T> {
    pub fn data_at(&self, idx: &[usize]) -> T {
        self.data()[self.to_offset(idx)]
    }
}

impl<T: Num> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice()),
            shape: shape.to_vec(),
            offset: 0,
            length,
        }
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    fn check_idx(&self, idx: &[usize]) {
        debug_assert!(self.shape().len() == idx.len());
        debug_assert!(self.shape().iter().zip(idx.iter()).all(|(&s, &i)| i < s));
    }

    pub fn index_to_offset(idx: &[usize], shape: &[usize]) -> usize {
        idx.iter()
            .zip(shape)
            .fold(0, |acc, (&i, &dim)| acc * dim + i)
    }
    pub fn offset_to_index(offset: usize, shape: &[usize]) -> Vec<usize> {
        let mut idx = Vec::with_capacity(shape.len());
        let mut offset = offset;
        for &dim in shape.iter().rev() {
            idx.push(offset % dim);
            offset /= dim;
        }
        idx.reverse();
        idx
    }

    pub fn to_offset(&self, idx: &[usize]) -> usize {
        self.check_idx(idx);
        Self::index_to_offset(idx, &self.shape)
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    /// Mutates the tensor at a given index,
    /// and returns the previous value.
    pub unsafe fn with_data_mut_at(&mut self, idx: &[usize], op: impl FnOnce(&T) -> T) -> T {
        let offset = self.to_offset(idx);
        let ptr = self.data.as_ptr().add(self.offset + offset) as *mut T;
        let prev_val = ptr.read();
        ptr.write(op(&prev_val));
        prev_val
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &[usize]) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.to_owned();
        self
    }

    pub fn slice(&self, start: usize, shape: &[usize]) -> Self {
        let length: usize = shape.iter().product();
        assert!(length <= self.length && start <= self.length - length);
        Tensor {
            data: self.data.clone(),
            shape: shape.to_owned(),
            offset: self.offset + start,
            length,
        }
    }
}

impl<T: Float> Tensor<T> {
    pub fn close_to(&self, other: &Self, rel: impl Float) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        self.data()
            .iter()
            .zip(other.data())
            .all(|(x, y)| float_eq(x, y, rel))
    }
}

// Some helper functions for testing and debugging
impl<T: Num + Debug> Tensor<T> {
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

pub fn float_eq<T: Float>(x: &T, y: &T, rel: impl Float) -> bool {
    let rel = T::from(rel).unwrap();
    x.sub(*y).abs() <= rel * (x.abs() + y.abs()) / T::from(2).unwrap()
}

#[test]
fn test_data_at_idx() {
    let t = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    // [[1., 2., 3.],
    // [4., 5., 6.]]
    assert!(t.data_at(&[0, 0]) == 1.);
    assert!(t.data_at(&[0, 1]) == 2.);
    assert!(t.data_at(&[0, 2]) == 3.);
    assert!(t.data_at(&[1, 0]) == 4.);
}

#[test]
fn test_mutate_dat_at_idx() {
    let mut t = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    // [[1., 2., 3.],
    // [4., 5., 6.]]
    assert!(unsafe { t.with_data_mut_at(&[0, 0], |x| x + 1.) } == 1.);
    assert!(unsafe { t.with_data_mut_at(&[0, 1], |x| x + 1.) } == 2.);
    assert!(unsafe { t.with_data_mut_at(&[0, 2], |x| x + 1.) } == 3.);
    assert!(unsafe { t.with_data_mut_at(&[1, 0], |x| x + 1.) } == 4.);
    assert!(t.data_at(&[0, 0]) == 2.);
    assert!(t.data_at(&[0, 1]) == 3.);
    assert!(t.data_at(&[0, 2]) == 4.);
    assert!(t.data_at(&[1, 0]) == 5.);
}
