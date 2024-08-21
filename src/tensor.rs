use getset::Getters;
use num_traits::Num;
use std::{slice, sync::Arc};

#[derive(Getters, Clone)]
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    #[getset(get = "pub")]
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Num + Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice()),
            shape: shape.to_vec(),
            offset: 0,
            length,
        }
    }

    pub fn default(shape: &[usize]) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
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
    pub fn data_at(&self, idx: &[usize]) -> T {
        self.data()[self.to_offset(idx)]
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

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
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

    pub fn plus_(&mut self, other: &Self) {
        assert!(self.shape() == other.shape());
        let data = unsafe { self.data_mut() };
        data.iter_mut()
            .zip(other.data())
            .for_each(|(x, &y)| *x += y);
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
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
