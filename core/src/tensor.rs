use getset::Getters;
use num_traits::{Float, Num};
use std::{fmt::Debug, ops::Index, slice, sync::Arc};

#[derive(Getters, Clone)]
pub struct Tensor<T: Num> {
    data: Arc<[T]>,
    #[getset(get = "pub")]
    shape: Vec<usize>,
    storage_offset: usize,
    length: usize,
}

impl<T: Num + Copy + Default> Tensor<T> {
    pub fn default(shape: &[usize]) -> Self {
        Self::new(vec![T::default(); shape.iter().product()], shape)
    }
}
impl<T: Num> Index<&[usize]> for Tensor<T>
where
    Tensor<T>: TensorIndex,
{
    type Output = T;
    fn index(&self, idx: &[usize]) -> &Self::Output {
        &self.data()[self.to_offset(idx)]
    }
}

pub trait TensorIndex {
    fn index_to_offset(idx: &[usize], shape: &[usize]) -> usize {
        idx.iter()
            .zip(shape)
            .fold(0, |acc, (&i, &dim)| acc * dim + i)
    }
    fn offset_to_index(offset: usize, shape: &[usize]) -> Vec<usize> {
        let mut idx = Vec::with_capacity(shape.len());
        let mut offset = offset;
        for &dim in shape.iter().rev() {
            idx.push(offset % dim);
            offset /= dim;
        }
        idx.reverse();
        idx
    }
    fn to_offset(&self, idx: &[usize]) -> usize {
        debug_assert!(self.shape().len() == idx.len());
        debug_assert!(self.shape().iter().zip(idx.iter()).all(|(&s, &i)| i < s));
        Self::index_to_offset(idx, self.shape())
    }
    fn size(&self) -> usize;
    fn shape(&self) -> &[usize];
}
impl<T: Num> TensorIndex for Tensor<T> {
    fn size(&self) -> usize {
        self.length
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub trait TensorView<T>: TensorIndex {
    /// Returns the reference to the data at a given index.
    fn data_at(&self, idx: &[usize]) -> &T;
    /// Returns an iterator over the data.
    fn data_iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;
    fn slice(&self, start: usize, shape: &[usize]) -> Self;
}
impl<T: Num> TensorView<T> for Tensor<T>
where
    Tensor<T>: TensorIndex,
{
    fn data_at(&self, idx: &[usize]) -> &T {
        &self.data()[self.to_offset(idx)]
    }
    fn data_iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self.data().iter()
    }
    fn slice(&self, start: usize, shape: &[usize]) -> Self {
        let length = shape.iter().product();
        assert!(length <= self.length && start <= self.length - length);
        Tensor {
            data: self.data.clone(),
            shape: shape.to_owned(),
            storage_offset: self.storage_offset + start,
            length,
        }
    }
}

/// # Safety
///
/// Writing to a tensor view is unsafe because it can not guarantee that the tensor is not shared.
pub unsafe trait WritableTensorView<T>: TensorView<T> {
    /// Mutates the tensor at a given index,
    /// and returns the previous value.
    ///
    /// # Safety
    ///
    /// Writing to a tensor view is unsafe because it can not guarantee that the tensor is not shared.
    unsafe fn with_data_mut_at(&mut self, idx: &[usize], op: impl FnOnce(&T) -> T) -> T;

    /// Returns a mutable iterator over the data.
    ///
    /// # Safety
    ///
    /// Writing to a tensor view is unsafe because it can not guarantee that the tensor is not shared.
    unsafe fn data_iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        T: 'a;
}
unsafe impl<T: Num> WritableTensorView<T> for Tensor<T> {
    unsafe fn with_data_mut_at(&mut self, idx: &[usize], op: impl FnOnce(&T) -> T) -> T {
        let offset = self.to_offset(idx);
        unsafe {
            let ptr = self.data.as_ptr().add(self.storage_offset + offset) as *mut T;
            let prev_val = ptr.read();
            ptr.write(op(&prev_val));
            prev_val
        }
    }
    unsafe fn data_iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        T: 'a,
    {
        self.data_mut().iter_mut()
    }
}

impl<T: Num> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
        let length = data.len();
        Tensor {
            data: data.into(),
            shape: shape.to_vec(),
            storage_offset: 0,
            length,
        }
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.storage_offset..][..self.length]
    }

    /// # Safety
    ///
    /// Writing to a tensor is unsafe because it can not guarantee that the tensor is not shared.
    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.storage_offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
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

    /// # Safety
    ///
    /// Writing to a tensor is unsafe because it can not guarantee that the tensor is not shared.
    pub unsafe fn erase(&mut self) {
        self.data_mut().iter_mut().for_each(|x| *x = T::zero());
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
            "shape: {:?}, storage_offset: {}, length: {}",
            self.shape, self.storage_offset, self.length
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
    assert_eq!(t[&[0, 0]], 1.);
    assert_eq!(t[&[0, 1]], 2.);
    assert_eq!(t[&[0, 2]], 3.);
    assert_eq!(t[&[1, 0]], 4.);
}

#[test]
fn test_mutate_data_at_idx() {
    let mut t = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    // [[1., 2., 3.],
    // [4., 5., 6.]]
    assert_eq!(unsafe { t.with_data_mut_at(&[0, 0], |x| x + 1.) }, 1.);
    assert_eq!(unsafe { t.with_data_mut_at(&[0, 1], |x| x + 1.) }, 2.);
    assert_eq!(unsafe { t.with_data_mut_at(&[0, 2], |x| x + 1.) }, 3.);
    assert_eq!(unsafe { t.with_data_mut_at(&[1, 0], |x| x + 1.) }, 4.);
    assert_eq!(t[&[0, 0]], 2.);
    assert_eq!(t[&[0, 1]], 3.);
    assert_eq!(t[&[0, 2]], 4.);
    assert_eq!(t[&[1, 0]], 5.);
}

#[test]
fn test_erase_tensor() {
    let mut t = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    unsafe {
        t.erase();
    }
    assert!(t.data().iter().all(|&x| x == 0.));
}
