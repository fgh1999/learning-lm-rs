use std::{slice, sync::Arc, vec};
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    fn check_idx(&self, idx: &[usize]) {
        assert!(self.shape().len() == idx.len());
        assert!(self.shape().iter().zip(idx.iter()).all(|(&s, &i)| i < s));
    }
    pub fn to_offset(&self, idx: &[usize]) -> usize {
        self.check_idx(idx);
        idx.iter()
            .zip(self.shape())
            .fold(0, |acc, (&i, &dim)| acc * dim + i)
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

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }

    pub fn copy_from_tensor(&mut self, other: &Self) {
        assert!(self.shape() == other.shape());
        let data = unsafe { self.data_mut() };
        data.copy_from_slice(other.data());
    }

    pub fn deep_copy(&self) -> Self {
        let mut copy = Self::default(self.shape());
        copy.copy_from_tensor(self);
        copy
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
            "shpae: {:?}, offset: {}, length: {}",
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
