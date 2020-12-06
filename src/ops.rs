//! Operations common to various bit-packed vectors.

//-----------------------------------------------------------------------------

// FIXME example

/// A vector that contains elements of a fixed type.
pub trait Element {
    /// The type of the elements in the vector.
    type Item;

    /// Returns the width of of an element in bits.
    fn width(&self) -> usize;
}

// FIXME example

/// Conversion into a more space-efficient representation of the same data in a vector of the same type.
///
/// This may, for example, reduce the width of an element to accommodate the largest element in the vector.
pub trait Pack: Element {
    /// Converts the vector into a more space-efficient vector of the same type.
    fn pack(self) -> Self;
}

//-----------------------------------------------------------------------------

// FIXME example

/// Append elements to a vector.
pub trait Push: Element {
    /// Appends an element to the vector.
    ///
    /// Behavior is undefined if there is an integer overflow.
    fn push(&mut self, element: <Self as Element>::Item);
}

// FIXME example

/// Remove and return top elements from a vector.
pub trait Pop: Element {
    /// The type of the elements in the vector.
    type Item;

    /// Removes and returns the last element from the vector.
    /// Returns `None` if there are no more elements in the vector.
    fn pop(&mut self) -> Option<<Self as Element>::Item>;
}

//-----------------------------------------------------------------------------

// FIXME example

/// Set elements in a vector.
pub trait Set: Element {
    /// Sets an element int the vector.
    ///
    /// Behavior is undefined if `index` is not a valid index in the vector.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `value`: New value of the element.
    fn set(&mut self, index: usize, value: <Self as Element>::Item);
}

// FIXME example

/// Get elements from a vector.
pub trait Get: Element {
    /// Gets an element from the vector.
    ///
    /// Behavior is undefined if `index` is not a valid index in the vector.
    fn get(&self, index: usize) -> <Self as Element>::Item;
}

//-----------------------------------------------------------------------------

// FIXME example

/// A vector that contains elements with a fixed number of subelements of a fixed type in each element.
///
/// Term *index* refers to the location of an element within a vector, while *offset* refers to the location of a subelement within an element.
/// Every subelement at the same offset has the same width in bits.
pub trait SubElement {
    /// The type of the subelements of an element.
    type SubItem;

    /// Returns the number of subelements in an element.
    fn element_len(&self) -> usize;

    /// Returns the width of the specified subelement in bits.
    ///
    /// Behavior is undefined if `offset >= self.element_len()`.
    fn sub_width(&self, offset: usize) -> usize;
}

// FIXME example

/// Set subelements in a vector.
pub trait SetSub: SubElement {
    /// Sets a subelement in the vector.
    ///
    /// Behavior is undefined if `index` is not a valid index in the vector or `offset` is not a valid offset in the element.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `offset`: Offset in the element.
    /// * `value`: New value of the subelement.
    fn set_sub(&mut self, index: usize, offset: usize, value: <Self as SubElement>::SubItem);
}

// FIXME example

/// Get subelements from a vector.
pub trait GetSub: SubElement {
    /// Gets a subelement from the vector.
    ///
    /// Behavior is undefined if `index` is not a valid index in the vector or `offset` is not a valid offset in the element.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `offset`: Offset in the element.
    fn get_sub(&self, index: usize, offset: usize) -> <Self as SubElement>::SubItem;
}

//-----------------------------------------------------------------------------
