use crate::circuit_functions::utils::UtilsError;

pub trait AsIsize {
    /// Converts a `usize` value to `isize`.
    ///
    /// # Returns
    /// - `Ok(isize)` if the conversion succeeds.
    ///
    /// # Errors
    /// - [`UtilsError::ValueConversionError`] if the value is too large to fit in an `isize`.
    ///
    /// # Examples
    /// ```
    /// use crate::jstprove_circuits::circuit_functions::utils::typecasting::AsIsize;
    /// let x: usize = 42;
    /// let y = x.as_isize().unwrap();
    /// assert_eq!(y, 42isize);
    /// ```
    fn as_isize(&self) -> Result<isize, UtilsError>;
}

impl AsIsize for usize {
    fn as_isize(&self) -> Result<isize, UtilsError> {
        isize::try_from(*self).map_err(|_| UtilsError::ValueConversionError {
            initial_var_type: "usize".to_string(),
            converted_var_type: "isize".to_string(),
        })
    }
}

pub trait AsUsize {
    /// Converts an `i32` value to `usize`.
    ///
    /// # Returns
    /// - `Ok(usize)` if the conversion succeeds.
    ///
    /// # Errors
    /// - [`UtilsError::ValueConversionError`] if the value is negative.
    ///
    /// # Examples
    /// ```
    /// use crate::jstprove_circuits::circuit_functions::utils::typecasting::AsUsize;
    /// let x: i32 = 42;
    /// let y = x.as_usize().unwrap();
    /// assert_eq!(y, 42usize);
    /// ```
    fn as_usize(&self) -> Result<usize, UtilsError>;
}

impl AsUsize for i32 {
    fn as_usize(&self) -> Result<usize, UtilsError> {
        usize::try_from(*self).map_err(|_| UtilsError::ValueConversionError {
            initial_var_type: "i32".to_string(),
            converted_var_type: "usize".to_string(),
        })
    }
}

pub trait AsI32 {
    /// Converts a `usize` value to `i32`.
    ///
    /// # Returns
    /// - `Ok(i32)` if the conversion succeeds.
    ///
    /// # Errors
    /// - [`UtilsError::ValueConversionError`] if the value is too large to fit in an `i32`.
    ///
    /// # Examples
    /// ```
    /// use crate::jstprove_circuits::circuit_functions::utils::typecasting::AsI32;
    /// let x: usize = 42;
    /// let y = x.as_i32().unwrap();
    /// assert_eq!(y, 42i32);
    /// ```
    fn as_i32(&self) -> Result<i32, UtilsError>;
}

impl AsI32 for usize {
    fn as_i32(&self) -> Result<i32, UtilsError> {
        i32::try_from(*self).map_err(|_| UtilsError::ValueConversionError {
            initial_var_type: "usize".to_string(),
            converted_var_type: "i32".to_string(),
        })
    }
}

impl AsI32 for u32 {
    fn as_i32(&self) -> Result<i32, UtilsError> {
        i32::try_from(*self).map_err(|_| UtilsError::ValueConversionError {
            initial_var_type: "u32".to_string(),
            converted_var_type: "i32".to_string(),
        })
    }
}

/// Converts a 32-bit signed integer into an unsigned usize value.
///
/// Attempts to safely transform the given `i32` into a `usize` using
/// Rust’s `TryInto` trait. Ensures that negative values and values
/// exceeding the platform’s `usize` range are caught.
///
/// # Arguments
/// - `val`: The `i32` integer to be converted.
///
/// # Returns
/// A `Result` containing the converted `usize` on success.
///
/// # Errors
/// - [`UtilsError::ValueConversionError`] if `val` is negative or does not fit
///   within the valid range of `usize` for the current platform.
pub fn i32_to_usize(val: i32) -> Result<usize, UtilsError> {
    val.try_into()
        .map_err(|_| UtilsError::ValueConversionError {
            initial_var_type: "i32".to_string(),
            converted_var_type: "usize".to_string(),
        })
}

/// Converts a `u64` value to `u32` safely, returning an error if it overflows.
pub trait AsU32 {
    /// Converts a `u64` to `u32`.
    ///
    /// # Returns
    /// - `Ok(u32)` if the conversion succeeds.
    ///
    /// # Errors
    /// - [`UtilsError::ValueConversionError`] if the value is too large to fit in `u32`.
    fn as_u32(&self) -> Result<u32, UtilsError>;
}

impl AsU32 for u64 {
    fn as_u32(&self) -> Result<u32, UtilsError> {
        u32::try_from(*self).map_err(|_| UtilsError::ValueConversionError {
            initial_var_type: "u64".to_string(),
            converted_var_type: "u32".to_string(),
        })
    }
}

pub trait UsizeAsU32 {
    /// Provides a safe conversion from `usize` to `u32`.
    ///
    /// # Returns
    /// - `Ok(u32)` if the conversion succeeds.
    /// - `Err(UtilsError::ValueConversionError)` if the `usize` value is too large.
    ///
    /// # Errors
    /// - [`UtilsError::ValueConversionError`] if the `usize` value is too large to fit in a `u32`.
    ///
    /// # Examples
    /// ```
    /// use crate::jstprove_circuits::circuit_functions::utils::typecasting::UsizeAsU32;
    /// let x: usize = 42;
    /// let y: u32 = x.as_u32().unwrap();
    /// assert_eq!(y, 42u32);
    /// ```
    fn as_u32(&self) -> Result<u32, UtilsError>;
}

impl UsizeAsU32 for usize {
    fn as_u32(&self) -> Result<u32, UtilsError> {
        u32::try_from(*self).map_err(|_| UtilsError::ValueConversionError {
            initial_var_type: "usize".to_string(),
            converted_var_type: "u32".to_string(),
        })
    }
}
