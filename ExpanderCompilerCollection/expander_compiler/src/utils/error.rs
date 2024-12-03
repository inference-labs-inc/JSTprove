use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    UserError(String),
    InternalError(String),
}

impl Error {
    pub fn is_user(&self) -> bool {
        matches!(self, Error::UserError(_))
    }

    pub fn is_internal(&self) -> bool {
        matches!(self, Error::InternalError(_))
    }

    pub fn prepend(&self, prefix: &str) -> Error {
        match self {
            Error::UserError(s) => Error::UserError(format!("{}: {}", prefix, s)),
            Error::InternalError(s) => Error::InternalError(format!("{}: {}", prefix, s)),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UserError(s) => write!(f, "{}", s),
            Error::InternalError(s) => write!(f, "{}", s),
        }
    }
}
