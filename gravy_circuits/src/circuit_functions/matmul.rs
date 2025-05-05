use expander_compiler::frontend::*;

/// Computes the matrix product AB over the finite field defined by the circuit config.
/// This assumes `a` and `b` contain `Variable`s that are already encoded as field elements
/// using some signed integer encoding (e.g., least residue).
///
/// # Arguments
/// * `api` - Mutable reference to the circuit builder.
/// * `a` - Matrix A of shape (l x m), represented as `Vec<Vec<Variable>>`.
/// * `b` - Matrix B of shape (m x n), represented as `Vec<Vec<Variable>>`.
///
/// # Returns
/// A matrix product AB of shape (l x n), where each entry is a `Variable`.
pub fn matrix_multiplication<C: Config, API: RootAPI<C>>(
    api: &mut API,
    a: Vec<Vec<Variable>>, // shape (l x m)
    b: Vec<Vec<Variable>>, // shape (m x n)
) -> Vec<Vec<Variable>> {
    let l = a.len();            // number of rows in A
    let m = a[0].len();         // number of cols in A = number of rows in B
    let n = b[0].len();         // number of cols in B

    let zero = api.constant(CircuitField::<C>::zero());
    let mut product = vec![vec![zero; n]; l]; // (l x n)

    for i in 0..l {
        for j in 0..n {
            for k in 0..m {
                let prod = api.mul(a[i][k], b[k][j]);
                product[i][j] = api.add(product[i][j], prod);
            }
        }
    }

    product
}

