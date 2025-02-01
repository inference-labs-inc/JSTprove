use expander_compiler::frontend::*;


pub fn matrix_addition<C: Config, const M: usize, const N: usize>(api: &mut API<C>, matrix_a: [[Variable; N]; M], matrix_b: [[Variable; N]; M]) -> [[Variable; N]; M]{
    let mut array:[[Variable; N]; M]  = [[Variable::default(); N]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..N {
            array[i][j] = api.add(matrix_a[i][j], matrix_b[i][j]);
        }                       
    }
    array
}