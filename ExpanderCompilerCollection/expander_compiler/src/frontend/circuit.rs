#[macro_export]
macro_rules! declare_circuit_field_type {
    (@type Variable) => {
        T
    };

    (@type PublicVariable) => {
        T
    };

    (@type [$elem:tt; $n:expr]) => {
        [$crate::frontend::internal::declare_circuit_field_type!(@type $elem); $n]
    };

    (@type [$elem:tt]) => {
        Vec<$crate::frontend::internal::declare_circuit_field_type!(@type $elem)>
    };

    (@type $other:ty) => {
        $other
    };
}

#[macro_export]
macro_rules! declare_circuit_dump_into {
    ($field_value:expr, @type Variable, $vars:expr, $public_vars:expr) => {
        $field_value.dump_into($vars);
    };

    ($field_value:expr, @type PublicVariable, $vars:expr, $public_vars:expr) => {
        $field_value.dump_into($public_vars);
    };

    ($field_value:expr, @type [$elem:tt; $n:expr], $vars:expr, $public_vars:expr) => {
        for _x in $field_value.iter() {
            $crate::frontend::internal::declare_circuit_dump_into!(_x, @type $elem, $vars, $public_vars);
        }
    };

    ($field_value:expr, @type [$elem:tt], $vars:expr, $public_vars:expr) => {
        for _x in $field_value.iter() {
            $crate::frontend::internal::declare_circuit_dump_into!(_x, @type $elem, $vars, $public_vars);
        }
    };

    ($field_value:expr, @type $other:ty, $vars:expr, $public_vars:expr) => {
    };
}

#[macro_export]
macro_rules! declare_circuit_load_from {
    ($field_value:expr, @type Variable, $vars:expr, $public_vars:expr) => {
        $field_value.load_from($vars);
    };

    ($field_value:expr, @type PublicVariable, $vars:expr, $public_vars:expr) => {
        $field_value.load_from($public_vars);
    };

    ($field_value:expr, @type [$elem:tt; $n:expr], $vars:expr, $public_vars:expr) => {
        for _x in $field_value.iter_mut() {
            $crate::frontend::internal::declare_circuit_load_from!(_x, @type $elem, $vars, $public_vars);
        }
    };

    ($field_value:expr, @type [$elem:tt], $vars:expr, $public_vars:expr) => {
        for _x in $field_value.iter_mut() {
            $crate::frontend::internal::declare_circuit_load_from!(_x, @type $elem, $vars, $public_vars);
        }
    };

    ($field_value:expr, @type $other:ty, $vars:expr, $public_vars:expr) => {
    };
}

#[macro_export]
macro_rules! declare_circuit_num_vars {
    ($field_value:expr, @type Variable, $cnt_sec:expr, $cnt_pub:expr, $array_cnt:expr) => {
        $cnt_sec += $array_cnt;
    };

    ($field_value:expr, @type PublicVariable, $cnt_sec:expr, $cnt_pub:expr, $array_cnt:expr) => {
        $cnt_pub += $array_cnt;
    };

    ($field_value:expr, @type [$elem:tt; $n:expr], $cnt_sec:expr, $cnt_pub:expr, $array_cnt:expr) => {
        $crate::frontend::internal::declare_circuit_num_vars!($field_value[0], @type $elem, $cnt_sec, $cnt_pub, $array_cnt * $n);
    };

    ($field_value:expr, @type [$elem:tt], $cnt_sec:expr, $cnt_pub:expr, $array_cnt:expr) => {
        for _x in $field_value.iter() {
            $crate::frontend::internal::declare_circuit_num_vars!(_x, @type $elem, $cnt_sec, $cnt_pub, $array_cnt);
        }
    };

    ($field_value:expr, @type $other:ty, $cnt_sec:expr, $cnt_pub:expr, $array_cnt:expr) => {
    };
}

#[macro_export]
macro_rules! declare_circuit_default {
    (@type Variable) => {
        Default::default()
    };

    (@type [$elem:tt; $n:expr]) => {
        [$crate::frontend::internal::declare_circuit_default!(@type $elem); $n]
    };

    (@type $other:ty) => {
        Default::default()
    };
}

#[macro_export]
macro_rules! declare_circuit {
    ($struct_name:ident { $($field_name:ident : $field_type:tt),* $(,)? }) => {
        pub struct $struct_name<T> {
            pub $($field_name: $crate::frontend::internal::declare_circuit_field_type!(@type $field_type)),*
        }

        impl<B: Clone> $crate::frontend::internal::DumpLoadTwoVariables<B> for $struct_name<B> where B: $crate::frontend::internal::DumpLoadVariables<B>{
            #[allow(unused_variables)]
            fn dump_into(&self, vars: &mut Vec<B>, public_vars: &mut Vec<B>) {
                $($crate::frontend::internal::declare_circuit_dump_into!(self.$field_name, @type $field_type, vars, public_vars);)*
            }
            #[allow(unused_variables)]
            fn load_from(&mut self, vars: &mut &[B], public_vars: &mut &[B]) {
                $($crate::frontend::internal::declare_circuit_load_from!(self.$field_name, @type $field_type, vars, public_vars);)*
            }
            #[allow(unused_mut)]
            fn num_vars(&self) -> (usize, usize) {
                let mut cnt_sec = 0;
                let mut cnt_pub = 0;
                $($crate::frontend::internal::declare_circuit_num_vars!(self.$field_name, @type $field_type, cnt_sec, cnt_pub, 1);)*
                (cnt_sec, cnt_pub)
            }
        }

        impl<T: Clone> Clone for $struct_name<T> {
            fn clone(&self) -> Self {
                Self {
                    $($field_name: self.$field_name.clone()),*
                }
            }
        }

        impl<T: Default + Copy> Default for $struct_name<T> {
            fn default() -> Self {
                Self {
                    $($field_name: $crate::frontend::internal::declare_circuit_default!(@type $field_type)),*
                }
            }
        }
    };
}

pub use declare_circuit;
pub use declare_circuit_default;
pub use declare_circuit_dump_into;
pub use declare_circuit_field_type;
pub use declare_circuit_load_from;
pub use declare_circuit_num_vars;

use crate::circuit::config::Config;

use super::builder::RootBuilder;
pub trait Define<C: Config> {
    fn define(&self, api: &mut RootBuilder<C>);
}
