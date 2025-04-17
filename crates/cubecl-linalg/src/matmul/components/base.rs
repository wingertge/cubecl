use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{InputRuntimeArg, MatmulConfigFactory, MatmulSpec, OutputRuntimeArg};

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MatmulSize {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

#[derive(Debug)]
pub struct MatmulSelection {
    pub tile_shape: MatmulSize,
    pub tile_count: MatmulSize,
    pub plane_dim: u32,
    pub rows_per_plane: u32,
}

/// Provides launch entry point to solve a matmul
pub trait MatmulLaunch: MatmulConfigFactory {
    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        size_k: ScalarArg<u32>,
        config: <Self as MatmulConfigFactory>::Config,
    );
}
