use crate::matmul::components::{
    Ident, MatmulConfig, MatrixLayout, TilingDimensions,
    stage::{self},
};

pub const PRECOMPUTE_JOB: bool = false;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the pipelined global matmul
pub struct CommonGlobalConfig<S: stage::StageConfig> {
    pub smm_config: S,
    pub check_m_bounds: bool,
    pub check_n_bounds: bool,
    pub check_k_bounds: bool,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u32,
    pub rhs_line_size: u32,
    pub out_line_size: u32,
    pub num_planes: u32,
}

impl<S: stage::StageConfig> super::GlobalConfig for CommonGlobalConfig<S> {
    type SmmConfig = S;

    fn to_smm_config(&self) -> Self::SmmConfig {
        self.smm_config
    }

    fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
        match ident.into() {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }

    fn tiling_dimensions<I: Into<Ident>>(&self, ident: I) -> TilingDimensions {
        self.smm_config.tiling_dimensions(ident.into())
    }

    fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout {
        match ident.into() {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => self.smm_config.matrix_layout(Ident::Out),
        }
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.smm_config.plane_dim()
    }

    fn check_row_bounds<I: Into<Ident>>(&self, ident: I) -> bool {
        match ident.into() {
            Ident::Lhs => self.check_m_bounds,
            Ident::Rhs => self.check_k_bounds,
            Ident::Out => self.check_m_bounds,
        }
    }

    fn check_col_bounds<I: Into<Ident>>(&self, ident: I) -> bool {
        match ident.into() {
            Ident::Lhs => self.check_k_bounds,
            Ident::Rhs => self.check_n_bounds,
            Ident::Out => self.check_n_bounds,
        }
    }

    fn check_k_bounds(&self) -> bool {
        self.check_k_bounds
    }

    fn precompute_job(&self) -> bool {
        PRECOMPUTE_JOB
    }
}

impl<S: stage::StageConfig> MatmulConfig for CommonGlobalConfig<S> {}

impl<S: stage::StageConfig> CommonGlobalConfig<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        smm_config: S,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
        num_planes: u32,
    ) -> Self {
        Self {
            smm_config,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
            num_planes,
        }
    }
}
