use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation,
        load::AsyncBufferLoadingStrategy,
        tensor_view::{TensorReader, Window},
    },
    stage::{Stage, StageConfig, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::AsyncLoadingJob;

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per contiguous slice.
/// The goal is to reduce the total number of memcpy_async calls, though it may result in idle threads.
pub struct LoadingStrategy {}

impl LoadingValidation for LoadingStrategy {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl AsyncBufferLoadingStrategy for LoadingStrategy {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] buffer_index: u32,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Job {
        let matrix_layout = config.matrix_layout(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);
        let line_size = config.to_smm_config().stage_line_size(input_ident.into());
        let num_buffers = 2;

        // If buffer is parallel to slices, slices are as long as in full stage, but there are less.
        // Otherwise, slices are shorter but there are as many as in full stage
        let (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset) = comptime! {
            match (input_ident, matrix_layout) {
                (InputIdent::Lhs, MatrixLayout::RowMajor) => {
                    let num_slices = tiling_dimensions.total_row();
                    let num_slices_buffer_offset = 0;
                    let slice_length = tiling_dimensions.total_col() / (num_buffers * line_size);
                    let slice_buffer_offset = buffer_index * slice_length;

                    (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset)
                },
                (InputIdent::Lhs, MatrixLayout::ColMajor) => {
                    let num_slices = tiling_dimensions.total_col() / num_buffers;
                    let num_slices_buffer_offset = buffer_index * num_slices;
                    let slice_length = tiling_dimensions.total_row() / line_size;
                    let slice_buffer_offset = 0;

                    (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset)
                },
                (InputIdent::Rhs, MatrixLayout::RowMajor) => {
                    let num_slices = tiling_dimensions.total_row() / num_buffers;
                    let num_slices_buffer_offset = buffer_index * num_slices;
                    let slice_length = tiling_dimensions.total_col() / line_size;
                    let slice_buffer_offset = 0;

                    (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset)
                },
                (InputIdent::Rhs, MatrixLayout::ColMajor) => {
                    let num_slices = tiling_dimensions.total_col();
                    let num_slices_buffer_offset = 0;
                    let slice_length = tiling_dimensions.total_row() / (num_buffers * line_size);
                    let slice_buffer_offset = buffer_index * slice_length;

                    (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset)
                },
            }
        };

        let unit_count = config.plane_dim() * config.num_planes();
        let num_tasks_per_unit = comptime!(num_slices.div_ceil(unit_count));

        Job {
            num_tasks_per_unit,
            unit_count,
            num_slices_buffer_offset,
            input_ident,
            slice_buffer_offset,
            slice_length,
            num_slices,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    num_slices_buffer_offset: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    slice_buffer_offset: u32,
    #[cube(comptime)]
    slice_length: u32,
    #[cube(comptime)]
    num_slices: u32,
}

#[cube]
impl<MP: MatmulPrecision> AsyncLoadingJob<MP, StridedTilingLayout> for Job {
    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, StridedTilingLayout>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let nth_slice_in_buffer = this.unit_count * task_id + UNIT_POS;

        let nth_slice = nth_slice_in_buffer + this.num_slices_buffer_offset;

        let window: Window<MP::EI> =
            tensor_reader.load_window_in_stage::<G>(nth_slice, this.input_ident, config);
        let mut destination: SliceMut<Line<MP::ES>> =
            StridedTilingLayout::nth_slice::<MP::ES, G::SmmConfig>(
                stage,
                nth_slice,
                comptime!(this.input_ident.as_ident()),
                config.to_smm_config(),
            );

        let start = this.slice_buffer_offset;
        let limit = select(
            this.slice_buffer_offset < window.size,
            this.slice_buffer_offset,
            window.size,
        );
        let end = start + Min::min(window.size - limit, this.slice_length);

        let src = window.slice.slice(start, end);
        let mut dest = destination.slice_mut(start, end);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.num_slices % this.unit_count == 0) {
            CM::memcpy_async(mechanism, &src.try_cast_unchecked(), &mut dest);
        } else {
            if nth_slice_in_buffer < this.num_slices {
                CM::memcpy_async(mechanism, &src.try_cast_unchecked(), &mut dest);
            }
        };
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
