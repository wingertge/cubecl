use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{Dimensions, Offsets, SharedMemories},
    config::CmmaConfig,
};

#[cube]
pub(crate) fn load_to_shared_memories<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    offsets: Offsets,
    shared: SharedMemories<F, FC>,
    config: Comptime<CmmaConfig>,
    dims: Dimensions,
) {
    load_lhs(lhs, offsets, shared.lhs, config, dims);
    load_rhs(rhs, offsets, shared.rhs, config, dims);
}

#[cube]
fn load_lhs<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    offsets: Offsets,
    mut shared_lhs: SharedMemory<FC>,
    config: Comptime<CmmaConfig>,
    dims: Dimensions,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let lhs_stride = dims.k;

    let tile_size = Comptime::map(config, |c| c.tile_size);

    let num_tiles_in_k = Comptime::runtime(block_size_k / tile_size); // 2
    let tile_size_r = Comptime::runtime(tile_size); // 16

    let tensor_vec = Comptime::vectorization(lhs);
    let tensor_vec_r = Comptime::runtime(tensor_vec); // 4
    let lhs_sm_stride = Comptime::runtime(block_size_m); // 64
    let lhs_sm_stride_vec = Comptime::runtime(block_size_m / tensor_vec); // Even if not really vectorized, because write tensor_vec_r values

    let subcube_dim = UInt::new(32);
    let within_tile_row_offset = subcube_dim / tensor_vec_r; // assuming subcube_dim is 32 -> 8
    let within_sm_row_offset = subcube_dim / Comptime::runtime(tile_size); // assuming subcube_dim is 32 -> 2
    let subcube_id = UNIT_POS_Y;
    let id_within_subcube = UNIT_POS_X;

    // There are two because 32 / 16. TODO generalize
    let unit_read_row_0 = id_within_subcube / tensor_vec_r;
    let unit_read_row_1 = unit_read_row_0 + within_tile_row_offset;
    let unit_read_col = id_within_subcube % tensor_vec_r;

    // LHS
    let lhs_tile_row = subcube_id / num_tiles_in_k;
    let lhs_tile_col = subcube_id % num_tiles_in_k;

    let lhs_offset = offsets.batch_lhs + offsets.k + offsets.cube_row * lhs_stride;
    let lhs_read_tile_offset =
        lhs_offset + lhs_tile_row * tile_size_r * lhs_stride + lhs_tile_col * tile_size_r;

    let lhs_read_pos_0 =
        lhs_read_tile_offset + unit_read_row_0 * lhs_stride + unit_read_col * tensor_vec_r;
    let lhs_read_pos_1 =
        lhs_read_tile_offset + unit_read_row_1 * lhs_stride + unit_read_col * tensor_vec_r;
    let lhs_sm_row_offset = Comptime::runtime(tile_size * tile_size / block_size_m) * subcube_id; // 4

    let lhs_write_row_0 = id_within_subcube / lhs_sm_stride_vec;
    let lhs_write_row_1 = lhs_write_row_0 + within_sm_row_offset;
    let lhs_write_col = id_within_subcube % lhs_sm_stride_vec;

    let lhs_write_pos_0 =
        (lhs_sm_row_offset + lhs_write_row_0) * lhs_sm_stride + lhs_write_col * tensor_vec_r;
    let lhs_write_pos_1 =
        (lhs_sm_row_offset + lhs_write_row_1) * lhs_sm_stride + lhs_write_col * tensor_vec_r;

    let a = lhs[lhs_read_pos_0 / tensor_vec_r];
    let b = lhs[lhs_read_pos_1 / tensor_vec_r];
    for i in range(0u32, 4u32, Comptime::new(true)) {
        shared_lhs[lhs_write_pos_0 + i] = FC::cast_from(a[i]);
    }
    for i in range(0u32, 4u32, Comptime::new(true)) {
        shared_lhs[lhs_write_pos_1 + i] = FC::cast_from(b[i]);
    }
}

#[cube]
fn load_rhs<F: Float, FC: Float>(
    rhs: &Tensor<F>,
    offsets: Offsets,
    mut shared_rhs: SharedMemory<FC>,
    config: Comptime<CmmaConfig>,
    dims: Dimensions,
) {
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let rhs_sm_stride = Comptime::runtime(block_size_n); // 64
    let rhs_stride = dims.n;

    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let tensor_vec = Comptime::vectorization(rhs);

    let num_tiles_in_k = Comptime::runtime(block_size_k / tile_size); // 2
    let tile_size_r = Comptime::runtime(tile_size); // 16
    let tensor_vec_r = Comptime::runtime(tensor_vec); // 4

    let subcube_dim = UInt::new(32);
    let within_tile_row_offset = subcube_dim / tensor_vec_r; // assuming subcube_dim is 32 -> 8
    let within_sm_row_offset = subcube_dim / Comptime::runtime(tile_size); // assuming subcube_dim is 32 -> 2
    let subcube_id = UNIT_POS_Y;
    let id_within_subcube = UNIT_POS_X;

    // There are two because 32 / 16. TODO generalize
    let unit_read_row_0 = id_within_subcube / tensor_vec_r;
    let unit_read_row_1 = unit_read_row_0 + within_tile_row_offset;
    let unit_read_col = id_within_subcube % tensor_vec_r;

    let rhs_tile_row = subcube_id % num_tiles_in_k;
    let rhs_tile_col = subcube_id / num_tiles_in_k;

    let rhs_offset = offsets.batch_rhs + offsets.k * rhs_stride + offsets.cube_col;
    let rhs_read_tile_offset =
        rhs_offset + rhs_tile_row * tile_size_r * rhs_stride + rhs_tile_col * tile_size_r;

    let rhs_read_pos_0 =
        rhs_read_tile_offset + unit_read_row_0 * rhs_stride + unit_read_col * tensor_vec_r;
    let rhs_read_pos_1 =
        rhs_read_tile_offset + unit_read_row_1 * rhs_stride + unit_read_col * tensor_vec_r;

    let rhs_sm_row_offset = Comptime::runtime(tile_size * tile_size / block_size_n) * subcube_id; // 4

    let rhs_write_row_0 = id_within_subcube / rhs_sm_stride;
    let rhs_write_row_1 = rhs_write_row_0 + within_sm_row_offset;
    let rhs_write_col = id_within_subcube % rhs_sm_stride;

    let rhs_write_pos_0 =
        (rhs_sm_row_offset + rhs_write_row_0) * rhs_sm_stride + rhs_write_col * tensor_vec_r;
    let rhs_write_pos_1 =
        (rhs_sm_row_offset + rhs_write_row_1) * rhs_sm_stride + rhs_write_col * tensor_vec_r;

    // TODO bug: because c is used in closures above, can't be used here
    let e = rhs[rhs_read_pos_0 / tensor_vec_r];
    let d = rhs[rhs_read_pos_1 / tensor_vec_r];

    for i in range(0u32, 4u32, Comptime::new(true)) {
        shared_rhs[rhs_write_pos_0 + i] = FC::cast_from(e[i]);
    }
    for i in range(0u32, 4u32, Comptime::new(true)) {
        shared_rhs[rhs_write_pos_1 + i] = FC::cast_from(d[i]);
    }
}

#[cfg(feature = "export_tests")]
/// Compute loop exported tests
pub mod tests {

    use crate::matmul::{
        cmma::base::{DimensionsExpand, OffsetsExpand},
        test_utils::{assert_equals_range, create_empty, range_tensor},
    };

    use super::*;

    #[cube(launch)]
    fn load_lhs_test<F: Float>(
        lhs_tensor: &Tensor<F>,
        lhs_sm_arr: &mut Array<F>,
        k_offset: UInt,
        config: Comptime<CmmaConfig>,
    ) {
        let offsets = Offsets {
            batch_lhs: UInt::new(0),
            batch_rhs: UInt::new(0),
            batch_out: UInt::new(0),
            cube_row: UInt::new(0),
            cube_col: UInt::new(0),
            k: k_offset,
        };

        let mut lhs_sm = SharedMemory::<F>::new(2048);
        for i in range(0u32, 2048u32, Comptime::new(false)) {
            lhs_sm[i] = lhs_sm_arr[i];
        }

        let dims = Dimensions {
            m: UInt::new(64),
            k: UInt::new(64),
            n: UInt::new(64),
        };

        load_lhs(lhs_tensor, offsets, lhs_sm, config, dims);

        for i in range(0u32, 2048u32, Comptime::new(false)) {
            lhs_sm_arr[i] = lhs_sm[i];
        }
    }

    #[cube(launch)]
    fn load_rhs_test<F: Float>(
        rhs_tensor: &Tensor<F>,
        rhs_sm_arr: &mut Array<F>,
        k_offset: UInt,
        config: Comptime<CmmaConfig>,
    ) {
        let offsets = Offsets {
            batch_lhs: UInt::new(0),
            batch_rhs: UInt::new(0),
            batch_out: UInt::new(0),
            cube_row: UInt::new(0),
            cube_col: UInt::new(0),
            k: k_offset,
        };

        let mut rhs_sm = SharedMemory::<F>::new(2048);
        for i in range(0u32, 2048u32, Comptime::new(false)) {
            rhs_sm[i] = rhs_sm_arr[i];
        }

        let dims = Dimensions {
            m: UInt::new(64),
            k: UInt::new(64),
            n: UInt::new(64),
        };

        load_rhs(rhs_tensor, offsets, rhs_sm, config, dims);

        for i in range(0u32, 2048u32, Comptime::new(false)) {
            rhs_sm_arr[i] = rhs_sm[i];
        }
    }

    /// Exported test
    pub fn load_shared_memory_lhs_unit_test<R: Runtime>(device: &R::Device) {
        let lhs_tensor = range_tensor::<R>(64, 64, device);
        let lhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_lhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &lhs_tensor.handle,
                &lhs_tensor.strides,
                &lhs_tensor.shape,
            ),
            ArrayArg::new(&lhs_sm, 64 * 32),
            ScalarArg::new(0),
            config,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 512.0, 513.0, 514.0, 515.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ];
        assert_equals_range::<R>(lhs_sm, expected, 0..256, device);
    }

    /// Exported test
    pub fn load_shared_memory_rhs_unit_test<R: Runtime>(device: &R::Device) {
        let rhs_tensor = range_tensor::<R>(64, 64, device);
        let rhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_rhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &rhs_tensor.handle,
                &rhs_tensor.strides,
                &rhs_tensor.shape,
            ),
            ArrayArg::new(&rhs_sm, 64 * 32),
            ScalarArg::new(0),
            config,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 512.0, 513.0, 514.0, 515.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ];
        assert_equals_range::<R>(rhs_sm, expected, 0..256, device);
    }

    /// Exported test
    pub fn load_shared_memory_lhs_warp_test<R: Runtime>(device: &R::Device) {
        let lhs_tensor = range_tensor::<R>(64, 64, device);
        let lhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(32, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_lhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &lhs_tensor.handle,
                &lhs_tensor.strides,
                &lhs_tensor.shape,
            ),
            ArrayArg::new(&lhs_sm, 64 * 32),
            ScalarArg::new(0),
            config,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0,
            78.0, 79.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0,
            138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0,
            198.0, 199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 256.0, 257.0,
            258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 264.0, 265.0, 266.0, 267.0, 268.0, 269.0,
            270.0, 271.0, 320.0, 321.0, 322.0, 323.0, 324.0, 325.0, 326.0, 327.0, 328.0, 329.0,
            330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 384.0, 385.0, 386.0, 387.0, 388.0, 389.0,
            390.0, 391.0, 392.0, 393.0, 394.0, 395.0, 396.0, 397.0, 398.0, 399.0, 448.0, 449.0,
            450.0, 451.0, 452.0, 453.0, 454.0, 455.0, 456.0, 457.0, 458.0, 459.0, 460.0, 461.0,
            462.0, 463.0, 512.0, 513.0, 514.0, 515.0, 516.0, 517.0, 518.0, 519.0, 520.0, 521.0,
            522.0, 523.0, 524.0, 525.0, 526.0, 527.0, 576.0, 577.0, 578.0, 579.0, 580.0, 581.0,
            582.0, 583.0, 584.0, 585.0, 586.0, 587.0, 588.0, 589.0, 590.0, 591.0, 640.0, 641.0,
            642.0, 643.0, 644.0, 645.0, 646.0, 647.0, 648.0, 649.0, 650.0, 651.0, 652.0, 653.0,
            654.0, 655.0, 704.0, 705.0, 706.0, 707.0, 708.0, 709.0, 710.0, 711.0, 712.0, 713.0,
            714.0, 715.0, 716.0, 717.0, 718.0, 719.0, 768.0, 769.0, 770.0, 771.0, 772.0, 773.0,
            774.0, 775.0, 776.0, 777.0, 778.0, 779.0, 780.0, 781.0, 782.0, 783.0, 832.0, 833.0,
            834.0, 835.0, 836.0, 837.0, 838.0, 839.0, 840.0, 841.0, 842.0, 843.0, 844.0, 845.0,
            846.0, 847.0, 896.0, 897.0, 898.0, 899.0, 900.0, 901.0, 902.0, 903.0, 904.0, 905.0,
            906.0, 907.0, 908.0, 909.0, 910.0, 911.0, 960.0, 961.0, 962.0, 963.0, 964.0, 965.0,
            966.0, 967.0, 968.0, 969.0, 970.0, 971.0, 972.0, 973.0, 974.0, 975.0,
        ];
        assert_equals_range::<R>(lhs_sm, expected, 0..256, device);
    }

    /// Exported test
    pub fn load_shared_memory_rhs_warp_test<R: Runtime>(device: &R::Device) {
        let rhs_tensor = range_tensor::<R>(64, 64, device);
        let rhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(32, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_rhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &rhs_tensor.handle,
                &rhs_tensor.strides,
                &rhs_tensor.shape,
            ),
            ArrayArg::new(&rhs_sm, 64 * 32),
            ScalarArg::new(0),
            config,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0,
            78.0, 79.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0,
            138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0,
            198.0, 199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 256.0, 257.0,
            258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 264.0, 265.0, 266.0, 267.0, 268.0, 269.0,
            270.0, 271.0, 320.0, 321.0, 322.0, 323.0, 324.0, 325.0, 326.0, 327.0, 328.0, 329.0,
            330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 384.0, 385.0, 386.0, 387.0, 388.0, 389.0,
            390.0, 391.0, 392.0, 393.0, 394.0, 395.0, 396.0, 397.0, 398.0, 399.0, 448.0, 449.0,
            450.0, 451.0, 452.0, 453.0, 454.0, 455.0, 456.0, 457.0, 458.0, 459.0, 460.0, 461.0,
            462.0, 463.0, 512.0, 513.0, 514.0, 515.0, 516.0, 517.0, 518.0, 519.0, 520.0, 521.0,
            522.0, 523.0, 524.0, 525.0, 526.0, 527.0, 576.0, 577.0, 578.0, 579.0, 580.0, 581.0,
            582.0, 583.0, 584.0, 585.0, 586.0, 587.0, 588.0, 589.0, 590.0, 591.0, 640.0, 641.0,
            642.0, 643.0, 644.0, 645.0, 646.0, 647.0, 648.0, 649.0, 650.0, 651.0, 652.0, 653.0,
            654.0, 655.0, 704.0, 705.0, 706.0, 707.0, 708.0, 709.0, 710.0, 711.0, 712.0, 713.0,
            714.0, 715.0, 716.0, 717.0, 718.0, 719.0, 768.0, 769.0, 770.0, 771.0, 772.0, 773.0,
            774.0, 775.0, 776.0, 777.0, 778.0, 779.0, 780.0, 781.0, 782.0, 783.0, 832.0, 833.0,
            834.0, 835.0, 836.0, 837.0, 838.0, 839.0, 840.0, 841.0, 842.0, 843.0, 844.0, 845.0,
            846.0, 847.0, 896.0, 897.0, 898.0, 899.0, 900.0, 901.0, 902.0, 903.0, 904.0, 905.0,
            906.0, 907.0, 908.0, 909.0, 910.0, 911.0, 960.0, 961.0, 962.0, 963.0, 964.0, 965.0,
            966.0, 967.0, 968.0, 969.0, 970.0, 971.0, 972.0, 973.0, 974.0, 975.0,
        ];
        assert_equals_range::<R>(rhs_sm, expected, 0..256, device);
    }

    /// Exported test
    pub fn load_shared_memory_lhs_second_warp_test<R: Runtime>(device: &R::Device) {
        let lhs_tensor = range_tensor::<R>(64, 64, device);
        let lhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(32, 2, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_lhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &lhs_tensor.handle,
                &lhs_tensor.strides,
                &lhs_tensor.shape,
            ),
            ArrayArg::new(&lhs_sm, 64 * 32),
            ScalarArg::new(0),
            config,
        );

        let expected = &[
            16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 80.,
            81., 82., 83., 84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 144., 145.,
            146., 147., 148., 149., 150., 151., 152., 153., 154., 155., 156., 157., 158., 159.,
            208., 209., 210., 211., 212., 213., 214., 215., 216., 217., 218., 219., 220., 221.,
            222., 223., 272., 273., 274., 275., 276., 277., 278., 279., 280., 281., 282., 283.,
            284., 285., 286., 287., 336., 337., 338., 339., 340., 341., 342., 343., 344., 345.,
            346., 347., 348., 349., 350., 351., 400., 401., 402., 403., 404., 405., 406., 407.,
            408., 409., 410., 411., 412., 413., 414., 415., 464., 465., 466., 467., 468., 469.,
            470., 471., 472., 473., 474., 475., 476., 477., 478., 479., 528., 529., 530., 531.,
            532., 533., 534., 535., 536., 537., 538., 539., 540., 541., 542., 543., 592., 593.,
            594., 595., 596., 597., 598., 599., 600., 601., 602., 603., 604., 605., 606., 607.,
            656., 657., 658., 659., 660., 661., 662., 663., 664., 665., 666., 667., 668., 669.,
            670., 671., 720., 721., 722., 723., 724., 725., 726., 727., 728., 729., 730., 731.,
            732., 733., 734., 735., 784., 785., 786., 787., 788., 789., 790., 791., 792., 793.,
            794., 795., 796., 797., 798., 799., 848., 849., 850., 851., 852., 853., 854., 855.,
            856., 857., 858., 859., 860., 861., 862., 863., 912., 913., 914., 915., 916., 917.,
            918., 919., 920., 921., 922., 923., 924., 925., 926., 927., 976., 977., 978., 979.,
            980., 981., 982., 983., 984., 985., 986., 987., 988., 989., 990., 991.,
        ];

        // We are testing second warp
        assert_equals_range::<R>(lhs_sm, expected, 256..512, device);
    }

    /// Exported test
    pub fn load_shared_memory_rhs_second_warp_test<R: Runtime>(device: &R::Device) {
        let rhs_tensor = range_tensor::<R>(64, 64, device);
        let rhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(32, 2, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_rhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &rhs_tensor.handle,
                &rhs_tensor.strides,
                &rhs_tensor.shape,
            ),
            ArrayArg::new(&rhs_sm, 64 * 32),
            ScalarArg::new(0),
            config,
        );

        let expected = &[
            1024., 1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035.,
            1036., 1037., 1038., 1039., 1088., 1089., 1090., 1091., 1092., 1093., 1094., 1095.,
            1096., 1097., 1098., 1099., 1100., 1101., 1102., 1103., 1152., 1153., 1154., 1155.,
            1156., 1157., 1158., 1159., 1160., 1161., 1162., 1163., 1164., 1165., 1166., 1167.,
            1216., 1217., 1218., 1219., 1220., 1221., 1222., 1223., 1224., 1225., 1226., 1227.,
            1228., 1229., 1230., 1231., 1280., 1281., 1282., 1283., 1284., 1285., 1286., 1287.,
            1288., 1289., 1290., 1291., 1292., 1293., 1294., 1295., 1344., 1345., 1346., 1347.,
            1348., 1349., 1350., 1351., 1352., 1353., 1354., 1355., 1356., 1357., 1358., 1359.,
            1408., 1409., 1410., 1411., 1412., 1413., 1414., 1415., 1416., 1417., 1418., 1419.,
            1420., 1421., 1422., 1423., 1472., 1473., 1474., 1475., 1476., 1477., 1478., 1479.,
            1480., 1481., 1482., 1483., 1484., 1485., 1486., 1487., 1536., 1537., 1538., 1539.,
            1540., 1541., 1542., 1543., 1544., 1545., 1546., 1547., 1548., 1549., 1550., 1551.,
            1600., 1601., 1602., 1603., 1604., 1605., 1606., 1607., 1608., 1609., 1610., 1611.,
            1612., 1613., 1614., 1615., 1664., 1665., 1666., 1667., 1668., 1669., 1670., 1671.,
            1672., 1673., 1674., 1675., 1676., 1677., 1678., 1679., 1728., 1729., 1730., 1731.,
            1732., 1733., 1734., 1735., 1736., 1737., 1738., 1739., 1740., 1741., 1742., 1743.,
            1792., 1793., 1794., 1795., 1796., 1797., 1798., 1799., 1800., 1801., 1802., 1803.,
            1804., 1805., 1806., 1807., 1856., 1857., 1858., 1859., 1860., 1861., 1862., 1863.,
            1864., 1865., 1866., 1867., 1868., 1869., 1870., 1871., 1920., 1921., 1922., 1923.,
            1924., 1925., 1926., 1927., 1928., 1929., 1930., 1931., 1932., 1933., 1934., 1935.,
            1984., 1985., 1986., 1987., 1988., 1989., 1990., 1991., 1992., 1993., 1994., 1995.,
            1996., 1997., 1998., 1999.,
        ];

        // We are testing second warp
        assert_equals_range::<R>(rhs_sm, expected, 256..512, device);
    }

    /// Exported test
    pub fn load_shared_memory_lhs_third_warp_test<R: Runtime>(device: &R::Device) {
        let lhs_tensor = range_tensor::<R>(64, 64, device);
        let lhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(32, 3, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_lhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &lhs_tensor.handle,
                &lhs_tensor.strides,
                &lhs_tensor.shape,
            ),
            ArrayArg::new(&lhs_sm, 64 * 32),
            ScalarArg::new(0),
            config,
        );

        let expected = &[
            1024., 1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035.,
            1036., 1037., 1038., 1039., 1088., 1089., 1090., 1091., 1092., 1093., 1094., 1095.,
            1096., 1097., 1098., 1099., 1100., 1101., 1102., 1103., 1152., 1153., 1154., 1155.,
            1156., 1157., 1158., 1159., 1160., 1161., 1162., 1163., 1164., 1165., 1166., 1167.,
            1216., 1217., 1218., 1219., 1220., 1221., 1222., 1223., 1224., 1225., 1226., 1227.,
            1228., 1229., 1230., 1231., 1280., 1281., 1282., 1283., 1284., 1285., 1286., 1287.,
            1288., 1289., 1290., 1291., 1292., 1293., 1294., 1295., 1344., 1345., 1346., 1347.,
            1348., 1349., 1350., 1351., 1352., 1353., 1354., 1355., 1356., 1357., 1358., 1359.,
            1408., 1409., 1410., 1411., 1412., 1413., 1414., 1415., 1416., 1417., 1418., 1419.,
            1420., 1421., 1422., 1423., 1472., 1473., 1474., 1475., 1476., 1477., 1478., 1479.,
            1480., 1481., 1482., 1483., 1484., 1485., 1486., 1487., 1536., 1537., 1538., 1539.,
            1540., 1541., 1542., 1543., 1544., 1545., 1546., 1547., 1548., 1549., 1550., 1551.,
            1600., 1601., 1602., 1603., 1604., 1605., 1606., 1607., 1608., 1609., 1610., 1611.,
            1612., 1613., 1614., 1615., 1664., 1665., 1666., 1667., 1668., 1669., 1670., 1671.,
            1672., 1673., 1674., 1675., 1676., 1677., 1678., 1679., 1728., 1729., 1730., 1731.,
            1732., 1733., 1734., 1735., 1736., 1737., 1738., 1739., 1740., 1741., 1742., 1743.,
            1792., 1793., 1794., 1795., 1796., 1797., 1798., 1799., 1800., 1801., 1802., 1803.,
            1804., 1805., 1806., 1807., 1856., 1857., 1858., 1859., 1860., 1861., 1862., 1863.,
            1864., 1865., 1866., 1867., 1868., 1869., 1870., 1871., 1920., 1921., 1922., 1923.,
            1924., 1925., 1926., 1927., 1928., 1929., 1930., 1931., 1932., 1933., 1934., 1935.,
            1984., 1985., 1986., 1987., 1988., 1989., 1990., 1991., 1992., 1993., 1994., 1995.,
            1996., 1997., 1998., 1999.,
        ];

        // We are testing second warp
        assert_equals_range::<R>(lhs_sm, expected, 512..768, device);
    }

    /// Exported test
    pub fn load_shared_memory_rhs_third_warp_test<R: Runtime>(device: &R::Device) {
        let rhs_tensor = range_tensor::<R>(64, 64, device);
        let rhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(32, 3, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_rhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &rhs_tensor.handle,
                &rhs_tensor.strides,
                &rhs_tensor.shape,
            ),
            ArrayArg::new(&rhs_sm, 64 * 32),
            ScalarArg::new(0),
            config,
        );

        let expected = &[
            16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 80.,
            81., 82., 83., 84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 144., 145.,
            146., 147., 148., 149., 150., 151., 152., 153., 154., 155., 156., 157., 158., 159.,
            208., 209., 210., 211., 212., 213., 214., 215., 216., 217., 218., 219., 220., 221.,
            222., 223., 272., 273., 274., 275., 276., 277., 278., 279., 280., 281., 282., 283.,
            284., 285., 286., 287., 336., 337., 338., 339., 340., 341., 342., 343., 344., 345.,
            346., 347., 348., 349., 350., 351., 400., 401., 402., 403., 404., 405., 406., 407.,
            408., 409., 410., 411., 412., 413., 414., 415., 464., 465., 466., 467., 468., 469.,
            470., 471., 472., 473., 474., 475., 476., 477., 478., 479., 528., 529., 530., 531.,
            532., 533., 534., 535., 536., 537., 538., 539., 540., 541., 542., 543., 592., 593.,
            594., 595., 596., 597., 598., 599., 600., 601., 602., 603., 604., 605., 606., 607.,
            656., 657., 658., 659., 660., 661., 662., 663., 664., 665., 666., 667., 668., 669.,
            670., 671., 720., 721., 722., 723., 724., 725., 726., 727., 728., 729., 730., 731.,
            732., 733., 734., 735., 784., 785., 786., 787., 788., 789., 790., 791., 792., 793.,
            794., 795., 796., 797., 798., 799., 848., 849., 850., 851., 852., 853., 854., 855.,
            856., 857., 858., 859., 860., 861., 862., 863., 912., 913., 914., 915., 916., 917.,
            918., 919., 920., 921., 922., 923., 924., 925., 926., 927., 976., 977., 978., 979.,
            980., 981., 982., 983., 984., 985., 986., 987., 988., 989., 990., 991.,
        ];

        // We are testing second warp
        assert_equals_range::<R>(rhs_sm, expected, 512..768, device);
    }

    /// Exported test
    pub fn load_shared_memory_lhs_k_offset_test<R: Runtime>(device: &R::Device) {
        let lhs_tensor = range_tensor::<R>(64, 64, device);
        let lhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(32, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_lhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &lhs_tensor.handle,
                &lhs_tensor.strides,
                &lhs_tensor.shape,
            ),
            ArrayArg::new(&lhs_sm, 64 * 32),
            ScalarArg::new(32),
            config,
        );

        let expected = &[
            32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 96.,
            97., 98., 99., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 110., 111.,
            160., 161., 162., 163., 164., 165., 166., 167., 168., 169., 170., 171., 172., 173.,
            174., 175., 224., 225., 226., 227., 228., 229., 230., 231., 232., 233., 234., 235.,
            236., 237., 238., 239., 288., 289., 290., 291., 292., 293., 294., 295., 296., 297.,
            298., 299., 300., 301., 302., 303., 352., 353., 354., 355., 356., 357., 358., 359.,
            360., 361., 362., 363., 364., 365., 366., 367., 416., 417., 418., 419., 420., 421.,
            422., 423., 424., 425., 426., 427., 428., 429., 430., 431., 480., 481., 482., 483.,
            484., 485., 486., 487., 488., 489., 490., 491., 492., 493., 494., 495., 544., 545.,
            546., 547., 548., 549., 550., 551., 552., 553., 554., 555., 556., 557., 558., 559.,
            608., 609., 610., 611., 612., 613., 614., 615., 616., 617., 618., 619., 620., 621.,
            622., 623., 672., 673., 674., 675., 676., 677., 678., 679., 680., 681., 682., 683.,
            684., 685., 686., 687., 736., 737., 738., 739., 740., 741., 742., 743., 744., 745.,
            746., 747., 748., 749., 750., 751., 800., 801., 802., 803., 804., 805., 806., 807.,
            808., 809., 810., 811., 812., 813., 814., 815., 864., 865., 866., 867., 868., 869.,
            870., 871., 872., 873., 874., 875., 876., 877., 878., 879., 928., 929., 930., 931.,
            932., 933., 934., 935., 936., 937., 938., 939., 940., 941., 942., 943., 992., 993.,
            994., 995., 996., 997., 998., 999., 1000., 1001., 1002., 1003., 1004., 1005., 1006.,
            1007.,
        ];

        // We are testing second warp
        assert_equals_range::<R>(lhs_sm, expected, 0..256, device);
    }

    /// Exported test
    pub fn load_shared_memory_rhs_k_offset_test<R: Runtime>(device: &R::Device) {
        let rhs_tensor = range_tensor::<R>(64, 64, device);
        let rhs_sm = create_empty::<R>(32, 64, device);
        let cube_dim = CubeDim::new(32, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        load_rhs_test::launch::<F32, R>(
            R::client(device),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                4,
                &rhs_tensor.handle,
                &rhs_tensor.strides,
                &rhs_tensor.shape,
            ),
            ArrayArg::new(&rhs_sm, 64 * 32),
            ScalarArg::new(32),
            config,
        );

        let expected = &[
            2048., 2049., 2050., 2051., 2052., 2053., 2054., 2055., 2056., 2057., 2058., 2059.,
            2060., 2061., 2062., 2063., 2112., 2113., 2114., 2115., 2116., 2117., 2118., 2119.,
            2120., 2121., 2122., 2123., 2124., 2125., 2126., 2127., 2176., 2177., 2178., 2179.,
            2180., 2181., 2182., 2183., 2184., 2185., 2186., 2187., 2188., 2189., 2190., 2191.,
            2240., 2241., 2242., 2243., 2244., 2245., 2246., 2247., 2248., 2249., 2250., 2251.,
            2252., 2253., 2254., 2255., 2304., 2305., 2306., 2307., 2308., 2309., 2310., 2311.,
            2312., 2313., 2314., 2315., 2316., 2317., 2318., 2319., 2368., 2369., 2370., 2371.,
            2372., 2373., 2374., 2375., 2376., 2377., 2378., 2379., 2380., 2381., 2382., 2383.,
            2432., 2433., 2434., 2435., 2436., 2437., 2438., 2439., 2440., 2441., 2442., 2443.,
            2444., 2445., 2446., 2447., 2496., 2497., 2498., 2499., 2500., 2501., 2502., 2503.,
            2504., 2505., 2506., 2507., 2508., 2509., 2510., 2511., 2560., 2561., 2562., 2563.,
            2564., 2565., 2566., 2567., 2568., 2569., 2570., 2571., 2572., 2573., 2574., 2575.,
            2624., 2625., 2626., 2627., 2628., 2629., 2630., 2631., 2632., 2633., 2634., 2635.,
            2636., 2637., 2638., 2639., 2688., 2689., 2690., 2691., 2692., 2693., 2694., 2695.,
            2696., 2697., 2698., 2699., 2700., 2701., 2702., 2703., 2752., 2753., 2754., 2755.,
            2756., 2757., 2758., 2759., 2760., 2761., 2762., 2763., 2764., 2765., 2766., 2767.,
            2816., 2817., 2818., 2819., 2820., 2821., 2822., 2823., 2824., 2825., 2826., 2827.,
            2828., 2829., 2830., 2831., 2880., 2881., 2882., 2883., 2884., 2885., 2886., 2887.,
            2888., 2889., 2890., 2891., 2892., 2893., 2894., 2895., 2944., 2945., 2946., 2947.,
            2948., 2949., 2950., 2951., 2952., 2953., 2954., 2955., 2956., 2957., 2958., 2959.,
            3008., 3009., 3010., 3011., 3012., 3013., 3014., 3015., 3016., 3017., 3018., 3019.,
            3020., 3021., 3022., 3023.,
        ];

        // We are testing second warp
        assert_equals_range::<R>(rhs_sm, expected, 0..256, device);
    }
}
