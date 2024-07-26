fn main() {
    #[cfg(feature = "burn-tch-cuda")]
    tch_gpu::run();
    #[cfg(feature = "cube-cuda")]
    cube_cuda::run();
}

#[cfg(feature = "burn-tch-cuda")]
mod tch_gpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice, TchTensor};
    use burn::tensor::{Distribution, Tensor};

    type Backend = LibTorch::<burn::tensor::f16>;

    pub fn run() {
        let device = LibTorchDevice::Cuda(0);
        let tensor_1: Tensor<Backend, 3> =
            Tensor::<Backend, 3>::random([12, 4096, 4096], Distribution::Default, &device);
        let tensor_2: Tensor<Backend, 3> =
            Tensor::<Backend, 3>::random([12, 4096, 4096], Distribution::Default, &device);
        let output = tensor_1.matmul(tensor_2);
    }
}

#[cfg(feature = "cube-cuda")]
mod cube_cuda {
    use std::time::Duration;

    use cubecl::cuda::{CudaDevice, CudaRuntime};
    use cubecl::linalg::{matmul, tensor::TensorHandle};
    use cubecl::prelude::*;
    use cubecl::Runtime;

    type Elem = F16;

    pub fn run() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        let num_of_batch = 12;
        let heigth = 4096;
        let width = 4096;

        let tensor_a_shape = vec![num_of_batch, heigth, width];
        let tensor_b_shape = vec![num_of_batch, heigth, width];
        let tensor_c_shape = vec![num_of_batch, heigth, width];

        std::thread::sleep(Duration::from_secs(1));

        let tensor_a = TensorHandle::<CudaRuntime, Elem>::zeros(&client, tensor_a_shape);
        let tensor_b = TensorHandle::<CudaRuntime, Elem>::zeros(&client, tensor_b_shape);
        let tensor_c = TensorHandle::<CudaRuntime, Elem>::zeros(&client, tensor_c_shape);

        client.read(tensor_c.handle.clone().binding());

        matmul::launch_ref::<CudaRuntime, Elem>(
            &client,
            tensor_a.as_ref(),
            tensor_b.as_ref(),
            tensor_c.as_ref(),
        );

        client.read(tensor_c.handle.binding());
    }
}
