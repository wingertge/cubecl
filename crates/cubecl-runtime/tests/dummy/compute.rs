use super::DummyServer;
use cubecl_common::CubeDim;
use cubecl_runtime::storage::BytesStorage;
use cubecl_runtime::tune::LocalTuner;
use cubecl_runtime::{ComputeRuntime, DeviceProperties, TimeMeasurement};
use cubecl_runtime::{channel::MutexComputeChannel, memory_management::HardwareProperties};
use cubecl_runtime::{client::ComputeClient, tune::TunableSet};
use cubecl_runtime::{
    memory_management::{MemoryConfiguration, MemoryDeviceProperties, MemoryManagement},
    server::Binding,
};

/// The dummy device.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyDevice;

pub type DummyChannel = MutexComputeChannel<DummyServer>;
pub type DummyClient = ComputeClient<DummyServer, DummyChannel>;

static RUNTIME: ComputeRuntime<DummyDevice, DummyServer, DummyChannel> = ComputeRuntime::new();
pub static TUNER_DEVICE_ID: &str = "dummy-device";
pub static TUNER_PREFIX: &str = "dummy-tests";
pub static TEST_TUNER: LocalTuner<String, String> = LocalTuner::new(TUNER_PREFIX);

pub fn autotune_execute(
    client: &ComputeClient<DummyServer, MutexComputeChannel<DummyServer>>,
    set: &TunableSet<String, Vec<Binding>, ()>,
    inputs: Vec<Binding>,
) {
    TEST_TUNER.execute(&TUNER_DEVICE_ID.to_string(), client, set, inputs)
}

pub fn init_client() -> ComputeClient<DummyServer, MutexComputeChannel<DummyServer>> {
    let storage = BytesStorage::default();
    let mem_properties = MemoryDeviceProperties {
        max_page_size: 1024 * 1024 * 512,
        alignment: 32,
    };
    let topology = HardwareProperties {
        plane_size_min: 32,
        plane_size_max: 32,
        max_bindings: 32,
        max_shared_memory_size: 48000,
        max_cube_count: CubeDim::new_3d(u16::MAX as u32, u16::MAX as u32, u16::MAX as u32),
        max_units_per_cube: 1024,
        max_cube_dim: CubeDim::new_3d(1024, 1024, 64),
        num_streaming_multiprocessors: None,
        num_tensor_cores: None,
    };
    let memory_management = MemoryManagement::from_configuration(
        storage,
        &mem_properties,
        MemoryConfiguration::default(),
    );
    let server = DummyServer::new(memory_management);
    let channel = MutexComputeChannel::new(server);
    ComputeClient::new(
        channel,
        DeviceProperties::new(&[], mem_properties, topology, TimeMeasurement::System),
        (),
    )
}

pub fn client(device: &DummyDevice) -> DummyClient {
    RUNTIME.client(device, init_client)
}
