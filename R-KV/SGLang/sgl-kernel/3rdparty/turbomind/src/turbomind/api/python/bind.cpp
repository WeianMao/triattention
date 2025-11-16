
#include "src/turbomind/api/python/dlpack.h"
#include "src/turbomind/api/python/linear.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/tensor.h"
#include <cuda_runtime.h>
#include <memory>
#include <numeric>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

static const char kDlTensorCapsuleName[] = "dltensor";

DLDevice getDLDevice(turbomind::Tensor& tensor)
{
    int device_id = 0;
    if (tensor.where == turbomind::MEMORY_GPU) {
        cudaPointerAttributes ptr_attr;
        cudaPointerGetAttributes(&ptr_attr, tensor.data);
        device_id = ptr_attr.device;
    }

    DLDevice device{kDLCPU, device_id};

    switch (tensor.where) {
        case turbomind::MEMORY_CPU:
            device.device_type = DLDeviceType::kDLCPU;
            break;
        case turbomind::MEMORY_CPU_PINNED:
            device.device_type = DLDeviceType::kDLCUDAHost;
            break;
        case turbomind::MEMORY_GPU:
            device.device_type = DLDeviceType::kDLCUDA;
            break;
        default:
            break;
    }

    return device;
}

DLManagedTensor* TurbomindTensorToDLManagedTensor(turbomind::Tensor& tensor)
{
    DLDevice device = getDLDevice(tensor);

    DLDataType data_type{0, 0, 1};
    switch (tensor.type) {
        case turbomind::TYPE_BOOL:
            data_type.code = DLDataTypeCode::kDLBool;
            data_type.bits = 8;
            break;
        case turbomind::TYPE_UINT8:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 8;
            break;
        case turbomind::TYPE_UINT16:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 16;
            break;
        case turbomind::TYPE_UINT32:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 32;
            break;
        case turbomind::TYPE_UINT64:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 64;
            break;
        case turbomind::TYPE_INT8:
        case turbomind::TYPE_BYTES:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 8;
            break;
        case turbomind::TYPE_INT16:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 16;
            break;
        case turbomind::TYPE_INT32:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 32;
            break;
        case turbomind::TYPE_INT64:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 64;
            break;
        case turbomind::TYPE_FP16:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 16;
            break;
        case turbomind::TYPE_FP32:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 32;
            break;
        case turbomind::TYPE_FP64:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 64;
            break;
        case turbomind::TYPE_BF16:
            data_type.code = DLDataTypeCode::kDLBfloat;
            data_type.bits = 16;
            break;
        default:
            break;
    }
    DLTensor dl_tensor{const_cast<void*>(tensor.data),
                       device,
                       (int32_t)(tensor.shape.size()),
                       data_type,
                       reinterpret_cast<int64_t*>(const_cast<size_t*>(tensor.shape.data())),
                       (int64_t*)(nullptr),
                       0};
    return new DLManagedTensor{dl_tensor, nullptr, [](DLManagedTensor* dlmt) { delete dlmt; }};
}

turbomind::MemoryType getMemoryType(DLDevice device)
{
    switch (device.device_type) {
        case DLDeviceType::kDLCUDAHost:
            return turbomind::MemoryType::MEMORY_CPU_PINNED;
        case DLDeviceType::kDLCUDA:
            return turbomind::MemoryType::MEMORY_GPU;
        case DLDeviceType::kDLCPU:
        default:
            return turbomind::MemoryType::MEMORY_CPU;
    }
}

turbomind::DataType getDataType(DLDataType data_type)
{
    switch (data_type.code) {
        case DLDataTypeCode::kDLUInt:
            switch (data_type.bits) {
                case 8:
                    return turbomind::TYPE_UINT8;
                case 16:
                    return turbomind::TYPE_UINT16;
                case 32:
                    return turbomind::TYPE_UINT32;
                case 64:
                    return turbomind::TYPE_UINT64;
                default:
                    return turbomind::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (data_type.bits) {
                case 8:
                    return turbomind::TYPE_INT8;
                case 16:
                    return turbomind::TYPE_INT16;
                case 32:
                    return turbomind::TYPE_INT32;
                case 64:
                    return turbomind::TYPE_INT64;
                default:
                    return turbomind::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (data_type.bits) {
                case 16:
                    return turbomind::TYPE_FP16;
                case 32:
                    return turbomind::TYPE_FP32;
                case 64:
                    return turbomind::TYPE_FP64;
                default:
                    return turbomind::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (data_type.bits) {
                case 16:
                    return turbomind::TYPE_BF16;
                default:
                    return turbomind::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLBool:
            return turbomind::TYPE_BOOL;
        default:
            return turbomind::TYPE_INVALID;
    }
}

std::shared_ptr<turbomind::Tensor> DLManagedTensorToTurbomindTensor(DLManagedTensor* tensor)
{
    auto& dl_tensor = tensor->dl_tensor;
    auto  where     = getMemoryType(dl_tensor.device);
    auto  dtype     = getDataType(dl_tensor.dtype);
    assert(dl_tensor.ndim > 0);
    std::vector<size_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);
    auto                data = dl_tensor.data;

    return std::make_shared<turbomind::Tensor>(where, dtype, shape, data);
}

std::shared_ptr<turbomind::Tensor> TorchTensorToTurbomindTensor(py::object obj)
{
    py::capsule      cap  = obj.attr("__dlpack__")();
    DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
    return DLManagedTensorToTurbomindTensor(dlmt);
}

PYBIND11_MODULE(_turbomind_ext, m)
{
    py::enum_<turbomind::WeightType>(m, "WeightType")
        .value("kFP32", turbomind::WeightType::kFP32)
        .value("kFP16", turbomind::WeightType::kFP16)
        .value("kFP8", turbomind::WeightType::kFP8)
        .value("kBF16", turbomind::WeightType::kBF16)
        .value("kINT8", turbomind::WeightType::kINT8)
        .value("kINT4", turbomind::WeightType::kINT4);

    // data type
    py::enum_<turbomind::DataType>(m, "DataType")
        .value("TYPE_INVALID", turbomind::DataType::TYPE_INVALID)
        .value("TYPE_BOOL", turbomind::DataType::TYPE_BOOL)
        .value("TYPE_UINT8", turbomind::DataType::TYPE_UINT8)
        .value("TYPE_UINT16", turbomind::DataType::TYPE_UINT16)
        .value("TYPE_UINT32", turbomind::DataType::TYPE_UINT32)
        .value("TYPE_UINT64", turbomind::DataType::TYPE_UINT64)
        .value("TYPE_INT8", turbomind::DataType::TYPE_INT8)
        .value("TYPE_INT16", turbomind::DataType::TYPE_INT16)
        .value("TYPE_INT32", turbomind::DataType::TYPE_INT32)
        .value("TYPE_INT64", turbomind::DataType::TYPE_INT64)
        .value("TYPE_FP16", turbomind::DataType::TYPE_FP16)
        .value("TYPE_FP32", turbomind::DataType::TYPE_FP32)
        .value("TYPE_FP64", turbomind::DataType::TYPE_FP64)
        .value("TYPE_BYTES", turbomind::DataType::TYPE_BYTES)
        .value("TYPE_BF16", turbomind::DataType::TYPE_BF16);

    // memory type
    py::enum_<turbomind::MemoryType>(m, "MemoryType")
        .value("MEMORY_CPU", turbomind::MemoryType::MEMORY_CPU)
        .value("MEMORY_CPU_PINNED", turbomind::MemoryType::MEMORY_CPU_PINNED)
        .value("MEMORY_GPU", turbomind::MemoryType::MEMORY_GPU);

    // tensor
    py::class_<turbomind::Tensor, std::shared_ptr<turbomind::Tensor>>(m, "Tensor")
        .def_readonly("where", &turbomind::Tensor::where)
        .def_readonly("type", &turbomind::Tensor::type)
        .def_readonly("shape", &turbomind::Tensor::shape)
        .def_readonly("data", &turbomind::Tensor::data)
        .def(py::init([](const turbomind::MemoryType where,
                         const turbomind::DataType   type,
                         const std::vector<size_t>&  shape,
                         const long                  data) {
            auto data_ptr = reinterpret_cast<void*>(data);
            return new turbomind::Tensor(where, type, shape, data_ptr);
        }))
        .def(
            "view",
            [](turbomind::Tensor* self, turbomind::DataType new_type) {
                return new turbomind::Tensor(self->where, new_type, self->shape, self->data);
            },
            "new_type"_a)
        .def(
            "view",
            [](turbomind::Tensor* self, std::vector<size_t> new_shape) {
                return new turbomind::Tensor(self->where, self->type, new_shape, self->data);
            },
            "new_shape"_a)
        .def(
            "copy_from",
            [](turbomind::Tensor* self, py::object obj) {
                py::capsule      cap = obj.attr("__dlpack__")();
                DLManagedTensor* dlmt =
                    static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
                auto src = DLManagedTensorToTurbomindTensor(dlmt);
                switch (self->type) {
                    case turbomind::TYPE_FP16:
                    case turbomind::TYPE_FP32:
                    case turbomind::TYPE_INT32:
                    case turbomind::TYPE_BF16: {
                        auto num_element =
                            std::accumulate(src->shape.begin(), src->shape.end(), 1LL, std::multiplies<int64_t>());
                        auto num_bytes = num_element * dlmt->dl_tensor.dtype.bits / 8;
                        turbomind::TM_CHECK(self->shape.size() == 1 && num_bytes == self->shape[0]);
                        cudaMemcpy(
                            const_cast<void*>(self->data), const_cast<void*>(src->data), num_bytes, cudaMemcpyDefault);
                        break;
                    }
                    default:
                        turbomind::TM_CHECK(0);
                }
            },
            "tensor"_a)
        .def(
            "__dlpack__",
            [](turbomind::Tensor* self, long stream) {
                DLManagedTensor* dlmt = TurbomindTensorToDLManagedTensor(*self);
                return py::capsule(dlmt, kDlTensorCapsuleName, [](PyObject* obj) {
                    DLManagedTensor* dlmt =
                        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, kDlTensorCapsuleName));
                    if (dlmt) {
                        dlmt->deleter(dlmt);
                    }
                    else {
                        // The tensor has been deleted. Clear any error from
                        // PyCapsule_GetPointer.
                        PyErr_Clear();
                    }
                });
            },
            "stream"_a = 0)
        .def("__dlpack_device__", [](turbomind::Tensor* self) {
            auto device = getDLDevice(*self);
            return std::tuple<int, int>(int(device.device_type), device.device_id);
        });
    m.def(
        "from_dlpack",
        [](py::object obj) {
            py::capsule      cap = obj.attr("__dlpack__")();
            DLManagedTensor* dlmt =
                static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
            auto ret = DLManagedTensorToTurbomindTensor(dlmt);
            return ret;
        },
        "dl_managed_tensor"_a);

    // Instantiate turbomind::Linear
    py::class_<turbomind::Linear, std::shared_ptr<turbomind::Linear>>(m, "Linear")
        .def(py::init([](size_t in_features, size_t out_features, int w_bit, int group_size) {
            return new turbomind::Linear(in_features, out_features, w_bit, group_size);
        }))
        .def("post_init",
             [](turbomind::Linear* self, py::object qweight, py::object scales, py::object qzeros, bool simt) {
                 auto _qweight = TorchTensorToTurbomindTensor(qweight);
                 auto _scales  = TorchTensorToTurbomindTensor(scales);
                 auto _qzeros  = TorchTensorToTurbomindTensor(qzeros);
                 self->post_init(_qweight, *_scales, *_qzeros, simt);
             })
        .def("forward", [](turbomind::Linear* self, py::object in, py::object out, int64_t stream_id = 0) {
            auto _in    = TorchTensorToTurbomindTensor(in);
            auto _out   = TorchTensorToTurbomindTensor(out);
            auto stream = reinterpret_cast<cudaStream_t>(stream_id);
            return self->forward(*_in, *_out, stream);
        });
}
