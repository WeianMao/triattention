#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <stdlib.h>
#include <string>
// #include <sys/stat.h>
// #include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace turbomind {

typedef enum datatype_enum
{
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_BYTES,
    TYPE_BF16
} DataType;

typedef enum memorytype_enum
{
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

struct Tensor {
    MemoryType          where;
    DataType            type;
    std::vector<size_t> shape;
    const void*         data;

    Tensor(): where(MEMORY_CPU), type(TYPE_INVALID), shape({}), data(nullptr) {}
    Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void* _data):
        where(_where), type(_type), shape(_shape), data(_data)
    {
    }

    size_t size() const
    {
        if (data == nullptr || shape.size() == 0) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    }

    size_t sizeBytes() const
    {
        return size() * typeSize();
    }

    size_t typeSize() const
    {
        static const std::unordered_map<DataType, size_t> type_map{{TYPE_BOOL, sizeof(bool)},
                                                                   {TYPE_BYTES, sizeof(char)},
                                                                   {TYPE_UINT8, sizeof(uint8_t)},
                                                                   {TYPE_UINT16, sizeof(uint16_t)},
                                                                   {TYPE_UINT32, sizeof(uint32_t)},
                                                                   {TYPE_UINT64, sizeof(uint64_t)},
                                                                   {TYPE_INT8, sizeof(int8_t)},
                                                                   {TYPE_INT16, sizeof(int16_t)},
                                                                   {TYPE_INT32, sizeof(int32_t)},
                                                                   {TYPE_INT64, sizeof(int64_t)},
#ifdef ENABLE_BF16
                                                                   {TYPE_BF16, sizeof(__nv_bfloat16)},
#endif
                                                                   {TYPE_FP16, sizeof(half)},
                                                                   {TYPE_FP32, sizeof(float)},
                                                                   {TYPE_FP64, sizeof(double)}};
        return type_map.at(type);
    }
};
}  // namespace turbomind
