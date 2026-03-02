#pragma once
#ifndef DEVICE_CONTEXT_H
#define DEVICE_CONTEXT_H

#include <cuda_runtime.h>
#include "gpu-utils/error_handler.cuh"

namespace device_context {

  constexpr std::size_t MAX_DEVICES = 32;

  /**
   * Properties of the device used in icicle functions.
   */
  struct DeviceContext {
    cudaStream_t& stream;  /**< Stream to use. Default value: 0. */
    std::size_t device_id; /**< Index of the currently used GPU. Default value: 0. */
    cudaMemPool_t mempool; /**< Mempool to use. Default value: 0. */
  };

  /**
   * Return default device context that corresponds to using the default stream of the first GPU
   */
  inline DeviceContext get_default_device_context() // TODO: naming convention ?
  {
    static cudaStream_t default_stream = (cudaStream_t)0;
    return DeviceContext{
      (cudaStream_t&)default_stream, // stream
      0,                             // device_id
      0,                             // mempool
    };
  }

  // checking whether a pointer is on host or device and asserts device matches provided device
  static bool is_host_ptr(const void* p, int device_id = 0)
  {
    cudaPointerAttributes attributes;
    CHK_STICKY(cudaPointerGetAttributes(&attributes, p));
    const bool is_on_host = attributes.type == cudaMemoryTypeHost ||
                            attributes.type == cudaMemoryTypeUnregistered; // unregistered is host memory
    const bool is_on_cur_device = !is_on_host && attributes.device == device_id;
    const bool is_valid_ptr = is_on_host || is_on_cur_device;
    if (!is_valid_ptr) { THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Invalid ptr"); }

    return is_on_host;
  }

  static int get_cuda_device(const void* p)
  {
    cudaPointerAttributes attributes;
    CHK_STICKY(cudaPointerGetAttributes(&attributes, p));
    const bool is_on_host = attributes.type == cudaMemoryTypeHost ||
                            attributes.type == cudaMemoryTypeUnregistered; // unregistered is host memory
    return is_on_host ? -1 : attributes.device;
  }

} // namespace device_context
#endif
