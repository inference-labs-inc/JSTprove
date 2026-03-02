#pragma once
#ifndef ERR_H
#define ERR_H

#include <iostream>

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

enum class IcicleError_t {
  IcicleSuccess = 0,
  InvalidArgument = 1,
  MemoryAllocationError = 2,
  UndefinedError = 999999999,
};

std::string inline IcicleGetErrorString(IcicleError_t error)
{
  switch (error) {
  case IcicleError_t::IcicleSuccess:
    return "Success";
  case IcicleError_t::InvalidArgument:
    return "Invalid argument";
  case IcicleError_t::MemoryAllocationError:
    return "Memory allocation error";
  case IcicleError_t::UndefinedError:
    return "Undefined error occurred";
  default:
    return "Unknown error code";
  }
}

class IcicleError : public std::runtime_error
{
private:
  int errCode; // Field to store the error code

public:
  // Constructor for cudaError_t with optional message
  IcicleError(cudaError_t cudaError, const std::string& msg = "")
      : std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(cudaError)) + " " + msg),
        errCode(static_cast<int>(cudaError))
  {
  }

  // Constructor for cudaError_t with const char* message
  IcicleError(cudaError_t cudaError, const char* msg) : IcicleError(cudaError, std::string(msg)) {}

  // Constructor for IcicleError_t with optional message
  IcicleError(IcicleError_t icicleError, const std::string& msg = "")
      : std::runtime_error("Icicle Error: " + IcicleGetErrorString(icicleError) + " " + msg),
        errCode(static_cast<int>(icicleError))
  {
  }

  // Constructor for IcicleError_t with const char* message
  IcicleError(IcicleError_t icicleError, const char* msg) : IcicleError(icicleError, std::string(msg)) {}

  // Getter for errCode
  int getErrorCode() const { return errCode; }
};

// TODO: ? do{..}while(0) as per https://hownot2code.wordpress.com/2016/12/05/do-while-0-in-macros/

#define CHK_ERR(err, func, file, line) check(err, func, file, line)
#define CHK_LOG(val)                   check((val), #val, __FILE__, __LINE__)
#define CHK_VAL(val, file, line)       check((val), #val, file, line)

cudaError_t inline check(cudaError_t err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error by: " << func << " at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl << std::endl;
  }

  return err;
}

// TODO: one macro that optionally (by compile-time switch) doesn't throw
#define CHK_STICKY_NO_THROW(val) checkCudaErrorIsSticky((val), #val, __FILE__, __LINE__, false)

#define CHK_LAST_STICKY_NO_THROW()                                                                                     \
  checkCudaErrorIsSticky(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__, false)

#define CHK_LAST() checkCudaErrorIsSticky(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__)

#define CHK_STICKY(val) checkCudaErrorIsSticky((val), #val, __FILE__, __LINE__)

#define THROW_ICICLE_CUDA(val)                       throwIcicleCudaErr(val, __FUNCTION__, __FILE__, __LINE__)
#define THROW_ICICLE_CUDA_ERR(val, func, file, line) throwIcicleCudaErr(val, func, file, line)
void inline throwIcicleCudaErr(
  cudaError_t err, const char* const func, const char* const file, const int line, bool isUnrecoverable = true)
{
  // TODO: fmt::format introduced only in C++20
  std::string err_msg = (isUnrecoverable ? "!!!Unrecoverable!!! : " : "") + std::string{cudaGetErrorString(err)} +
                        " : detected by: " + func + " at: " + file + ":" + std::to_string(line) +
                        "\nThe error is reported there and may be caused by prior calls.\n";
  std::cerr << err_msg << std::endl; // TODO: Logging
  throw IcicleError{err, err_msg};
}

#define THROW_ICICLE(val, reason, func, file, line) throwIcicleErr(val, reason, func, file, line)
#define THROW_ICICLE_ERR(val, reason)               throwIcicleErr(val, reason, __FUNCTION__, __FILE__, __LINE__)
void inline throwIcicleErr(
  IcicleError_t err, const char* const reason, const char* const func, const char* const file, const int line)
{
  std::string err_msg = std::string{IcicleGetErrorString(err)} + " : by: " + func + " at: " + file + ":" +
                        std::to_string(line) + " error: " + reason;
  std::cerr << err_msg << std::endl; // TODO: Logging
  throw IcicleError{err, err_msg};
}

cudaError_t inline checkCudaErrorIsSticky(
  cudaError_t err, const char* const func, const char* const file, const int line, bool isThrowing = true)
{
  if (err != cudaSuccess) {
    // check for sticky (unrecoverable) error when the only option is to restart process
    cudaError_t err2 = cudaDeviceSynchronize();
    bool is_logged;
    if (err2 != cudaSuccess) { // we suspect sticky error
      if (err != err2) {
        is_logged = true;
        CHK_ERR(err, func, file, line);
      }
      // we are practically almost sure error is sticky
      if (isThrowing) {
        THROW_ICICLE_CUDA_ERR(err, func, file, line);
      } else {
        err = err2;
      }
    }
    if (!is_logged) CHK_ERR(err, func, file, line);
  }

  return err;
}

// most common macros to use
#define CHK_INIT_IF_RETURN()                                                                                           \
  {                                                                                                                    \
    cudaError_t err_result = CHK_LAST();                                                                               \
    if (err_result != cudaSuccess) return err_result;                                                                  \
  }

#define CHK_IF_RETURN(val)                                                                                             \
  {                                                                                                                    \
    cudaError_t err_result = CHK_STICKY(val);                                                                          \
    if (err_result != cudaSuccess) return err_result;                                                                  \
  }

#endif
