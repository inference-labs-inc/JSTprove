#if defined(DEVMODE) || defined(DEBUG)
#define INLINE_MACRO
#define UNROLL
#else
#define INLINE_MACRO __forceinline__
#define UNROLL       #pragma unroll
#endif

#define HOST_INLINE        __host__ INLINE_MACRO
#define DEVICE_INLINE      __device__ INLINE_MACRO
#define HOST_DEVICE_INLINE __host__ __device__ INLINE_MACRO
