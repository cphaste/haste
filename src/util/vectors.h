#ifndef UTIL_VECTORS_H_
#define UTIL_VECTORS_H_

#ifdef HASTE
    // use CUDA vectors for the GPU version
    #include <cuda_runtime.h>
#else
    // define them for the CPU version
    #include <stdint.h>

    struct float2 {
        float x, y;
    };

    struct float3 {
        float x, y, z;
    };

    struct float4 {
        float x, y, z, w;
    };

    struct int2 {
        int32_t x, y;
    };

    struct int3 {
        int32_t x, y, z;
    };

    struct int4 {
        int32_t x, y, z, w;
    };

    typedef struct float2 float2;
    typedef struct float3 float3;
    typedef struct float4 float4;
    typedef struct int2 int2;
    typedef struct int3 int3;
    typedef struct int4 int4;

    inline float2 make_float2(float x, float y) {
        float2 v;
        v.x = x;
        v.y = y;
        return v;
    }

    inline float3 make_float3(float x, float y, float z) {
        float3 v;
        v.x = x;
        v.y = y;
        v.z = z;
        return v;
    }

    inline float4 make_float4(float x, float y, float z, float w) {
        float4 v;
        v.x = x;
        v.y = y;
        v.z = z;
        v.w = w;
        return v;
    }

    inline int2 make_int2(int32_t x, int32_t y) {
        int2 v;
        v.x = x;
        v.y = y;
        return v;
    }

    inline int3 make_int3(int32_t x, int32_t y, int32_t z) {
        int3 v;
        v.x = x;
        v.y = y;
        v.z = z;
        return v;
    }

    inline int4 make_int4(int32_t x, int32_t y, int32_t z, int32_t w) {
        int4 v;
        v.x = x;
        v.y = y;
        v.z = z;
        v.w = w;
        return v;
    }
#endif

#endif // UTIL_VECTORS_H_
