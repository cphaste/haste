#ifndef UTIL_VECTORS_H_
#define UTIL_VECTORS_H_

#include <cuda_runtime.h>
#include <math.h>

// addition
inline float3 operator+(const float3 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x + rhs.x;
    r.y = lhs.y + rhs.y;
    r.z = lhs.z + rhs.z;
    return r;
}

// subtraction
inline float3 operator-(const float3 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x - rhs.x;
    r.y = lhs.y - rhs.y;
    r.z = lhs.z - rhs.z;
    return r;
}

// multiplication
inline float3 operator*(const float3 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x * rhs.x;
    r.y = lhs.y * rhs.y;
    r.z = lhs.z * rhs.z;
    return r;
}

// division
inline float3 operator/(const float3 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x / rhs.x;
    r.y = lhs.y / rhs.y;
    r.z = lhs.z / rhs.z;
    return r;
}

// dot product
inline float dot(const float3 &lhs, const float3 &rhs) {
    return lhs.x * rhs.x +
           lhs.y * rhs.y +
           lhs.z * rhs.z;
}

// cross product
inline float3 cross(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.y * rhs.z - rhs.y * lhs.z,
                       lhs.z * rhs.x - rhs.z * lhs.x,
                       lhs.x * rhs.y - rhs.x * lhs.y);
}

// length of a vector
inline float length(const float3 &v) {
    return sqrtf(v.x * v.x +
                 v.y * v.y +
                 v.z * v.z);
}

// distance between two points (always positive)
inline float distance(const float3 &a, const float3 &b) {
    return length(a - b);
}

// normalization
inline float3 normalize(const float3 &v) {
    float len = length(v);
    return v / make_float3(len, len, len);
}

#endif // UTIL_VECTORS_H_