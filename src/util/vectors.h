#ifndef UTIL_VECTORS_H_
#define UTIL_VECTORS_H_

#include <cuda_runtime.h>
#include <math.h>

// addition
inline float3 operator+(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x + rhs.x,
                        lhs.y + rhs.y,
                        lhs.z + rhs.z);
}

// subtraction
inline float3 operator-(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x - rhs.x,
                        lhs.y - rhs.y,
                        lhs.z - rhs.z);
}

// multiplication
inline float3 operator*(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x * rhs.x,
                        lhs.y * rhs.y,
                        lhs.z * rhs.z);
}

// division
inline float3 operator/(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x / rhs.x,
                        lhs.y / rhs.y,
                        lhs.z / rhs.z);
}

// dot product
inline float dot(const float3 &lhs, const float3 &rhs) {
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

// cross product
inline float3 cross(const float3 &lhs, const float3 &rhs) {
    return make_float3((lhs.y * rhs.z) - (lhs.z * rhs.y),
                        (lhs.z * rhs.x) - (lhs.x * rhs.z),
                        (lhs.x * rhs.y) - (lhs.y * rhs.x));
}

// length of a vector
inline float length(const float3 &v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// distance between two points (always positive)
inline float distance(const float3 &a, const float3 &b) {
    return length(a - b);
}

// normalization
inline float3 normalize(const float3 &v) {
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

#endif // UTIL_VECTORS_H_

