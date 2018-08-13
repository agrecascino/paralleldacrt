#ifndef VECLIB_H
#define VECLIB_H
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct vec3 {
    union {
        float xyz[4];
        struct {
            float x, y, z, w;
        };
    };
};

static inline struct vec3 vec_add(struct vec3 a, struct vec3 b) {
    #pragma omp simd
    for(int i = 0; i < 3; i++) {
        a.xyz[i] += b.xyz[i];
    }
//    a.x += b.x;
//    a.y += b.y;
//    a.z += b.z;
    return a;
}

static inline struct vec3 vec_mul(struct vec3 a, struct vec3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

static inline struct vec3 vec_sub(struct vec3 a, struct vec3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

static inline struct vec3 vec_dup(float f) {
    struct vec3 res;
    res.x = f;
    res.y = f;
    res.z = f;
    return res;
}

static inline struct vec3 vec_cross(struct vec3 a, struct vec3 b) {
    struct vec3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

static inline float vec_length(struct vec3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

static inline struct vec3 vec_norm(struct vec3 a) {
    float length = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    a.x = a.x/length;
    a.y = a.y/length;
    a.z = a.z/length;
    return a;
}

static inline struct vec3 vec_mid(struct vec3 a, struct vec3 b) {
    struct vec3 res;
    res.x = (a.x + b.x) * 0.5f;
    res.y = (a.y + b.y) * 0.5f;
    res.z = (a.z + b.z) * 0.5f;
    return res;
}

static inline float vec_dot(struct vec3 a, struct vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static void vec_print(struct vec3 a) {
    printf("Values of vec3 \"a\": (%f, %f, %f)\n", a.x, a.y, a.z);
}

static char* vec_sprint(struct vec3 a) {
    char *s = malloc(80);
    snprintf(s, 80, "Values of vec3 \"a\": (%f, %f, %f)\n", a.x, a.y, a.z);
    return s;
}

#endif // VECLIB_H
