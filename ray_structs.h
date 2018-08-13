#ifndef RAY_STRUCTS_H
#define RAY_STRUCTS_H
#include <stdint.h>
#include <veclib.h>
#include <stdlib.h>

enum ObjectType {
    Sphere,
    Triangle
};


struct Material {
    float reflect;
    float diffuse;
    float refract;
    float ior;
    float emit;
    struct vec3 (*eval)(float, float, float);
};


struct Camera {
    struct vec3 center;
    struct vec3 lookat;
    struct vec3 up;
};

struct Triangle {
    struct vec3 pt0;
    struct vec3 u;
    struct vec3 v;
    struct vec3 normal;
    struct Material m;
};

struct Sphere {
    struct vec3 origin;
    float radius;
    struct Material m;
};

struct Scene {
    float *pt0[3];
    float *u[3];
    float *v[3];
    float *normal[3];
    float *origins[3];
    float *radius;
    size_t numtris;
    size_t numspheres;
};

struct SceneAOS {
    struct Triangle *tris;
    struct Sphere *spheres;
    size_t numtris;
    size_t numspheres;
};
struct Ray {
    struct vec3 origin;
    struct vec3 direction;
    struct vec3 inv_dir;
    struct vec3 normal;
    size_t bounces;
    size_t id;
    float t;
    struct Material m;
    struct vec3 lit;
    float u, v;
};

struct RaySet {
    struct Ray *r;
    size_t numrays;
};

struct AABB {
    struct vec3 min;
    struct vec3 max;
};
struct DACRTPartition {
    int terminatedRay;
    int rayStart, rayEnd;
    int triStart, triEnd;
    int sphereStart, sphereEnd;
    struct AABB bounds;
};

struct Texture {
    uint8_t *data;
    uint16_t x;
    uint16_t y;
    float scale;
    float xoff;
    float yoff;
};

enum DivisionAxis {
    X,
    Y,
    Z
};

struct StorageTriangle {
    struct vec3 pts[3];
    struct Material mat;
};

struct StorageSphere {
    struct vec3 origin;
    float radius;
    struct Material mat;
};

struct SceneIndirect {
    int *tris;
    int *spheres;
    int *rays;
};

struct DuoPartition {
    struct DACRTPartition part[2];
};

#endif // RAY_STRUCTS_H
