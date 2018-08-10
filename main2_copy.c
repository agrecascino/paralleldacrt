#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <immintrin.h>

/*  Written in 2016-2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. Th is software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include <stdint.h>

/* This is xoroshiro128+ 1.0, our best and fastest small-state generator
   for floating-point numbers. We suggest to use its upper bits for
   floating-point generation, as it is slightly faster than
   xoroshiro128**. It passes all tests we are aware of except for the four
   lower bits, which might fail linearity tests (and just those), so if
   low linear complexity is not considered an issue (as it is usually the
   case) it can be used to generate 64-bit outputs, too; moreover, this
   generator has a very mild Hamming-weight dependency making our test
   (http://prng.di.unimi.it/hwd.php) fail after 8 TB of output; we believe
   this slight bias cannot affect any application. If you are concerned,
   use xoroshiro128** or xoshiro256+.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s.

   NOTE: the parameters (a=24, b=16, b=37) of this version give slightly
   better results in our test than the 2016 version (a=55, b=14, c=37).
*/

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}


static uint64_t s[2] ={12, 424};

uint64_t next(void) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
    s[1] = rotl(s1, 37); // c

    return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^64 calls to next(); it can be used to generate 2^64
   non-overlapping subsequences for parallel computations. */

void jump(void) {
    static const uint64_t JUMP[] = { 0xdf900294d8f554a5, 0x170865df4b3201fc };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
            }
            next();
        }

    s[0] = s0;
    s[1] = s1;
}


/* This is the long-jump function for the generator. It is equivalent to
   2^96 calls to next(); it can be used to generate 2^32 starting points,
   from each of which jump() will generate 2^32 non-overlapping
   subsequences for parallel distributed computations. */

void long_jump(void) {
    static const uint64_t LONG_JUMP[] = { 0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1 };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
            }
            next();
        }

    s[0] = s0;
    s[1] = s1;
}


struct vec3 {
    union {
        float xyz[3];
        struct {
            float x, y, z;
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

struct vec3 vec_norm(struct vec3 a) {
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

void vec_print(struct vec3 a) {
    printf("Values of vec3 \"a\": (%f, %f, %f)\n", a.x, a.y, a.z);
}

char* vec_sprint(struct vec3 a) {
    char *s = malloc(80);
    snprintf(s, 80, "Values of vec3 \"a\": (%f, %f, %f)\n", a.x, a.y, a.z);
    return s;
}

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
};

struct Sphere {
    struct vec3 origin;
    float radius;
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

struct Ray {
    struct vec3 origin;
    struct vec3 direction;
    struct vec3 inv_dir;
    struct vec3 normal;
    size_t bounces;
    size_t id;
    float t;
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

struct AABB AABBFromScene(struct Scene *s) {
    struct vec3 minimum;
    minimum.x = INFINITY;
    minimum.y = INFINITY;
    minimum.z = INFINITY;
    struct vec3 maximum;
    maximum.x = -INFINITY;
    maximum.y = -INFINITY;
    maximum.z = -INFINITY;
    for(size_t i = 0; i < s->numtris; i++) {
        struct vec3 pt0;
        pt0.x = s->pt0[0][i];
        pt0.y = s->pt0[1][i];
        pt0.z = s->pt0[2][i];
        struct vec3 pt1;
        pt1.x = pt0.x + s->u[0][i];
        pt1.y = pt0.y + s->u[1][i];
        pt1.z = pt0.z + s->u[2][i];
        struct vec3 pt2;
        pt2.x = pt0.x + s->v[0][i];
        pt2.y = pt0.y + s->v[1][i];
        pt2.z = pt0.z + s->v[2][i];
        minimum.x = fmin(fmin(fmin(pt0.x, pt1.x), pt2.x), minimum.x);
        minimum.y = fmin(fmin(fmin(pt0.y, pt1.y), pt2.y), minimum.y);
        minimum.z = fmin(fmin(fmin(pt0.z, pt1.z), pt2.z), minimum.z);
        maximum.x = fmax(fmax(fmax(pt0.x, pt1.x), pt2.x), maximum.x);
        maximum.y = fmax(fmax(fmax(pt0.y, pt1.y), pt2.y), maximum.y);
        maximum.z = fmax(fmax(fmax(pt0.z, pt1.z), pt2.z), maximum.z);
    }
    for(size_t i = 0; i < s->numspheres; i++) {
        struct vec3 origin;
        origin.x = s->origins[0][i];
        origin.y = s->origins[1][i];
        origin.z = s->origins[2][i];
        struct vec3 pt0 = vec_sub(origin, vec_dup(s->radius[i]));
        struct vec3 pt1 = vec_add(origin, vec_dup(s->radius[i]));
        minimum.x = fmin(fmin(pt0.x, pt1.x), minimum.x);
        minimum.y = fmin(fmin(pt0.y, pt1.y), minimum.y);
        minimum.z = fmin(fmin(pt0.z, pt1.z), minimum.z);
        maximum.x = fmax(fmax(pt0.x, pt1.x), maximum.x);
        maximum.y = fmax(fmax(pt0.y, pt1.y), maximum.y);
        maximum.z = fmax(fmax(pt0.z, pt1.z), maximum.z);
    }
    minimum.x -= 0.001f;
    minimum.y -= 0.001f;
    minimum.z -= 0.001f;
    maximum.x += 0.001f;
    maximum.y += 0.001f;
    maximum.z += 0.001f;
    struct AABB sceneaabb;
    sceneaabb.min = minimum;
    sceneaabb.max = maximum;
    return sceneaabb;
}

int intersectSphere(struct Scene *s, int i, struct Ray *r) {
    struct vec3 origin;
    origin.x = s->origins[0][i];
    origin.y = s->origins[1][i];
    origin.z = s->origins[2][i];
    float radius = s->radius[i];
    const struct vec3 o = vec_sub(origin, r->origin);
    const float tca = vec_dot(o, r->direction);
    float d2 = vec_dot(o, o) - tca*tca;
    if(d2 > radius*radius)
        return 0;
    const float tc = tca - sqrtf(radius*radius - d2);
    if(tc < 0)
        return 0;
    if(tc > r->t)
        return 0;
    r->t = tc;
    r->normal = vec_norm(vec_sub(vec_add(r->origin, vec_mul(vec_dup(r->t), r->direction)), origin));
    return 1;
}

int intersectSphere8x(struct Sphere *s, struct Ray *r) {
    const struct vec3 o = vec_sub(s->origin, r->origin);
    const float tca = vec_dot(o, r->direction);
    float d2 = vec_dot(o, o) - tca*tca;
    if(d2 > s->radius*s->radius)
        return 0;
    const float tc = tca - sqrtf(s->radius*s->radius - d2);
    if(tc < 0)
        return 0;
    if(tc > r->t)
        return 0;
    r->t = tc;
    r->normal = vec_norm(vec_sub(vec_add(r->origin, vec_mul(vec_dup(r->t), r->direction)), s->origin));
    return 1;
}

int intersectTriangle(struct Scene *scene, int i, struct Ray *r) {
    struct vec3 pt0;
    pt0.x = scene->pt0[0][i];
    pt0.y = scene->pt0[1][i];
    pt0.z = scene->pt0[2][i];
    struct vec3 ut;
    ut.x = scene->u[0][i];
    ut.y = scene->u[1][i];
    ut.z = scene->u[2][i];
    struct vec3 vt;
    vt.x = scene->v[0][i];
    vt.y = scene->v[1][i];
    vt.z = scene->v[2][i];
    const float eps = 0.0001f;
    const struct vec3 h = vec_cross(r->direction, vt);
    const float a = vec_dot(ut, h);
    if(a > -eps && a < eps) {
        return 0;
    }
    const float f = 1.0f / a;
    const struct vec3 s = vec_sub(r->origin, pt0);
    const float u = f * vec_dot(s, h);
    if(u < 0.0f || u > 1.0f) {
        return 0;
    }
    const struct vec3 q = vec_cross(s, ut);
    const float v = f * vec_dot(r->direction, q);
    if(v < 0.0f || u+v > 1.0f) {
        return 0;
    }
    const float ts = f * vec_dot(vt, q);
    if(ts < eps) {
        return 0;
    }
    if(ts > r->t) {
        return 0;
    }
    r->t = ts;
    r->normal.x = scene->normal[0][i];
    r->normal.y = scene->normal[1][i];
    r->normal.z = scene->normal[2][i];
    return 1;
}

struct StorageTriangle {
    struct vec3 pts[3];
};

struct StorageSphere {
    struct vec3 origin;
    float radius;
};

struct Scene generateSceneGraphFromStorage(struct StorageTriangle *tris, struct StorageSphere *spheres, size_t numtris, size_t numspheres) {
    struct Scene scene;
    scene.numtris = numtris;
    scene.numspheres = numspheres;
    scene.pt0[0] = malloc(sizeof(float) * scene.numtris);
    scene.pt0[1] = malloc(sizeof(float) * scene.numtris);
    scene.pt0[2] = malloc(sizeof(float) * scene.numtris);
    scene.u[0] = malloc(sizeof(float) * scene.numtris);
    scene.u[1] = malloc(sizeof(float) * scene.numtris);
    scene.u[2] = malloc(sizeof(float) * scene.numtris);
    scene.v[0] = malloc(sizeof(float) * scene.numtris);
    scene.v[1] = malloc(sizeof(float) * scene.numtris);
    scene.v[2] = malloc(sizeof(float) * scene.numtris);
    scene.normal[0] = malloc(sizeof(float) * scene.numtris);
    scene.normal[1] = malloc(sizeof(float) * scene.numtris);
    scene.normal[2] = malloc(sizeof(float) * scene.numtris);
    scene.origins[0] = malloc(sizeof(float) * scene.numspheres);
    scene.origins[1] = malloc(sizeof(float) * scene.numspheres);
    scene.origins[2] = malloc(sizeof(float) * scene.numspheres);
    scene.radius = malloc(sizeof(float) * scene.numspheres);
    //scene.tris = malloc(sizeof(struct Triangle) * scene.numtris);
    //scene.spheres = malloc(sizeof(struct Sphere) * scene.numspheres);
    for(size_t i = 0; i < scene.numtris; i++) {
        struct vec3 pt0 = tris[i].pts[0];
        struct vec3 u = vec_sub(tris[i].pts[1], tris[i].pts[0]);
        struct vec3 v = vec_sub(tris[i].pts[2], tris[i].pts[0]);
        //scene.tris[i].pt0 = pt0;
        scene.pt0[0][i] = pt0.x;
        scene.pt0[1][i] = pt0.y;
        scene.pt0[2][i] = pt0.z;
        scene.u[0][i] = u.x;
        scene.u[1][i] = u.y;
        scene.u[2][i] = u.z;
        scene.v[0][i] = v.x;
        scene.v[1][i] = v.y;
        scene.v[2][i] = v.z;
        //scene.tris[i].u = u;
        //scene.tris[i].v = v;
        struct vec3 normal = vec_norm(vec_cross(u, v));
        scene.normal[0][i] = normal.x;
        scene.normal[1][i] = normal.y;
        scene.normal[2][i] = normal.z;
    }
    for(size_t i = 0; i < scene.numspheres; i++) {
        scene.origins[0][i] = spheres[i].origin.x;
        scene.origins[1][i] = spheres[i].origin.y;
        scene.origins[2][i] = spheres[i].origin.z;
        scene.radius[i] = spheres[i].radius;
    }
    return scene;
}

void deallocScene(struct Scene s) {
    //free(s.spheres);
    //free(s.tris);
    for(int i = 0; i < 3; i++)
        free(s.origins[i]);
    for(int i = 0; i < 3; i++)
        free(s.pt0[i]);
    free(s.radius);
    for(int i = 0; i < 3; i++)
        free(s.u[i]);
    for(int i = 0; i < 3; i++)
        free(s.v[i]);
    for(int i = 0; i < 3; i++)
        free(s.normal[i]);
}

enum DivisionAxis {
    X,
    Y,
    Z
};

static inline int AABBintersection(struct AABB b, struct Ray *r, float *t) {
    float tx1 = (b.min.x - r->origin.x) * r->inv_dir.x;
    float tx2 = (b.max.x - r->origin.x) * r->inv_dir.x;

    float tmin = fmin(tx1, tx2);
    float tmax = fmax(tx1, tx2);

    float ty1 = (b.min.y - r->origin.y) * r->inv_dir.y;
    float ty2 = (b.max.y - r->origin.y) * r->inv_dir.y;

    tmin = fmax(tmin, fmin(ty1, ty2));
    tmax = fmin(tmax, fmax(ty1, ty2));

    float tz1 = (b.min.z - r->origin.z) * r->inv_dir.z;
    float tz2 = (b.max.z - r->origin.z) * r->inv_dir.z;

    tmin = fmax(tmin, fmin(tz1, tz2));
    tmax = fmin(tmax, fmax(tz1, tz2));

    *t = tmin;

    return tmax >= tmin && (*t > 0);
}

struct vec3 boxNormal(struct AABB box, struct Ray ray, float t) {
    struct vec3 hit = vec_add(ray.origin, vec_mul(ray.direction, vec_dup(t)));
    struct vec3 c = vec_mul(vec_add(box.min, box.max), vec_dup(0.5f));
    struct vec3 p = vec_sub(hit, c);
    struct vec3 d = vec_mul(vec_sub(box.min, box.max), vec_dup(0.5f));
    float bias = 1.00001f;
    struct vec3 normal;
    normal.x = (int)(p.x / fabs(d.x) * bias);
    normal.y = (int)(p.y / fabs(d.y) * bias);
    normal.z = (int)(p.z / fabs(d.z) * bias);
    return normal;
}

enum DivisionAxis bestAxis(struct Camera cam, struct DACRTPartition part) {
    struct vec3 c = vec_add(part.bounds.min, part.bounds.max);
    c = vec_mul(c, vec_dup(0.5f));
    struct vec3 spacevec = vec_sub(c, cam.center);
    struct vec3 direction = vec_norm(spacevec);
    struct Ray r;
    r.origin = cam.center;
    r.direction = direction;
    r.inv_dir.x = 1.0f/direction.x;
    r.inv_dir.y = 1.0f/direction.y;
    r.inv_dir.z = 1.0f/direction.z;
    float t;
    AABBintersection(part.bounds, &r, &t);
    struct vec3 normal = boxNormal(part.bounds, r, t);
    normal.x = fabs(normal.x);
    normal.y = fabs(normal.y);
    normal.z = fabs(normal.z);
    if((int)normal.x == 1) {
        return X;
    }
    if((int)normal.y == 1) {
        return Y;
    }
    if((int)normal.z == 1) {
        return Z;
    }
    printf("We have officially caught on fire. Your Geometry may be fourth-dimensional.\n");
    exit(EXIT_FAILURE);
    return 0;
}

const char* axisString(enum DivisionAxis a) {
    switch(a) {
    case X:
        return "X";
    case Y:
        return "Y";
    case Z:
        return "Z";
    }
    return "";
}

struct DuoPartition {
    struct DACRTPartition part[2];
};

struct DuoPartition averageSpaceCut(struct DACRTPartition part, enum DivisionAxis axis) {
    struct DuoPartition duo;
    struct vec3 diff = vec_sub(part.bounds.max, part.bounds.min);
    struct vec3 m1 = part.bounds.min;
    struct vec3 m2 = part.bounds.max;
    m2.x -= diff.x * ((axis == X) ? 0.5f : 0.0f);
    m2.y -= diff.y * ((axis == Y) ? 0.5f : 0.0f);
    m2.z -= diff.z * ((axis == Z) ? 0.5f : 0.0f);
    struct vec3 m3 = part.bounds.min;
    m3.x += diff.x * ((axis == X) ? 0.5f : 0.0f);
    m3.y += diff.y * ((axis == Y) ? 0.5f : 0.0f);
    m3.z += diff.z * ((axis == Z) ? 0.5f : 0.0f);
    struct vec3 m4 = part.bounds.max;
    duo.part[0].bounds.min = m1;
    duo.part[0].bounds.max = m2;
    duo.part[1].bounds.min = m3;
    duo.part[1].bounds.max = m4;
    return duo;
}

static inline void list_swap(void *i1, void *i2, size_t size) {
    unsigned char object[size];
    memcpy(object, i2, size);
    memcpy(i2, i1, size);
    memcpy(i1, object, size);
}

//struct PivotPair {
//    int firstEnd;
//    int secondStart;
//};

//struct PivotPair findTrianglePivots(struct DuoPartition duo, struct DACRTPartition p, struct Scene *scene, enum DivisionAxis axis) {
//    int sharedStart = p.triStart;
//    int knownSorted = p.triStart;
//    float bboxsplit = duo.part[0].bounds.max.xyz[axis];
//    for(int i = p.triStart; i < p.triEnd; i++) {
//        struct vec3 pt0 = scene->tris[i].pt0;
//        struct vec3 pt1 = vec_add(pt0, scene->tris[i].u);
//        struct vec3 pt2 = vec_add(pt0, scene->tris[i].v);
//        float v = fmin(pt0.xyz[axis], fmin(pt1.xyz[axis], pt2.xyz[axis]));
//        if(v < bboxsplit) {
//            list_swap(&scene->tris[i], &scene->tris[knownSorted], sizeof(struct Triangle));
//            knownSorted++;
//        }
//    }
//    sharedStart = knownSorted;
//    for(int i = p.triStart; i < knownSorted; i++) {
//        struct vec3 pt0 = scene->tris[i].pt0;
//        struct vec3 pt1 = vec_add(pt0, scene->tris[i].u);
//        struct vec3 pt2 = vec_add(pt0, scene->tris[i].v);
//        float v = fmax(pt0.xyz[axis], fmax(pt1.xyz[axis], pt2.xyz[axis]));
//        if(v > bboxsplit) {
//            list_swap(&scene->tris[i], &scene->tris[sharedStart], sizeof(struct Triangle));
//            sharedStart--;
//        }
//    }
//    struct PivotPair piv;
//    piv.firstEnd = knownSorted;
//    piv.secondStart = sharedStart;
//    return piv;
//}

//struct PivotPair findSpherePivots(struct DuoPartition duo, struct DACRTPartition p, struct Scene *scene, enum DivisionAxis axis) {
//    int sharedStart = p.sphereStart;
//    int knownSorted = p.sphereStart;
//    float bboxsplit = duo.part[0].bounds.max.xyz[axis];
//    for(int i = p.sphereStart; i < p.sphereEnd; i++) {
//        float a = scene->spheres[i].origin.xyz[axis] + scene->spheres[i].radius;
//        float b = scene->spheres[i].origin.xyz[axis] - scene->spheres[i].radius;
//        float v = fmin(a, b);
//        if(v < bboxsplit) {
//            list_swap(&scene->spheres[i], &scene->spheres[knownSorted], sizeof(struct Sphere));
//            knownSorted++;
//        }
//    }
//    sharedStart = knownSorted;
//    for(int i = p.sphereStart; i < knownSorted; i++) {
//        float a = scene->spheres[i].origin.xyz[axis] + scene->spheres[i].radius;
//        float b = scene->spheres[i].origin.xyz[axis] - scene->spheres[i].radius;
//        float v = fmax(a, b);
//        if(v > bboxsplit) {
//            list_swap(&scene->spheres[i], &scene->spheres[sharedStart], sizeof(struct Sphere));
//            sharedStart--;
//        }
//    }
//    struct PivotPair piv;
//    piv.firstEnd = knownSorted;
//    piv.secondStart = sharedStart;
//    return piv;
//}

//struct RayPivots {
//    int terminatedRay;
//    int firstEnd;
//    int secondStart;
//};

//struct RayPivots findRayPivots(struct DuoPartition duo, struct DACRTPartition p, struct Ray *rays) {
//    int terminatedRay = p.rayStart;
//    int split = p.rayStart;
//    int share = p.rayStart;
//    for(int i = p.rayStart; i < p.rayEnd; i++) {
//        float t;
//        if(!AABBintersection(p.bounds, rays[i], &t)) {
//            list_swap(&rays[i], &rays[terminatedRay], sizeof(struct Ray));
//            terminatedRay++;
//        }
//    }
//    split = terminatedRay;
//    for(int i = terminatedRay; i < p.rayEnd; i++) {
//        float t;
//        if(!AABBintersection(duo.part[0].bounds, rays[i], &t)) {
//            list_swap(&rays[i], &rays[split], sizeof(struct Ray));
//            split++;
//        }
//    }
//    share = split;
//    for(int i = terminatedRay; i < split; i++) {
//        float t;
//        if(!AABBintersection(duo.part[1].bounds, rays[i], &t)) {
//            list_swap(&rays[i], &rays[share], sizeof(struct Ray));
//            share--;
//        }
//    }
//    struct RayPivots piv;
//    piv.terminatedRay = terminatedRay;
//    piv.firstEnd = split;
//    piv.secondStart = share;
//    return piv;
//}

//struct DuoPartition subdivideSpace(struct DACRTPartition part, enum DivisionAxis axis, struct Camera cam, struct Scene *scene, struct Ray *rays) {
//    struct DuoPartition duo;
//    duo = averageSpaceCut(part, axis);
//    float rlength = vec_length(vec_sub(vec_mid(duo.part[1].bounds.min, duo.part[1].bounds.max), cam.center));
//    float llength = vec_length(vec_sub(vec_mid(duo.part[0].bounds.min, duo.part[0].bounds.max), cam.center));
//    if(rlength < llength) {
//        struct DACRTPartition p = duo.part[0];
//        duo.part[0] = duo.part[1];
//        duo.part[1] = p;
//    }
//    struct PivotPair pT = findTrianglePivots(duo, part, scene, axis);
//    struct PivotPair pS = findSpherePivots(duo, part, scene, axis);
//    return duo;
//}

//void DACRTNonPacketParallel(struct Camera cam, struct AABB space, struct Scene *scene, struct Ray *rays, size_t nthreads, size_t numrays) {
//    struct DACRTPartition part;
//    part.bounds = space;
//    part.terminatedRay = 0;
//    part.rayStart = 0;
//    part.triStart = 0;
//    part.sphereStart = 0;
//    part.triEnd = scene->numtris - 1;
//    part.rayEnd = numrays-1;
//    part.sphereEnd = scene->numspheres - 1;
//    enum DivisionAxis axis = bestAxis(cam, part);

//}

//struct vec3 RT(struct Ray *r, struct Scene *scene, int noshade) {
//    struct vec3 color;
//    struct vec3 normal;
//    float t = 65537.0f;
//    for(size_t i = 0; i < scene->numspheres; i++) {
//        intersectSphere(&scene->spheres[i], r);
//    }
//    for(size_t i = 0; i < scene->numtris; i++) {
//        intersectTriangle(&scene->tris[i], r);
//    }
//    if(t > 65536.0f) {
//        return vec_dup(0.04f);
//    }
//    if(noshade)
//        return vec_dup(0.0f);
//    float accum = 0.0f;
//    if(vec_dot(normal, vec_mul(r->direction, vec_dup(-1.0f))) < 0.0f) {
//        normal = vec_mul(normal, vec_dup(-1.0f));
//    }
//    for(int i = 0; i < 1; i++) {
//        float x = (((float)(next()&0xFFFF) - 0xFFFF/2.0f)/32767.5f) * 0.8;
//        float y = (((float)(next()&0xFFFF) - 0xFFFF/2.0f)/32767.5f) * 0.8;
//        float z = (((float)(next()&0xFFFF) - 0xFFFF/2.0f)/32767.5f) * 0.8;
//        struct vec3 randvec;
//        randvec.x = x;
//        randvec.y = y;
//        randvec.z = z;
//        randvec = randvec;
//        struct vec3 dir = normal;
//        struct vec3 org = vec_add(vec_add(r->origin, vec_mul(r->direction, vec_dup(t))), vec_mul(normal, vec_dup(0.02f)));
//        struct Ray r2;
//        dir = vec_norm(vec_add(dir, randvec));
//        r2.direction = dir;
//        r2.origin = org;
//        r2.t = 65537.0f;
//        RT(&r2, scene, 1);
//        accum += (1/1.0f)*(r2.t/65537.0f);
//    }
//    struct vec3 lloc;
//    lloc.x = 0.0f;
//    lloc.z = 0.0f;
//    lloc.y = 10.0f;
//    struct vec3 ldir = vec_norm(vec_sub(lloc, vec_add(r->origin, vec_mul(r->direction, vec_dup(t)))));
//    color = vec_mul(vec_dup(1.0f*accum), vec_dup(fmax((vec_dot(normal, ldir)+1.0f)/2.0f, 0.0f)));
//    return color;
//}

static inline double fastPow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

struct SceneIndirect {
    int *tris;
    int *spheres;
    int *rays;
};

//struct intersectTriangle8x(struct Ray *r, struct Scene *scene, int start) {
//    __m128 a;
//}
void NRT(struct Ray *r, struct Scene *scene, struct DACRTPartition part) {
    for(int rx = part.rayStart; rx < part.rayEnd; rx++) {
        for(size_t i = part.sphereStart; i < part.sphereEnd; i++) {
            intersectSphere(scene, i, r + rx);
        }
        for(size_t i = part.triStart; i < part.triEnd; i++) {
            intersectTriangle(scene, i, r + rx);
        }
    }
}
void DACRTWorkingNoEarlyTerm(struct DACRTPartition space, struct Ray *r, struct Scene s, struct Camera cam, int depth) {
    if((space.rayEnd-space.rayStart) < 20 || ((space.triEnd-space.triStart) + (space.sphereEnd-space.sphereStart)) < 16) {
        NRT(r, &s, space);
        return;
    }
    enum DivisionAxis axis = depth % 3;
    struct DuoPartition d2 = averageSpaceCut(space, axis);
    float rlength = vec_length(vec_sub(vec_mid(d2.part[1].bounds.min, d2.part[1].bounds.max), cam.center));
    float llength = vec_length(vec_sub(vec_mid(d2.part[0].bounds.min, d2.part[0].bounds.max), cam.center));
    //float rlength = d2.part[0].bounds.min.xyz[axis];
    //float llength = d2.part[1].bounds.min.xyz[axis];
    int truezero = 0;
    if(rlength > llength) {
        struct DACRTPartition p = d2.part[0];
        d2.part[0] = d2.part[1];
        d2.part[1] = p;
        truezero = 1;
    }
    for(int ps = 0; ps < 2; ps++) {
        int pivot = space.rayStart;
        int tpivot = space.triStart;
        int spivot = space.sphereStart;
        for(int i = pivot; i < space.rayEnd; i++) {
            float t = INFINITY;
//            struct vec3 o = r[i].origin;
//            struct vec3 bcmid = vec_mid(d2.part[ps].bounds.min, d2.part[ps].bounds.max);
//            struct vec3 bomid = vec_mid(d2.part[(ps + 1) % 2].bounds.min, d2.part[(ps + 1) % 2].bounds.max);
//            struct vec3 ts = vec_sub(o, bomid);
//            struct vec3 tf = vec_sub(o, bcmid);
//            float vl = vec_length(ts);
//            float vlo  = vec_length(tf);
//            float tother = INFINITY;
            //tother += 0.001f;
//            int hitother = AABBintersection(d2.part[(ps + 1) % 2].bounds, r[i], &tother);
            int hit = AABBintersection(d2.part[ps].bounds, r + i, &t);
            int terminated = r[i].t < t; /*&& (tother < t) && hitother;*/
            if(hit && !terminated) {
                if(i != pivot)
                    list_swap(r + i, r + pivot, sizeof(struct Ray));
                pivot++;
            }
        }
        for(int i = tpivot; i < space.triEnd; i++) {
//            struct vec3 pt0;
//            pt0.x = s.pt0[0][i];
//            pt0.y = s.pt0[1][i];
//            pt0.z = s.pt0[2][i];
//            struct vec3 pt1;
//            pt1.x = pt0.x + s.u[0][i];
//            pt1.y = pt0.y + s.u[1][i];
//            pt1.z = pt0.z + s.u[2][i];
//            struct vec3 pt2;
//            pt2.x = pt0.x + s.v[0][i];
//            pt2.y = pt0.y + s.v[1][i];
//            pt2.z = pt0.z + s.v[2][i];
            float p1, p2, p3;
            p1 = s.pt0[axis][i];
            p2 = p1 + s.u[axis][i];
            p3 = p1 + s.v[axis][i];
//            float v = (ps == truezero) ? fmin(pt0.xyz[axis], fmin(pt1.xyz[axis], pt2.xyz[axis])) : fmax(pt0.xyz[axis], fmax(pt1.xyz[axis], pt2.xyz[axis]));
            float v = (ps == truezero) ? fmin(p1, fmin(p2, p3)) : fmax(p1, fmax(p2, p3));
            if((ps == truezero && v < d2.part[truezero].bounds.max.xyz[axis] + 0.01f) || (ps != truezero && v > d2.part[truezero].bounds.max.xyz[axis] - 0.01f)) {
                if(i != tpivot) {
                    list_swap(s.pt0[0] + i, s.pt0[0] + tpivot, sizeof(float));
                    list_swap(s.pt0[1] + i, s.pt0[1] + tpivot, sizeof(float));
                    list_swap(s.pt0[2] + i, s.pt0[2] + tpivot, sizeof(float));
                    list_swap(s.u[0] + i, s.u[0] + tpivot, sizeof(float));
                    list_swap(s.u[1] + i, s.u[1] + tpivot, sizeof(float));
                    list_swap(s.u[2] + i, s.u[2] + tpivot, sizeof(float));
                    list_swap(s.v[0] + i, s.v[0] + tpivot, sizeof(float));
                    list_swap(s.v[1] + i, s.v[1] + tpivot, sizeof(float));
                    list_swap(s.v[2] + i, s.v[2] + tpivot, sizeof(float));
                    list_swap(s.normal[0] + i, s.normal[0] + tpivot, sizeof(float));
                    list_swap(s.normal[1] + i, s.normal[1] + tpivot, sizeof(float));
                    list_swap(s.normal[2] + i, s.normal[2] + tpivot, sizeof(float));
                }
                tpivot++;
            }
        }
        for(int i = spivot; i < space.sphereEnd; i++) {
            float a = s.origins[axis][i] + s.radius[i];
            float b = s.origins[axis][i] - s.radius[i];
            float v = (ps == 0) ? fmin(a, b) : fmax(a, b);
            if((ps == 0 && v < d2.part[truezero].bounds.max.xyz[axis]) || (ps == 1 && v > d2.part[truezero].bounds.max.xyz[axis])) {
                //list_swap(s.spheres + i, s.spheres + pivot, sizeof(struct Sphere));
                list_swap(s.origins[0] + i, s.origins[0] + pivot, sizeof(float));
                list_swap(s.origins[1] + i, s.origins[1] + pivot, sizeof(float));
                list_swap(s.origins[2] + i, s.origins[2] + pivot, sizeof(float));
                list_swap(s.radius + i, s.radius + pivot, sizeof(float));
                spivot++;
            }
        }
        struct DACRTPartition p;
        p.bounds = d2.part[ps].bounds;
        p.rayEnd = pivot;
        p.rayStart = space.rayStart;
        p.sphereEnd = spivot;
        p.sphereStart = space.sphereStart;
        p.triEnd = tpivot;
        p.triStart = space.triStart;
        DACRTWorkingNoEarlyTerm(p, r, s, cam, depth + 1);
    }

}

void NRTIndirect(struct Ray *r, struct Scene *scene, struct DACRTPartition *part, struct SceneIndirect *si) {
    for(int rx = part->rayStart; rx < part->rayEnd; rx++) {

        for(size_t i = part->sphereStart; i < part->sphereEnd; i++) {
            intersectSphere(scene, si->spheres[i], r + si->rays[rx]);
        }
        for(size_t i = part->triStart; i < part->triEnd; i++) {
            intersectTriangle(scene, si->tris[i], r + si->rays[rx]);
        }
    }
}

void DACRTWorkingNoEarlyTermIndirect(struct DACRTPartition *space, struct Ray *r, struct Scene *s, struct Camera *cam, int depth, struct SceneIndirect *si) {
    if((space->rayEnd-space->rayStart) < 20 || ((space->triEnd-space->triStart) + (space->sphereEnd-space->sphereStart)) < 16) {
        NRTIndirect(r, s, space, si);
        return;
    }
    enum DivisionAxis axis = depth % 3;
    struct DuoPartition d2 = averageSpaceCut(*space, axis);
    float rlength = vec_length(vec_sub(vec_mid(d2.part[1].bounds.min, d2.part[1].bounds.max), cam->center));
    float llength = vec_length(vec_sub(vec_mid(d2.part[0].bounds.min, d2.part[0].bounds.max), cam->center));
    //float rlength = d2.part[0].bounds.min.xyz[axis];
    //float llength = d2.part[1].bounds.min.xyz[axis];
    int truezero = 0;
    if(rlength > llength) {
        struct DACRTPartition p = d2.part[0];
        d2.part[0] = d2.part[1];
        d2.part[1] = p;
        truezero = 1;
    }
    for(int ps = 0; ps < 2; ps++) {
        int pivot = space->rayStart;
        int tpivot = space->triStart;
        int spivot = space->sphereStart;
        for(int i = pivot; i < space->rayEnd; i++) {
            int trueitem = si->rays[i];
            float t = INFINITY;
            int hit = AABBintersection(d2.part[ps].bounds, r + trueitem, &t);
            int terminated = r[trueitem].t < t; /*&& (tother < t) && hitother;*/
            if(hit && !terminated) {
                if(i != pivot) {
                    si->rays[i] = si->rays[pivot];
                    si->rays[pivot] = trueitem;
                }
                pivot++;
            }
        }
        for(int i = tpivot; i < space->triEnd; i++) {
            int trueitem = si->tris[i];
            float p1, p2, p3;
            p1 = s->pt0[axis][trueitem];
            p2 = p1 + s->u[axis][trueitem];
            p3 = p1 + s->v[axis][trueitem];
            float v = (ps == truezero) ? fmin(p1, fmin(p2, p3)) : fmax(p1, fmax(p2, p3));
            if((ps == truezero && v < d2.part[truezero].bounds.max.xyz[axis] + 0.01f) || (ps != truezero && v > d2.part[truezero].bounds.max.xyz[axis] - 0.01f)) {
                if(i != tpivot) {
                    si->tris[i] = si->tris[tpivot];
                    si->tris[tpivot] = trueitem;
                }
                tpivot++;
            }
        }
        for(int i = spivot; i < space->sphereEnd; i++) {
            int trueitem = si->spheres[i];
            float a = s->origins[axis][trueitem] + s->radius[trueitem];
            float b = s->origins[axis][trueitem] - s->radius[trueitem];
            float v = (ps == 0) ? fmin(a, b) : fmax(a, b);
            if((ps == 0 && v < d2.part[truezero].bounds.max.xyz[axis]) || (ps == 1 && v > d2.part[truezero].bounds.max.xyz[axis])) {
                if(i != spivot) {
                    si->spheres[i] = si->spheres[spivot];
                    si->spheres[spivot] = trueitem;
                }
                spivot++;
            }
        }
        struct DACRTPartition p;
        p.bounds = d2.part[ps].bounds;
        p.rayEnd = pivot;
        p.rayStart = space->rayStart;
        p.sphereEnd = spivot;
        p.sphereStart = space->sphereStart;
        p.triEnd = tpivot;
        p.triStart = space->triStart;
        DACRTWorkingNoEarlyTermIndirect(&p, r, s, cam, depth + 1, si);
    }

}

//void NRTIndirect(struct Ray *r, struct Scene *scene, struct DACRTPartition part, struct SceneIndirect *si) {
//    for(int rx = part.rayStart; rx < part.rayEnd; rx++) {
//        for(size_t i = part.sphereStart; i < part.sphereEnd; i++) {
//            intersectSphere(&scene->spheres[si->spheres[i]], r + si->rays[rx]);
//        }
//        for(size_t i = part.triStart; i < part.triEnd; i++) {
//            intersectTriangle(&scene->tris[si->tris[i]], r + si->rays[rx]);
//        }
//    }
//}

int inum = 0;
int dnum = 0;

//void DACRT(struct DACRTPartition space, struct Ray *r, struct Scene s, struct Camera cam, int depth) {
//    if((space.rayEnd-space.rayStart) < 20 || ((space.triEnd-space.triStart) + (space.sphereEnd-space.sphereStart)) < 16) {
//        dnum += space.rayEnd + space.rayStart + space.triEnd + space.triStart + space.sphereEnd + space.sphereStart;
//        NRT(r, &s, space);
//        return;
//    }
//    enum DivisionAxis axis = depth % 3;
//    struct DuoPartition d2 = averageSpaceCut(space, axis);
//    //float rlength = vec_length(vec_sub(vec_mid(d2.part[1].bounds.min, d2.part[1].bounds.max), cam.center));
//    //float llength = vec_length(vec_sub(vec_mid(d2.part[0].bounds.min, d2.part[0].bounds.max), cam.center));
//    float rlength = d2.part[0].bounds.min.xyz[axis];
//    float llength = d2.part[1].bounds.min.xyz[axis];
//    if(rlength > llength) {
//        struct DACRTPartition p = d2.part[0];
//        d2.part[0] = d2.part[1];
//        d2.part[1] = p;
//    }
//    for(int ps = 0; ps < 2; ps++) {
//        int pivot = space.rayStart;
//        int tpivot = space.triStart;
//        int spivot = space.sphereStart;
//        for(int i = pivot; i < space.rayEnd; i++) {
//            float t = INFINITY;
////            struct vec3 o = r[i].origin;
////            struct vec3 bcmid = vec_mid(d2.part[ps].bounds.min, d2.part[ps].bounds.max);
////            struct vec3 bomid = vec_mid(d2.part[(ps + 1) % 2].bounds.min, d2.part[(ps + 1) % 2].bounds.max);
////            struct vec3 ts = vec_sub(o, bomid);
////            struct vec3 tf = vec_sub(o, bcmid);
////            float vl = vec_length(ts);
////            float vlo  = vec_length(tf);
//            float tother = INFINITY;
//            //tother += 0.001f;
////            int hitother = AABBintersection(d2.part[(ps + 1) % 2].bounds, r[i], &tother);
//            int hit = AABBintersection(d2.part[ps].bounds, r[i], &t);
//            int terminated = r[i].t < t; /*&& (tother < t) && hitother;*/
//            if(hit && !terminated) {
//                list_swap(r + i, r + pivot, sizeof(struct Ray));
//                pivot++;
//            }
//        }
//        for(int i = tpivot; i < space.triEnd; i++) {
//            struct vec3 pt0 = s.tris[i].pt0;
//            struct vec3 pt1 = vec_add(pt0, s.tris[i].u);
//            struct vec3 pt2 = vec_add(pt0, s.tris[i].v);
//            float v = (ps == 0) ? fmin(pt0.xyz[axis], fmin(pt1.xyz[axis], pt2.xyz[axis])) : fmax(pt0.xyz[axis], fmax(pt1.xyz[axis], pt2.xyz[axis]));
//            if((ps == 0 && v < d2.part[0].bounds.max.xyz[axis] + 0.01f) || (ps != 0 && v > d2.part[0].bounds.max.xyz[axis] - 0.01f)) {
//            //if(1) {
//                list_swap(s.tris + i, s.tris + tpivot, sizeof(struct Triangle));
//                tpivot++;
//            }
//        }
//        for(int i = spivot; i < space.sphereEnd; i++) {
//            float a = s.spheres[i].origin.xyz[axis] + s.spheres[i].radius;
//            float b = s.spheres[i].origin.xyz[axis] - s.spheres[i].radius;
//            float v = (ps == 0) ? fmin(a, b) : fmax(a, b);
//            if((ps == 0 && v < d2.part[0].bounds.max.xyz[axis]) || (ps == 1 && v > d2.part[0].bounds.max.xyz[axis])) {
//                list_swap(s.spheres + i, s.spheres + pivot, sizeof(struct Sphere));
//                spivot++;
//            }
//        }
//        struct DACRTPartition p;
//        p.bounds = d2.part[ps].bounds;
//        p.rayEnd = pivot;
//        p.rayStart = space.rayStart;
//        p.sphereEnd = spivot;
//        p.sphereStart = space.sphereStart;
//        p.triEnd = tpivot;
//        p.triStart = space.triStart;
//        dnum++;
//        DACRT(p, r, s, cam, depth + 1);
//    }

//}

//void NewDACRTIndirect(struct DACRTPartition space, struct Ray *r, struct Scene s, struct Camera cam, int depth, struct SceneIndirect si) {
//    if((space.rayEnd-space.rayStart) < 20 || ((space.triEnd-space.triStart) + (space.sphereEnd-space.sphereStart)) < 16) {
//        inum += space.rayEnd + space.rayStart + space.triEnd + space.triStart + space.sphereEnd + space.sphereStart;
//        //NRTIndirect(r, &s, space, &si);
//        return;
//    }
//    enum DivisionAxis axis = depth % 3;
//    struct DuoPartition d2 = averageSpaceCut(space, axis);
//    //float rlength = vec_length(vec_sub(vec_mid(d2.part[1].bounds.min, d2.part[1].bounds.max), cam.center));
//    //float llength = vec_length(vec_sub(vec_mid(d2.part[0].bounds.min, d2.part[0].bounds.max), cam.center));
//    float rlength = d2.part[0].bounds.min.xyz[axis];
//    float llength = d2.part[1].bounds.min.xyz[axis];
//    if(rlength > llength) {
//        struct DACRTPartition p = d2.part[0];
//        d2.part[0] = d2.part[1];
//        d2.part[1] = p;
//    }
//    for(int ps = 0; ps < 2; ps++) {
//        int pivot = space.rayStart;
//        int tpivot = space.triStart;
//        int spivot = space.sphereStart;
//        for(int i = pivot; i < space.rayEnd; i++) {
//            float t = INFINITY;
////            struct vec3 o = r[i].origin;
////            struct vec3 bcmid = vec_mid(d2.part[ps].bounds.min, d2.part[ps].bounds.max);
////            struct vec3 bomid = vec_mid(d2.part[(ps + 1) % 2].bounds.min, d2.part[(ps + 1) % 2].bounds.max);
////            struct vec3 ts = vec_sub(o, bomid);
////            struct vec3 tf = vec_sub(o, bcmid);
////            float vl = vec_length(ts);
////            float vlo  = vec_length(tf);
//            float tother = INFINITY;
//            //tother += 0.001f;
//            int reali = si.rays[i];
////            int hitother = AABBintersection(d2.part[(ps + 1) % 2].bounds, r[i], &tother);
//            int hit = AABBintersection(d2.part[ps].bounds, r[reali], &t);
//            int terminated = r[reali].t < t; /*&& (tother < t) && hitother;*/
//            if(hit && !terminated) {
//                //list_swap(r + i, r + pivot, sizeof(struct Ray));
//                si.rays[i] = si.rays[pivot];
//                si.rays[pivot] = reali;
//                pivot++;
//            }
//        }
//        for(int i = tpivot; i < space.triEnd; i++) {
//            int reali = si.tris[i];
//            struct vec3 pt0 = s.tris[reali].pt0;
//            struct vec3 pt1 = vec_add(pt0, s.tris[reali].u);
//            struct vec3 pt2 = vec_add(pt0, s.tris[reali].v);
//            float v = (ps == 0) ? fmin(pt0.xyz[axis], fmin(pt1.xyz[axis], pt2.xyz[axis])) : fmax(pt0.xyz[axis], fmax(pt1.xyz[axis], pt2.xyz[axis]));
//            if((ps == 0 && v < d2.part[0].bounds.max.xyz[axis] + 0.01f) || (ps != 0 && v > d2.part[0].bounds.max.xyz[axis] - 0.01f)) {
//            //if(1) {
//                si.tris[i] = si.rays[tpivot];
//                si.tris[tpivot] = reali;
//                //list_swap(s.tris + i, s.tris + tpivot, sizeof(struct Triangle));
//                tpivot++;
//            }
//        }
//        for(int i = spivot; i < space.sphereEnd; i++) {
//            int reali = si.spheres[i];

//            float a = s.spheres[reali].origin.xyz[axis] + s.spheres[reali].radius;
//            float b = s.spheres[reali].origin.xyz[axis] - s.spheres[reali].radius;
//            float v = (ps == 0) ? fmin(a, b) : fmax(a, b);
//            if((ps == 0 && v < d2.part[0].bounds.max.xyz[axis]) || (ps == 1 && v > d2.part[0].bounds.max.xyz[axis])) {
//                si.spheres[i] = si.spheres[spivot];
//                si.spheres[spivot] = reali;
//                //list_swap(s.spheres + i, s.spheres + pivot, sizeof(struct Sphere));
//                spivot++;
//            }
//        }
//        struct DACRTPartition p;
//        p.bounds = d2.part[ps].bounds;
//        p.rayEnd = pivot;
//        p.rayStart = space.rayStart;
//        p.sphereEnd = spivot;
//        p.sphereStart = space.sphereStart;
//        p.triEnd = tpivot;
//        p.triStart = space.triStart;
//        inum++;
//        NewDACRTIndirect(p, r, s, cam, depth + 1, si);
//    }

//}
size_t xres = 640;
size_t yres = 400;

const GLfloat gv[108] = {
        -1.0f,-1.0f,-1.0f, // triangle 1 : begin
        -1.0f,-1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f, // triangle 1 : end
        1.0f, 1.0f,-1.0f, // triangle 2 : begin
        -1.0f,-1.0f,-1.0f,
        -1.0f, 1.0f,-1.0f, // triangle 2 : end
        1.0f,-1.0f, 1.0f,
        -1.0f,-1.0f,-1.0f,
        1.0f,-1.0f,-1.0f,
        1.0f, 1.0f,-1.0f,
        1.0f,-1.0f,-1.0f,
        -1.0f,-1.0f,-1.0f,
        -1.0f,-1.0f,-1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f,-1.0f,
        1.0f,-1.0f, 1.0f,
        -1.0f,-1.0f, 1.0f,
        -1.0f,-1.0f,-1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f,-1.0f, 1.0f,
        1.0f,-1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f,-1.0f,-1.0f,
        1.0f, 1.0f,-1.0f,
        1.0f,-1.0f,-1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f,-1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f,-1.0f,
        -1.0f, 1.0f,-1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f,-1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f,-1.0f, 1.0f
};

int alignAt(int offset, int alignment) {
    return ((offset + alignment - 1)/alignment) * alignment;
}

struct Scene copyScene(struct Scene s) {
    struct Scene sc;
    size_t page_size = 4096;
    size_t sz = 0;
    for(int i = 0; i < 12; i++) {
        sz += (sizeof(float) * s.numtris) * page_size;
    }
    sc.pt0[0] = malloc(sizeof(float) * s.numtris);
    sc.pt0[1] = malloc(sizeof(float) * s.numtris);
    sc.pt0[2] = malloc(sizeof(float) * s.numtris);
    sc.u[0] = malloc(sizeof(float) * s.numtris);
    sc.u[1] = malloc(sizeof(float) * s.numtris);
    sc.u[2] = malloc(sizeof(float) * s.numtris);
    sc.v[0] = malloc(sizeof(float) * s.numtris);
    sc.v[1] = malloc(sizeof(float) * s.numtris);
    sc.v[2] = malloc(sizeof(float) * s.numtris);
    sc.normal[0] = malloc(sizeof(float) * s.numtris);
    sc.normal[1] = malloc(sizeof(float) * s.numtris);
    sc.normal[2] = malloc(sizeof(float) * s.numtris);
    sc.origins[0] = malloc(sizeof(float) * s.numspheres);
    sc.origins[1] = malloc(sizeof(float) * s.numspheres);
    sc.origins[2] = malloc(sizeof(float) * s.numspheres);
    sc.radius = malloc(sizeof(float) * s.numspheres);
    sc.numspheres = s.numspheres;
    sc.numtris = s.numtris;
    memcpy(sc.pt0[0], s.pt0[0], sizeof(float) * s.numtris);
    memcpy(sc.pt0[1], s.pt0[1], sizeof(float) * s.numtris);
    memcpy(sc.pt0[2], s.pt0[2], sizeof(float) * s.numtris);
    memcpy(sc.u[0], s.u[0], sizeof(float) * s.numtris);
    memcpy(sc.u[1], s.u[1], sizeof(float) * s.numtris);
    memcpy(sc.u[2], s.u[2], sizeof(float) * s.numtris);
    memcpy(sc.v[0], s.v[0], sizeof(float) * s.numtris);
    memcpy(sc.v[1], s.v[1], sizeof(float) * s.numtris);
    memcpy(sc.v[2], s.v[2], sizeof(float) * s.numtris);
    memcpy(sc.normal[0], s.normal[0], sizeof(float) * s.numtris);
    memcpy(sc.normal[1], s.normal[1], sizeof(float) * s.numtris);
    memcpy(sc.normal[2], s.normal[2], sizeof(float) * s.numtris);
    memcpy(sc.origins[0], s.origins[0], sizeof(float) * s.numspheres);
    memcpy(sc.origins[1], s.origins[1], sizeof(float) * s.numspheres);
    memcpy(sc.origins[2], s.origins[2], sizeof(float) * s.numspheres);
    memcpy(sc.radius, s.radius, sizeof(float) * s.numspheres);

    return sc;
}

struct SceneIndirect genIndirect(struct Scene s, int numrays) {
    struct SceneIndirect si;
    si.tris = malloc(s.numtris*sizeof(int));
    si.spheres = malloc(s.numspheres*sizeof(int));
    si.rays = malloc(numrays*sizeof(int));
    for(int i = 0; i < s.numtris; i++) {
        si.tris[i] = i;
    }
    for(int i = 0; i < s.numspheres; i++) {
        si.spheres[i] = i;
    }
    for(int i = 0; i < numrays; i++) {
        si.rays[i] = i;
    }
    return si;
}

void destroyIndirect(struct SceneIndirect si) {
    free(si.spheres);
    free(si.tris);
    free(si.rays);
}

int main(int argc, char* argv[])
{
    glfwInit();
    GLFWwindow *win = glfwCreateWindow(640*4, 400*4, "hi", NULL, NULL);
    glfwMakeContextCurrent(win);
    glewInit();
    glutInit(&argc, argv);
    struct Camera camera;
    struct StorageTriangle t[192*4];
    for(int i = 0; i < 16; i++) {
        float xoff = 3*i;
        for(int k = 0; k < 12; k++) {
            struct vec3 p1;
            struct vec3 p2;
            struct vec3 p3;
            p1.x = gv[k*9] + xoff;
            p1.y = gv[k*9 + 1];
            p1.z = gv[k*9 + 2];
            p2.x = gv[k*9 + 3] + xoff;
            p2.y = gv[k*9 + 4];
            p2.z = gv[k*9 + 5];
            p3.x = gv[k*9 + 6] + xoff;
            p3.y = gv[k*9 + 7];
            p3.z = gv[k*9 + 8];

            t[i*12 + k].pts[0] = p1;
            t[i*12 + k].pts[1] = p2;
            t[i*12 + k].pts[2] = p3;
        }
    }
    for(int i = 0; i < 16; i++) {
        float xoff = 3*i;
        for(int k = 0; k < 12; k++) {
            struct vec3 p1;
            struct vec3 p2;
            struct vec3 p3;
            p1.x = gv[k*9] + xoff;
            p1.y = gv[k*9 + 1] + 3.0f;
            p1.z = gv[k*9 + 2];
            p2.x = gv[k*9 + 3] + xoff;
            p2.y = gv[k*9 + 4] + 3.0f;
            p2.z = gv[k*9 + 5];
            p3.x = gv[k*9 + 6] + xoff;
            p3.y = gv[k*9 + 7] + 3.0f;
            p3.z = gv[k*9 + 8];

            t[192 + i*12 + k].pts[0] = p1;
            t[192 + i*12 + k].pts[1] = p2;
            t[192 + i*12 + k].pts[2] = p3;
        }
    }
    for(int i = 0; i < 16; i++) {
        float xoff = 3*i;
        for(int k = 0; k < 12; k++) {
            struct vec3 p1;
            struct vec3 p2;
            struct vec3 p3;
            p1.x = gv[k*9] + xoff;
            p1.y = gv[k*9 + 1] + 3.0f;
            p1.z = gv[k*9 + 2] + 3.0f;
            p2.x = gv[k*9 + 3] + xoff;
            p2.y = gv[k*9 + 4] + 3.0f;
            p2.z = gv[k*9 + 5] + 3.0f;
            p3.x = gv[k*9 + 6] + xoff;
            p3.y = gv[k*9 + 7] + 3.0f;
            p3.z = gv[k*9 + 8] + 3.0f;

            t[384 + i*12 + k].pts[0] = p1;
            t[384 + i*12 + k].pts[1] = p2;
            t[384 + i*12 + k].pts[2] = p3;
        }
    }
    for(int i = 0; i < 16; i++) {
        float xoff = 3*i;
        for(int k = 0; k < 12; k++) {
            struct vec3 p1;
            struct vec3 p2;
            struct vec3 p3;
            p1.x = gv[k*9] + xoff;
            p1.y = gv[k*9 + 1];
            p1.z = gv[k*9 + 2] + 3.0f;
            p2.x = gv[k*9 + 3] + xoff;
            p2.y = gv[k*9 + 4];
            p2.z = gv[k*9 + 5] + 3.0f;
            p3.x = gv[k*9 + 6] + xoff;
            p3.y = gv[k*9 + 7];
            p3.z = gv[k*9 + 8] + 3.0f;

            t[576 + i*12 + k].pts[0] = p1;
            t[576 + i*12 + k].pts[1] = p2;
            t[576 + i*12 + k].pts[2] = p3;
        }
    }
    struct StorageSphere s;
    camera.center.x = 47.867912f;
    camera.center.y = -0.693855f;
    camera.center.z = -2.437953f;
    camera.lookat.x = -0.704917f;
    camera.lookat.y = 0.281158f;
    camera.lookat.z = 0.651185f;
    //camera.lookat = vec_norm(vec_sub(camera.lookat, camera.center));
    camera.up.x = 0.206372f;
    camera.up.y = 0.959661f;
    camera.up.z = -0.190946f;
    s.origin.x = 0.0f;
    s.origin.y = 1.0f;
    s.radius   = 0.8f;
    s.origin.z = 0.0f;
//    t[0].pts[0].x = -20.0f;
//    t[0].pts[0].z = -20.0f;
//    t[0].pts[0].y =  0.0f;
//    t[0].pts[1].x = 20.0f;
//    t[0].pts[1].z = 20.0f;
//    t[0].pts[1].y =  0.0f;
//    t[0].pts[2].x = 20.0f;
//    t[0].pts[2].z = -20.0f;
//    t[0].pts[2].y =  0.0f;
//    t[1].pts[0].x = -20.0f;
//    t[1].pts[0].z = -20.0f;
//    t[1].pts[0].y =  0.0f;
//    t[1].pts[1].x = -20.0f;
//    t[1].pts[1].z = 20.0f;
//    t[1].pts[1].y =  0.0f;
//    t[1].pts[2].x = 20.0f;
//    t[1].pts[2].z = 20.0f;
//    t[1].pts[2].y =  0.0f;
    unsigned char *fb = malloc(xres*yres*3);
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    float horizontal = 1.5f;
    float vertical = 0.0f;
    glfwSetCursorPos(win, xres/2, yres/2);
    glfwSetInputMode(win , GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
    omp_set_num_threads(8);
    int frame = 0;
    int fps = 0;
    float time = glfwGetTime();  
    while(!glfwWindowShouldClose(win)) {
        memset(fb, 0, xres*yres*3);
        struct Scene scene = generateSceneGraphFromStorage(t, &s, 192*4, 0);
        struct AABB aabb = AABBFromScene(&scene);
        struct DACRTPartition part;
        part.bounds = aabb;
        float t = glfwGetTime();
        float x = 5*sin(t);
        float z = 10*cos(t);
        s.origin.x = x;
        s.origin.z = z;
        int w,h;
        glfwGetWindowSize(win, &w, &h);
        double xpos = w/2, ypos = h/2;
        float mspeed = 0.005f;
        glfwGetCursorPos(win, &xpos, &ypos);
        glfwSetCursorPos(win, w/2, h/2);
        horizontal += mspeed * -(w/2- xpos);
        vertical += mspeed * (h/2 - ypos);

        if (vertical > 1.5f) {
            vertical = 1.5f;
        }
        else if (vertical < -1.5f) {
            vertical = -1.5f;
        }
//        camera.lookat.x = cos(vertical) * sin(horizontal);
//        camera.lookat.y = sin(vertical);
//        camera.lookat.z = cos(horizontal) * cos(vertical);
//        right.x = sin(horizontal - 3.14f / 2.0f);
//        right.y = 0.0f;
//        right.z = cos(horizontal - 3.14f / 2.0f);
//        camera.up = vec_cross(right, camera.lookat);
//        right = vec_mul(right, vec_dup(-1.0f));
        float speedup = 1.0f;
        if(glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) {
            camera.center = vec_add(camera.center,vec_mul(camera.lookat,vec_dup(speedup)));
        }
        if(glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) {
            camera.center = vec_sub(camera.center,vec_mul(camera.lookat,vec_dup(speedup)));
        }
        if(glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) {
            camera.center = vec_sub(camera.center,vec_mul(right,vec_dup(speedup)));
        }
        if(glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) {
            camera.center = vec_add(camera.center,vec_mul(right,vec_dup(speedup)));
        }
        if(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            return -1;
        }
        #pragma omp parallel for
        for(size_t y = 0; y < yres; y++) {
            struct Scene sc = copyScene(scene);
            struct SceneIndirect si = genIndirect(scene, xres);
            float yf = (float)y/(float)yres - 0.5;
            struct Ray r[xres];
            for(size_t x = 0; x < xres; x++) {
                float xf = (float)x/(float)xres - 0.5;
                struct vec3 rightm = vec_mul(right, vec_dup(xf));
                struct vec3 upm = vec_mul(vec_mul(camera.up, vec_dup(0.66f)), vec_dup(yf));
                struct vec3 direction = vec_norm(vec_add(vec_add(upm, rightm), camera.lookat));
                struct vec3 inv_dir;
                inv_dir.x = 1.0f/direction.x;
                inv_dir.y = 1.0f/direction.y;
                inv_dir.z = 1.0f/direction.z;
                r[x].inv_dir = inv_dir;
                r[x].direction = direction;
                r[x].origin = camera.center;
                r[x].bounces = 0;
                r[x].t = INFINITY;
                r[x].id = x;
            }
            struct DACRTPartition p;
            p.bounds = aabb;
            p.rayStart = 0;
            p.rayEnd = xres;
            p.sphereEnd = scene.numspheres;
            p.sphereStart = 0;
            p.triEnd = scene.numtris;
            p.triStart = 0;
            //NewDACRTIndirect(p, r, scene, camera, bestAxis(camera, p), si);
            DACRTWorkingNoEarlyTermIndirect(&p, r, &sc, &camera, bestAxis(camera, p), &si);
            //DACRTWorkingNoEarlyTerm(p, r, sc, camera, bestAxis(camera, p));
            //printf("%d\n", dnum);
            //printf("%d\n", inum);
            //exit(-1);
            //NRT(r, &scene, p);
            #pragma omp simd
            for(size_t x = 0; x < xres; x++) {
                struct vec3 color;
                color.z = ((r[x].t == INFINITY) ? 0.0f : vec_dot(r[x].direction, vec_mul(vec_dup(-1.0f), r[x].normal)) * 1.0f);
                color.x = 0.0f;
                color.y = 0.0f;
                //color[0] = RT(&r, &scene, 0);
                //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
                fb[y*3*xres + r[x].id*3] = fastPow(color.x, 1 / 2.2f)*255;
                fb[y*3*xres + r[x].id*3 + 1] = fastPow(color.y, 1 / 2.2f)*255;
                fb[y*3*xres + r[x].id*3 + 2] = fastPow(color.z, 1 / 2.2f)*255;
            }
            destroyIndirect(si);
            deallocScene(sc);
        }
        glMatrixMode(GL_PROJECTION);
        gluOrtho2D(0, 100, 0, 100);
        glRasterPos2i(0, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        glPixelZoom(4.0f, 4.0f);
        glDrawPixels(640, 400, GL_RGB, GL_UNSIGNED_BYTE, fb);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        char *s = vec_sprint(camera.center);
        glRasterPos2i(1, 1);
        glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)s);
        free(s);
        s = vec_sprint(camera.lookat);
        glRasterPos2i(1, 3);
        glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)s);
        free(s);
        glRasterPos2i(1, 5);
        glutBitmapString(GLUT_BITMAP_8_BY_13, axisString(bestAxis(camera, part)));
        s = vec_sprint(camera.up);
        glRasterPos2i(1, 7);
        glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)s);
        free(s);
        frame++;
        float ctime = glfwGetTime();
        if(ctime > time+1) {
            float tdiff = ctime-time;
            time = ctime;
            fps = frame/tdiff;
            frame = 0;
        }
        char fp[(int)((ceil(log10(fps+1))+6)*sizeof(char))];
        sprintf(fp, "FPS: %d", fps);
        glRasterPos2i(1, 9);
        glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)fp);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glfwSwapBuffers(win);
        glfwPollEvents();
        deallocScene(scene);
    }
    return 0;
}
