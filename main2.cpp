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
#include <stack>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
extern "C"
{
#include <libfont.h>
#include <veclib.h>
#include <vector.h>
}
#include "obj.h"
#include <glm/glm.hpp>
#include <glm/gtx/projection.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <libopenmpt/libopenmpt.h>
#include <libopenmpt/libopenmpt_stream_callbacks_file.h>
#include <portaudio.h>
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

std::map<int32_t, std::map<int32_t, int32_t>> effectsforpattern;

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

enum ObjectType {
    Sphere,
    Triangle
};

float fall  = 0.0f;

void thread() {
    Pa_Initialize();
    int interpol = 1;
    int ss = 100;
    const int BUFFERSIZE = 2000;
    const int SAMPLERATE = 48000;
    static int16_t left[BUFFERSIZE];
    static int16_t right[BUFFERSIZE];
    static int16_t * const buffers[2] = { left, right };
    FILE *file;
    openmpt_module * mod = 0;
    size_t count = 0;
    PaStream * stream = 0;
    Pa_OpenDefaultStream(&stream, 0, 2, paInt16 | paNonInterleaved, SAMPLERATE, paFramesPerBufferUnspecified, NULL, NULL);
    Pa_StartStream(stream);

    file = fopen("song.mptm", "rb");
    if(file != NULL)
    {

        mod = openmpt_module_create(openmpt_stream_get_file_callbacks(), file, NULL, NULL, NULL);
        openmpt_module_set_render_param(mod, OPENMPT_MODULE_RENDER_INTERPOLATIONFILTER_LENGTH, interpol);
        openmpt_module_set_render_param(mod, OPENMPT_MODULE_RENDER_STEREOSEPARATION_PERCENT, ss);
        fclose(file);
        while (1) {

            count = openmpt_module_read_stereo(mod, SAMPLERATE, BUFFERSIZE, left, right);
            if (count == 0) {
                break;
            }
            Pa_WriteStream(stream, buffers, (unsigned long)count);
            if(effectsforpattern[openmpt_module_get_current_pattern(mod)][openmpt_module_get_current_row(mod)]) {
                std::mutex mtx;
                std::lock_guard<std::mutex> g(mtx);
                fall += 1.0f;
            }
            effectsforpattern[openmpt_module_get_current_pattern(mod)][openmpt_module_get_current_pattern(mod)] = 0;
        }

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        openmpt_module_destroy(mod);
    }
}

struct vec3 red(float u, float v, float t) {
    struct vec3 r;
    r.x = 1.0f*(1/(t+1));
    r.y = 0.0f;
    r.z = 0.0f;
    return r;
}

struct vec3 checker(float u, float v, float t) {
    u = u;
    v = v;
    struct vec3 black;
    black.x = 0.0f;
    black.y = 0.0f;
    black.z = 0.0f;
    struct vec3 white;
    white.x = 1.0f;
    white.y = 1.0f;
    white.z = 1.0f;
    white = vec_mul(vec_dup(1/(t+1)),white);
    int u8 = floor(u*4);
    int v8 = floor(v*4);
    if(((u8+v8) % 2) == 0) {
        return black;
    }
    return white;
}

struct Material {
    int reflect;
    int diffuse;
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

struct AABB AABBFromSceneAOS(struct SceneAOS *s) {
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
        pt0 = s->tris[i].pt0;
        struct vec3 pt1;
        pt1 = vec_add(pt0, s->tris[i].u);
        struct vec3 pt2;
        pt2 = vec_add(pt0, s->tris[i].v);
        minimum.x = fmin(fmin(fmin(pt0.x, pt1.x), pt2.x), minimum.x);
        minimum.y = fmin(fmin(fmin(pt0.y, pt1.y), pt2.y), minimum.y);
        minimum.z = fmin(fmin(fmin(pt0.z, pt1.z), pt2.z), minimum.z);
        maximum.x = fmax(fmax(fmax(pt0.x, pt1.x), pt2.x), maximum.x);
        maximum.y = fmax(fmax(fmax(pt0.y, pt1.y), pt2.y), maximum.y);
        maximum.z = fmax(fmax(fmax(pt0.z, pt1.z), pt2.z), maximum.z);
    }
    for(size_t i = 0; i < s->numspheres; i++) {
        const struct vec3 origin = s->spheres[i].origin;
        const float radius = s->spheres[i].radius;
        struct vec3 pt0 = vec_sub(origin, vec_dup(radius));
        struct vec3 pt1 = vec_add(origin, vec_dup(radius));
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

int intersectSphereAOS(struct SceneAOS *s, int i, struct Ray *r) {
    struct vec3 origin;
    origin = s->spheres[i].origin;
    float radius = s->spheres[i].radius;
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
    r->m = s->spheres[i].m;
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

int intersectTriangleAOS(struct Triangle *tri, struct Ray *r) {
    struct vec3 pt0 = tri->pt0;
    struct vec3 ut = tri->u;
    struct vec3 vt = tri->v;
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
    r->normal = tri->normal;
    r->m = tri->m;
    r->u = u;
    r->v = v;
    return 1;
}

struct StorageTriangle {
    struct vec3 pts[3];
    struct Material mat;
};

struct StorageSphere {
    struct vec3 origin;
    float radius;
    struct Material mat;
};

struct SceneAOS generateSceneGraphFromStorageAOS(struct StorageTriangle *tris, struct StorageSphere *spheres, size_t numtris, size_t numspheres) {
    struct SceneAOS scene;
    scene.numtris = numtris;
    scene.numspheres = numspheres;
    scene.tris = malloc(sizeof(struct Triangle) * scene.numtris);
    scene.spheres = malloc(sizeof(struct Sphere) * scene.numspheres);
    for(size_t i = 0; i < scene.numtris; i++) {
        struct vec3 pt0 = tris[i].pts[0];
        struct vec3 u = vec_sub(tris[i].pts[1], tris[i].pts[0]);
        struct vec3 v = vec_sub(tris[i].pts[2], tris[i].pts[0]);
        scene.tris[i].pt0 = pt0;
        scene.tris[i].u = u;
        scene.tris[i].v = v;
        scene.tris[i].normal = vec_norm(vec_cross(u, v));
        scene.tris[i].m = tris[i].mat;
    }
    for(size_t i = 0; i < scene.numspheres; i++) {
        scene.spheres[i].origin = spheres[i].origin;
        scene.spheres[i].radius = spheres[i].radius;
        scene.spheres[i].m = spheres[i].mat;

    }
    return scene;
}
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
const vectorizedTriangleAOS(struct Ray *r, struct Triangle *tris, int nValid) {
    __m128
}

void NRTIndirectAOS(struct Ray *r, struct SceneAOS *scene, struct DACRTPartition *part, struct SceneIndirect *si) {
    for(int rx = part->rayStart; rx < part->rayEnd; rx++) {
        for(size_t i = part->sphereStart; i < part->sphereEnd; i++) {
            intersectSphereAOS(scene, si->spheres[i], r + si->rays[rx]);
        }
        const int granularity = 4;
        const int bitmask = 0x03;
        for(size_t i = part->triStart; i < part->triEnd; i += granularity) {
            struct Triangle grabx[granularity];
            int nValid = 0;
            for(size_t x = i; (x < part->triEnd) && (x < i + granularity); x++) {
                grabx[x & bitmask] = *(scene->tris + si->tri[i]);
                nValid++;
            }
            vectorizedTriangleAOS(r, grabx, nValid);
            //intersectTriangleAOS(scene->tris + si->tris[i], r + si->rays[rx]);
        }
    }
}

void DACRTWorkingNoEarlyTermIndirectAOS(struct DACRTPartition *space, struct Ray *r, struct SceneAOS *s, struct Camera *cam, int depth, struct SceneIndirect *si) {
    if((space->rayEnd-space->rayStart) < 20 || ((space->triEnd-space->triStart) + (space->sphereEnd-space->sphereStart)) < 16) {
        NRTIndirectAOS(r, s, space, si);
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
            struct Triangle t = s->tris[trueitem];
            float p1, p2, p3;
            p1 = t.pt0.xyz[axis];
            p2 = p1 + t.u.xyz[axis];
            p3 = p1 + t.v.xyz[axis];
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
            struct Sphere sph = s->spheres[trueitem];
            float a = sph.origin.xyz[axis] + sph.radius;
            float b = sph.origin.xyz[axis] - sph.radius;
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
        DACRTWorkingNoEarlyTermIndirectAOS(&p, r, s, cam, depth + 1, si);
    }

}

void NRT(struct Ray *r, struct SceneAOS *scene, struct DACRTPartition *part) {
    for(int rx = part->rayStart; rx < part->rayEnd; rx++) {
        for(size_t i = part->sphereStart; i < part->sphereEnd; i++) {
            intersectSphereAOS(scene, i, r + rx);
        }
        for(size_t i = part->triStart; i < part->triEnd; i++) {
            intersectTriangleAOS(scene->tris + i, r + rx);
        }
    }
}

uint32_t reduce(uint32_t x, uint32_t N) {
    return ((uint64_t) x * (uint64_t) N) >> 32 ;
}

enum DivisionAxis longestAxis(struct AABB a) {
    float xlength = fabs(a.max.x - a.min.x);
    float ylength = fabs(a.max.y - a.min.y);
    float zlength = fabs(a.max.z - a.min.z);
    float longest = fmax(xlength, fmax(ylength, zlength));
    if(xlength >= longest) {
        return X;
    }
    if(ylength >= longest) {
        return Y;
    }
    if(zlength >= longest) {
        return Z;
    }
    exit(-1);
}


void DACRTWorkingNoEarlyTermAOS(struct DACRTPartition *space, struct Ray *r, struct SceneAOS *s, struct Camera *cam, int depth) {
    if((space->rayEnd-space->rayStart) < 20 || ((space->triEnd-space->triStart) + (space->sphereEnd-space->sphereStart)) < 16) {
        NRT(r, s, space);
        return;
    }
    enum DivisionAxis axis = longestAxis(space->bounds);
    struct DuoPartition d2 = averageSpaceCut(*space, axis);
    //float rlength = vec_length(vec_sub(vec_mid(d2.part[1].bounds.min, d2.part[1].bounds.max), cam->center));
    //float llength = vec_length(vec_sub(vec_mid(d2.part[0].bounds.min, d2.part[0].bounds.max), cam->center));
    const float rlength = d2.part[0].bounds.min.xyz[axis];
    const float llength = d2.part[1].bounds.min.xyz[axis];
    if(rlength > llength) {
        struct DACRTPartition p = d2.part[0];
        d2.part[0] = d2.part[1];
        d2.part[1] = p;
    }
    for(int ps = 0; ps < 2; ps++) {
        int pivot = space->rayStart;
        int tpivot = space->triStart;
        int spivot = space->sphereStart;
        for(int i = pivot; i < space->rayEnd; i++) {
            float t = INFINITY;
            const int hit = AABBintersection(d2.part[ps].bounds, r + i, &t);
            const int terminated = r[i].t < t; /*&& (tother < t) && hitother;*/
            if(hit && !terminated) {
                if(i != pivot) {
                    list_swap(r + pivot, r + i, sizeof(struct Ray));
                }
                pivot++;
            }
        }
        for(int i = tpivot; i < space->triEnd; i++) {
            const struct Triangle t = s->tris[i];
            float p1, p2, p3;
            p1 = t.pt0.xyz[axis];
            p2 = p1 + t.u.xyz[axis];
            p3 = p1 + t.v.xyz[axis];
            const float v = (!ps) ? fmin(p1, fmin(p2, p3)) : fmax(p1, fmax(p2, p3));
            if((!ps && v < d2.part[0].bounds.max.xyz[axis] + 0.01f) || (ps && v > d2.part[0].bounds.max.xyz[axis] - 0.01f)) {
                if(i != tpivot) {
                    list_swap(s->tris + tpivot, s->tris + i, sizeof(struct Triangle));
                }
                tpivot++;
            }
        }
        for(int i = spivot; i < space->sphereEnd; i++) {
            struct Sphere sph = s->spheres[i];
            float a = sph.origin.xyz[axis] + sph.radius;
            float b = sph.origin.xyz[axis] - sph.radius;
            float v = (!ps) ? fmin(a, b) : fmax(a, b);
            if((!ps && v < d2.part[0].bounds.max.xyz[axis]) || (ps && v > d2.part[0].bounds.max.xyz[axis])) {
                if(i != spivot) {
                    list_swap(s->spheres + spivot, s->spheres + i, sizeof(struct Sphere));
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
        DACRTWorkingNoEarlyTermAOS(&p, r, s, cam, depth + 1);
    }

}

void DACRTWorkingNoEarlyTermAOSIndirect2(struct DACRTPartition *space, struct Ray *r, struct SceneAOS *s, struct Camera *cam, int depth, struct SceneIndirect *si) {
    if((space->rayEnd-space->rayStart) < 20 || ((space->triEnd-space->triStart) + (space->sphereEnd-space->sphereStart)) < 16) {
        NRTIndirectAOS(r, s, space, si);
        return;
    }
    enum DivisionAxis axis = longestAxis(space->bounds);
    struct DuoPartition d2 = averageSpaceCut(*space, axis);
    //float rlength = vec_length(vec_sub(vec_mid(d2.part[1].bounds.min, d2.part[1].bounds.max), cam->center));
    //float llength = vec_length(vec_sub(vec_mid(d2.part[0].bounds.min, d2.part[0].bounds.max), cam->center));
    const float rlength = d2.part[0].bounds.min.xyz[axis];
    const float llength = d2.part[1].bounds.min.xyz[axis];
    if(rlength > llength) {
        struct DACRTPartition p = d2.part[0];
        d2.part[0] = d2.part[1];
        d2.part[1] = p;
    }
    for(int ps = 0; ps < 2; ps++) {
        int pivot = space->rayStart;
        int tpivot = space->triStart;
        int spivot = space->sphereStart;
        for(int i = pivot; i < space->rayEnd; i++) {
            int trueitem = si->rays[i];
            float t = INFINITY;
            const int hit = AABBintersection(d2.part[ps].bounds, r + trueitem, &t);
            const int terminated = r[trueitem].t < t; /*&& (tother < t) && hitother;*/
            if(hit && !terminated) {
                if(i != pivot) {
                    si->rays[i] = si->rays[pivot];
                    si->rays[pivot] = trueitem;
                    //list_swap(r + pivot, r + i, sizeof(struct Ray));
                }
                pivot++;
            }
        }
        for(int i = tpivot; i < space->triEnd; i++) {
            int trueitem = si->tris[i];
            const struct Triangle t = s->tris[trueitem];
            float p1, p2, p3;
            p1 = t.pt0.xyz[axis];
            p2 = p1 + t.u.xyz[axis];
            p3 = p1 + t.v.xyz[axis];
            const float v = (!ps) ? fmin(p1, fmin(p2, p3)) : fmax(p1, fmax(p2, p3));
            if((!ps && v < d2.part[0].bounds.max.xyz[axis] + 0.01f) || (ps && v > d2.part[0].bounds.max.xyz[axis] - 0.01f)) {
                if(i != tpivot) {
                    si->tris[i] = si->tris[tpivot];
                    si->tris[tpivot] = trueitem;
                    //list_swap(s->tris + tpivot, s->tris + i, sizeof(struct Triangle));
                }
                tpivot++;
            }
        }
        for(int i = spivot; i < space->sphereEnd; i++) {
            struct Sphere sph = s->spheres[i];
            float a = sph.origin.xyz[axis] + sph.radius;
            float b = sph.origin.xyz[axis] - sph.radius;
            float v = (!ps) ? fmin(a, b) : fmax(a, b);
            if((!ps && v < d2.part[0].bounds.max.xyz[axis]) || (ps && v > d2.part[0].bounds.max.xyz[axis])) {
                if(i != spivot) {
                    list_swap(s->spheres + spivot, s->spheres + i, sizeof(struct Sphere));
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
        DACRTWorkingNoEarlyTermAOSIndirect2(&p, r, s, cam, depth + 1, si);
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
size_t xres = 960;
size_t yres = 540;

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

void drawtexture(uint8_t *fb, uint16_t xfb, uint16_t yfb, uint16_t offsetx, uint16_t offsety, struct Texture t) {
    for(int y = 0; y < t.y; y++) {
        for(int x = 0; x < t.x; x++) {
            float alpha = t.data[y*t.x*4 + t.x*4 + 3]/255.0f;
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3)] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3)] + t.data[y*t.x*4 + t.x*4]*(1-alpha);
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 1] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 1] + t.data[y*t.x*4 + t.x*4 + 1]*(1-alpha);
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 2] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 2] + t.data[y*t.x*4 + t.x*4 + 2]*(1-alpha);

        }
    }
}

struct SceneAOS copySceneAOS(struct SceneAOS s) {
    struct SceneAOS sc;
    sc.numspheres = s.numspheres;
    sc.numtris = s.numtris;
    sc.tris = malloc(sizeof(struct Triangle) * s.numtris);
    sc.spheres = malloc(sizeof(struct Sphere) * s.numspheres);
    memcpy(sc.tris, s.tris, sizeof(struct Triangle) * s.numtris);
    memcpy(sc.spheres, s.spheres, sizeof(struct Sphere) * s.numspheres);
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

struct SceneIndirect genIndirectAOS(struct SceneAOS s, int numrays) {
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

struct Effect {
    float length;
    void (*func)(float, uint8_t*);
};

struct Star {
    struct vec3 location;
    float speed;
};

float clip(float n, float lower, float upper) {
    return fmax(lower, fmin(n, upper));
}

struct vec3 vec_make(float x, float y, float z) {
    struct vec3 xyz;
    xyz.x = x;
    xyz.y = y;
    xyz.z = z;
    return xyz;
}

size_t yscr = 1080;
size_t xscr = 1920;


struct Star field[1000];
struct Star newfield[1000];

float cmul = 1.0f;

void starterf(float time, uint8_t *fb) {
    glViewport(0, 0, 1920, 1080);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glMatrixMode(GL_PROJECTION);
    gluPerspective(45.0f, 16/9.0f, 0.001f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);
    gluLookAt(0.0, 0.0, 0.0, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    for(int i = 0; i < 1000; i++) {
        glColor3f(1.0f, 1.0f, 1.0f);
        glm::mat4 mat;
        mat = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::vec4 pt(field[i].location.x, field[i].location.y, field[i].location.z, 1.0f);
        glm::mat4 mat2;
        mat2 = glm::perspective(45.0f, 16.0f/9.0f, 0.001f, 1000.0f);
        pt = mat * pt;
        pt = mat2 * pt;
        float z = (pt.z/100.0f);
        glColor3f(z, z, z);
        glPointSize(10.0f);
        glBegin(GL_POINTS);
        glVertex3f(field[i].location.x, field[i].location.y, field[i].location.z);
        glEnd();
    }
    glColor3f(1.0f, 1.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float namefade = clip(time - 0.0f, 0.0f, 0.8f);
    float blitfade = clip(time - 6.0f, 0.0f, 1.0f) * cmul;
    float asmfade  = (clip(time - 4.0f, 0.0f, 0.8f)/2.0f);
    float madewithfade = (clip(time - 18.5f, 0.0f, 0.8f)/2.0f);
    unsigned char *name = drawText("name presents", vec_dup(namefade));
    unsigned char *superblit = drawText("MISSING THE DEADLINE", vec_dup(blitfade));
    unsigned char *assembly = drawText("for assembly 18", vec_dup(asmfade));
    unsigned char *madewith = drawText("made with 100% organic recycled demo parts", vec_dup(madewithfade));

    struct Texture t[4];
    t[0].data = name;
    t[0].x = (9 * strlen("name presents"));
    t[0].y = 15;
    t[0].scale = 1.0f;
    t[0].yoff = 98.0f;
    t[0].xoff = 50.0f;
    t[1].data = superblit;
    t[1].x = (9 * strlen("MISSING THE DEADLINE"));
    t[1].y = 15;
    t[1].scale = 8.0f;
    //t[1].scale = 4.0f;
    t[1].xoff = 50.0f + 25.0f;
    t[1].yoff = 75.0f;
    t[2].data = assembly;
    t[2].x = (9 * strlen("for assembly 18"));
    t[2].y = 15;
    t[2].scale = 1.5f;
    t[2].yoff = 2.5f;
    t[2].xoff = 85.0f;
    t[3].data = madewith;
    t[3].x = (9 * strlen("made with 100% organic recycled demo parts"));
    t[3].y = 15;
    t[3].scale = 1.5f;
    t[3].yoff = 65.0f;
    t[3].xoff = 50.0f;
    for(int i = 0; i < 4; i++) {
        GLuint tex;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
        float xsz = ((t[i].x)/(float)xscr)*100;
        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
        glEnd();
        glDeleteTextures(1, &tex);
        glDisable(GL_TEXTURE_2D);
    }
    float flash = clip(time-25.2f, 0.0f, 0.2f)*5.0f;
    if(flash != 0.0f && flash < 1.0f) {
        uint8_t fbs[4];
        fbs[0] = flash*255;
        fbs[1] = flash*255;
        fbs[2] = flash*255;
        fbs[3] = flash*255;
        glPixelZoom(xscr, yscr);
        glDrawPixels(1,1, GL_RGBA, GL_UNSIGNED_BYTE, fbs);
    }
    glDisable(GL_DEPTH_TEST);
}

void plasmaf(float time, uint8_t *fb) {
}

void starfield3df(float time, uint8_t *fb) {
    glViewport(0, 0, xscr, yscr);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, 4/3.0f, 0.1f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glm::mat4 mat;
    gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
    //mat = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    //glLoadMatrixf(&mat[0][0]);
    ///gluLookAt(1.0, 0.0, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    //glScalef(10.0f, 10.0f, 10.0f);
//    for(int i = 0; i < 1000; i++) {
//        glColor3f(1.0f, 1.0f, 1.0f);
//        glm::vec4 pt(newfield[i].location.x, newfield[i].location.y, newfield[i].location.z, 1.0f);
//        glm::mat4 mat2;
//        mat2 = glm::perspective(45.0f, 4.0f/3.0f, 0.001f, 1000.0f);
//        pt = mat * pt;
//        pt = mat2 * pt;
//        float z = pt.z/100.0f;
//        glScalef(1.0f, 1.0f, 1.0f);
//        glColor3f(newfield[i].location.x/100.0f, newfield[i].location.y/100.0f, newfield[i].location.z/100.0f);
//        glPointSize(10.0f);
//        glBegin(GL_POINTS);
//        glVertex3f(newfield[i].location.x, newfield[i].location.y, newfield[i].location.z);
//        glEnd();
//    }
    glPointSize(10.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS);
    glVertex3f(0, 0, 0);
    glEnd();
    glColor3f(1.0f, 1.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

struct Scene scene;
struct SceneAOS sceneaos;
float velocity = 0.0f;
float acceleration = -0.098f;
float lastdiff = 0.0f;
void spotlightf(float time, uint8_t *fb) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
    float diff = time - lastdiff;
    lastdiff = time;
    velocity += acceleration;
    sceneaos.spheres[0].origin.y += velocity*diff;
    if(sceneaos.spheres[0].origin.y < 0.2) {
        velocity = 4.0f;
    }
    float want = clip(time - 5.0f, 0.0f, 1.0f);
    const char *makeitend;
    if(time > 5.0f && time < 10.0f) {
        makeitend = "right?";
    } else if (time > 10.0f) {
        makeitend = "are you saying no?";
    }else {
        makeitend = "raytracing?";
    }
    const char *stop;
    if(time < 10.0f) {
        stop = "this is what you wanted, right?";
    } else {
        stop = "you hate me, don't you?";
    }
    unsigned char *name = drawText(stop, vec_dup(want));
    unsigned char *raytracing = drawText(makeitend, vec_dup(1.0f));
    const int amt = 150;
    struct Texture t[amt];
    t[0].data = name;
    t[0].x = (9 * strlen(stop));
    t[0].y = 15;
    t[0].scale = 1.0f;
    t[0].yoff = 98.0f;
    t[0].xoff = 50.0f;
    for(int i = 1; i < 50 + clip(time - 12.0f, 0.0f, 3.0f)*33; i++) {
        t[i].data = raytracing;
        t[i].x = 9*strlen(makeitend);
        t[i].y = 15;
        t[i].scale = 1.0f + ((next() % 9)-4)/8.0f;
        t[i].xoff = next() % 100;
        t[i].yoff = next() % 100;
    }
    for(int i = 0; i < 50 + clip(time - 12.0f, 0.0f, 3.0f)*33; i++) {
        GLuint tex;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
        float xsz = ((t[i].x)/(float)xscr)*100;
        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
        glEnd();
        glDeleteTextures(1, &tex);
        glDisable(GL_TEXTURE_2D);
    }
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    struct Camera camera;
    camera.center.x = -0.4f-time;
    camera.center.y = 1.0f;
    camera.center.z = 0.0f;
    struct vec3 a;
    a.x = 1.0f;
    a.y = 0.0f;
    a.z = 0.0f;
    camera.lookat = a;
    camera.up.x = 0.0f;
    camera.up.y = 1.0f;
    camera.up.z = 0.0f;
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    struct AABB aabb = AABBFromSceneAOS(&sceneaos);
#pragma omp parallel for
    for(size_t y = 0; y < yres; y++) {
        struct SceneAOS sc = copySceneAOS(sceneaos);
        struct SceneIndirect si = genIndirectAOS(sceneaos, xres);
        float yf = (float)y/(float)yres - 0.5;
        struct Ray r[xres];
        for(size_t x = 0; x < xres; x++) {
            float xf = (float)x/(float)xres - 0.5;
            struct vec3 rightm = vec_mul(right, vec_dup(xf));
            struct vec3 upm = vec_mul(vec_mul(camera.up, vec_dup(0.625f)), vec_dup(yf));
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
        p.sphereEnd = sceneaos.numspheres;
        p.sphereStart = 0;
        p.triEnd = sceneaos.numtris;
        p.triStart = 0;
        //NRTIndirectAOS(r, &sceneaos, &p, &si);
        //NewDACRTIndirect(p, r, scene, camera, bestAxis(camera, p), si);
        //DACRTWorkingNoEarlyTermAOSIndirect2(&p, r, &sc, &camera, bestAxis(camera, p), &si);
        DACRTWorkingNoEarlyTermAOS(&p, r, &sc, &camera, 0);
        //DACRTWorkingNoEarlyTermIndirectAOS(&p, r, &sceneaos, &camera, bestAxis(camera, p), &si);
        //DACRTWorkingNoEarlyTerm(p, r, sc, camera, bestAxis(camera, p));
        //printf("%d\n", dnum);
        //printf("%d\n", inum);
        //exit(-1);
        //NRT(r, &scene, p);
#pragma omp simd
        for(size_t x = 0; x < xres; x++) {
            float fact = (r[x].t == INFINITY) ? 0.0f : vec_dot(r[x].direction, vec_mul(vec_dup(-1.0f), r[x].normal));
            struct vec3 color;
            if(fact > 0.0f) {
                color = vec_mul(vec_dup(fact), r[x].m.eval(r[x].u, r[x].v, r[x].t));
            } else {
                color.x = 0.0f;
                color.y = 0.0f;
                color.z = 0.0f;
            }
            //color[0] = RT(&r, &scene, 0);
            //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
            fb[y*3*xres + r[x].id*3] = fastPow(color.x, 1 / 2.2f)*255;
            fb[y*3*xres + r[x].id*3 + 1] = fastPow(color.y, 1 / 2.2f)*255;
            fb[y*3*xres + r[x].id*3 + 2] = fastPow(color.z, 1 / 2.2f)*255;
        }
        //destroyIndirect(si);
        //deallocScene(sc);
    }
    glRasterPos2i(0, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glPixelZoom((float)xscr/xres, (float)yscr/yres);
    glDrawPixels(xres, yres, GL_RGB, GL_UNSIGNED_BYTE, fb);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    float flash = clip(time-19.8f, 0.0f, 0.2f)*5.0f;
    if(flash != 0.0f && flash < 1.0f) {
        uint8_t fbs[4];
        fbs[0] = 0;
        fbs[1] = 0;
        fbs[2] = 0;
        fbs[3] = flash*255;
        glPixelZoom(xscr, yscr);
        glDrawPixels(1,1, GL_RGBA, GL_UNSIGNED_BYTE, fbs);
    }
    //        char *s = vec_sprint(camera.center);
}

void creditsf(float time, uint8_t *fb) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float px = 19.0f/yscr * 200.0f;
    const int len = 4;
    const char *text[len] = { "code: name", "music: name", "design: name's insecurities", "bye now :>"};
    struct Texture t[len];
    for(int i = 0; i < len; i++) {
        t[i].data = drawText(text[i], vec_dup(0.99f));
        t[i].yoff = 100 -(px*(i+1)) + sin(time+i);
        t[i].x = 9*strlen(text[i]);
        t[i].y = 15;
        if(i == 3) {
            if(time > 5.5f)
                t[i].xoff = 50.0f + t[i].x/((float)yscr) * 30.0f;
            else
                t[i].xoff = 9000.0f;
        } else {
            t[i].xoff = 50.0f + t[i].x/((float)yscr) * 25.0f;
        }
        t[i].scale = 2.0f;
    }
    for(int i = 0; i < len; i++) {
        GLuint tex;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
        float xsz = ((t[i].x)/(float)xscr)*100;
        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
        glEnd();
        glDeleteTextures(1, &tex);
        glDisable(GL_TEXTURE_2D);
    }
}

void greetsf(float time, uint8_t *fb) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float px = 19.0f/400.0f * 100.0f;
    const int len = 7;
    const char *text[len] = { "greetings to:", "i don't know anybody", "i guess #scenelounge", "truck", "that one guy who got this to be actually shown", "nonsceners:", "neuralspaz"};
    struct Texture t[len];
    for(int i = 0; i < len; i++) {
        t[i].data = drawText(text[i], vec_dup(0.99f));
        t[i].yoff = 100 -(px*(i+1)) + sin(time+i);
        t[i].x = 9*strlen(text[i]);
        t[i].y = 15;
        t[i].xoff = 50.0f;
        t[i].scale = 1.0f;
    }
    for(int i = 0; i < len; i++) {
        GLuint tex;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
        float xsz = ((t[i].x)/(float)xscr)*100;
        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
        glEnd();
        glDeleteTextures(1, &tex);
        glDisable(GL_TEXTURE_2D);
    }
}
objl::Loader loader;
struct SceneAOS chess;

int frame = 0;
int fps = 0;
GLFWwindow *win;
float horizontal = 0.0f, vertical = 0.0f;

struct vec3 blueish(float u, float v, float t) {
    struct vec3 blue;
    blue.x = 0.01f;
    blue.y = 0.08f;
    blue.z = 0.433;
    return blue;
}

struct Camera camera;

void chessf(float time, uint8_t *fb) {

    //    glEnable(GL_BLEND);
    //    glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
    //    float want = 1.0f;
    //    const char *makeitend;
    //    if(time < 5.0f) {
    //        makeitend = "is this better?";
    //    } else {
    //        makeitend = "is it ever enough for you?";
    //    }
    //    const char *stop;
    //    if(time < 10.0f) {
    //        stop = "this is what you wanted, right?";
    //    } else {
    //        stop = "you hate me, don't you?";
    //    }
    //    unsigned char *name = drawText(stop, vec_dup(want));
    //    unsigned char *raytracing = drawText(makeitend, vec_dup(1.0f));
    //    const int amt = 150;
    //    struct Texture t[amt];
    //    t[0].data = name;
    //    t[0].x = (9 * strlen(stop));
    //    t[0].y = 15;
    //    t[0].scale = 1.0f;
    //    t[0].yoff = 98.0f;
    //    t[0].xoff = 50.0f;
    //    for(int i = 1; i < 50 + clip(time - 12.0f, 0.0f, 3.0f)*33; i++) {
    //        t[i].data = raytracing;
    //        t[i].x = 9*strlen(makeitend);
    //        t[i].y = 15;
    //        t[i].scale = 1.0f + ((next() % 9)-4)/8.0f;
    //        t[i].xoff = next() % 100;
    //        t[i].yoff = next() % 100;
    //    }
    //    for(int i = 0; i < 50 + clip(time - 12.0f, 0.0f, 3.0f)*33; i++) {
    //        GLuint tex;
    //        glEnable(GL_TEXTURE_2D);
    //        glGenTextures(1, &tex);
    //        glBindTexture(GL_TEXTURE_2D, tex);
    //        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
    //        glBindTexture(GL_TEXTURE_2D, tex);
    //        glBegin(GL_QUADS);
    //        float xsz = ((t[i].x)/(float)xscr)*100;
    //        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
    //        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
    //        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
    //        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
    //        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
    //        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
    //        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
    //        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
    //        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
    //        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
    //        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
    //        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
    //        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
    //        glEnd();
    //        glDeleteTextures(1, &tex);
    //        glDisable(GL_TEXTURE_2D);
    //    }
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //camera;
    struct vec3 a;
    a = vec_mul(vec_dup(-1.0f), camera.center);
    camera.lookat = vec_norm(a);
    camera.up.x = 0.0f;
    camera.up.y = 1.0f;
    camera.up.z = 0.0f;
    struct vec3 right = vec_cross(camera.up, camera.lookat);
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
    camera.lookat.x = cos(vertical) * sin(horizontal);
    camera.lookat.y = sin(vertical);
    camera.lookat.z = cos(horizontal) * cos(vertical);
    right.x = sin(horizontal - 3.14f / 2.0f);
    right.y = 0.0f;
    right.z = cos(horizontal - 3.14f / 2.0f);
    camera.up = vec_cross(right, camera.lookat);
    right = vec_mul(right, vec_dup(-1.0f));
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
    struct AABB aabb = AABBFromSceneAOS(&chess);
#pragma omp parallel for
    for(size_t y = 0; y < yres; y++) {
        struct SceneAOS sc = copySceneAOS(chess);
        struct SceneIndirect si = genIndirectAOS(chess, xres);
        float yf = (float)y/(float)yres - 0.5;
        struct Ray r[xres];
        for(size_t x = 0; x < xres; x++) {
            float xf = (float)x/(float)xres - 0.5;
            struct vec3 rightm = vec_mul(right, vec_dup(xf));
            struct vec3 upm = vec_mul(vec_mul(camera.up, vec_dup(0.625f)), vec_dup(yf));
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
        p.sphereEnd = chess.numspheres;
        p.sphereStart = 0;
        p.triEnd = chess.numtris;
        p.triStart = 0;
        //NRTIndirectAOS(r, &sceneaos, &p, &si);
        //NewDACRTIndirect(p, r, scene, camera, bestAxis(camera, p), si);
        DACRTWorkingNoEarlyTermAOSIndirect2(&p, r, &sc, &camera, bestAxis(camera, p), &si);
        //DACRTWorkingNoEarlyTermAOS(&p, r, &sc, &camera, 0);
        //DACRTWorkingNoEarlyTermIndirectAOS(&p, r, &sceneaos, &camera, 0, &si);
        //DACRTWorkingNoEarlyTerm(p, r, sc, camera, bestAxis(camera, p));
        //printf("%d\n", dnum);
        //printf("%d\n", inum);
        //exit(-1);
        //NRT(r, &scene, p);
#pragma omp simd
        for(size_t x = 0; x < xres; x++) {
            if(vec_dot(r[x].normal, vec_mul(vec_dup(-1.0f), r[x].direction)) < 0) {
                r[x].normal = vec_mul(vec_dup(-1.0f),r[x].normal);
            }
            float fact = (r[x].t == INFINITY) ? 0.0f : vec_dot(r[x].direction, vec_mul(vec_dup(-1.0f), r[x].normal));
            struct vec3 color;
            if(fact > 0.0f) {
                color = vec_mul(vec_dup(fact), r[x].m.eval(r[x].u, r[x].v, r[x].t));
            } else {
                color.x = 0.0f;
                color.y = 0.0f;
                color.z = 0.0f;
            }
            //color[0] = RT(&r, &scene, 0);
            //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
            fb[y*3*xres + r[x].id*3] = fastPow(color.x, 1 / 2.2f)*255;
            fb[y*3*xres + r[x].id*3 + 1] = fastPow(color.y, 1 / 2.2f)*255;
            fb[y*3*xres + r[x].id*3 + 2] = fastPow(color.z, 1 / 2.2f)*255;
        }
        //destroyIndirect(si);
        //deallocScene(sc);
    }
    glRasterPos2i(0, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glPixelZoom((float)xscr/xres, (float)yscr/yres);
    glDrawPixels(xres, yres, GL_RGB, GL_UNSIGNED_BYTE, fb);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    float flash = clip(time-19.8f, 0.0f, 0.2f)*5.0f;
    if(flash != 0.0f && flash < 1.0f) {
        uint8_t fbs[4];
        fbs[0] = 0;
        fbs[1] = 0;
        fbs[2] = 0;
        fbs[3] = flash*255;
        glPixelZoom(xscr, yscr);
        glDrawPixels(1,1, GL_RGBA, GL_UNSIGNED_BYTE, fbs);
    }
    const char *str = vec_sprint(right);
    glRasterPos2i(1, 1);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free(str);
    str = vec_sprint(camera.lookat);
    glRasterPos2i(1, 3);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free(str);
    glRasterPos2i(1, 5);
    str = vec_sprint(camera.up);
    glRasterPos2i(1, 7);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free(str);
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
}

int main(int argc, char* argv[])
{
    //std::thread tm(thread);
    camera.center.x = 0.4f;
    camera.center.y = 1.0f;
    camera.center.z = 0.0f;
    loader.LoadFile("pawn.obj");
    std::stack<glm::vec3> vertices;
    std::vector<unsigned int> &v = loader.LoadedMeshes[0].Indices;
    objl::Vertex v0 = loader.LoadedMeshes[0].Vertices[v[0]];
    glm::vec3 minimum = glm::vec3(v0.Position.X/6.0f, v0.Position.Y/6.0f, v0.Position.Z/3.6f);
    glm::vec3 maximum = minimum;
    for(size_t m = 0; m < loader.LoadedMeshes.size(); m++) {
        for(size_t i = 0; i < loader.LoadedMeshes[m].Indices.size(); i++) {
            v = loader.LoadedMeshes[m].Indices;
            objl::Vertex a = loader.LoadedMeshes[m].Vertices[v[i]];
            minimum.x = std::min(a.Position.Y/6.0f, minimum.x);
            minimum.y = std::min(a.Position.Z/3.6f, minimum.y);
            minimum.z = std::min(a.Position.X/6.0f, minimum.z);
            maximum.x = std::max(a.Position.Y/3.6f, maximum.x);
            maximum.y = std::max(a.Position.Z/6.0f, maximum.y);
            maximum.z = std::max(a.Position.X/6.0f, maximum.z);

            vertices.push(glm::vec3(a.Position.Y/6.0f, a.Position.Z/4.0f, a.Position.X/6.0f));
        }
    }
    glm::vec3 origin = (maximum + minimum)/2.0f;
    std::vector<StorageTriangle> help;
    while(!vertices.empty()) {
        glm::vec3 v1 = vertices.top(); vertices.pop();
        glm::vec3 v2 = vertices.top(); vertices.pop();
        glm::vec3 v3 = vertices.top(); vertices.pop();
        glm::vec3 vec[3];
        vec[0] = v1 - origin;
        vec[1] = v2 - origin;
        vec[2] = v3 - origin;
        Material mat;
        mat.diffuse = 1.0f;
        mat.eval = blueish;
        mat.reflect = 0.0f;
        struct StorageTriangle tris;
        tris.pts[0].x = vec[0].x;
        tris.pts[1].x = vec[1].x;
        tris.pts[2].x = vec[2].x;
        tris.pts[0].y = vec[0].y + 2.2f;
        tris.pts[1].y = vec[1].y + 2.2f;
        tris.pts[2].y = vec[2].y + 2.2f;
        tris.pts[0].z = vec[0].z;
        tris.pts[1].z = vec[1].z;
        tris.pts[2].z = vec[2].z;
        struct StorageTriangle tris2;
        tris2.pts[0].x = vec[0].x - 2.5f;
        tris2.pts[1].x = vec[1].x - 2.5f;
        tris2.pts[2].x = vec[2].x - 2.5f;
        tris2.pts[0].y = vec[0].y + 2.2f;
        tris2.pts[1].y = vec[1].y + 2.2f;
        tris2.pts[2].y = vec[2].y + 2.2f;
        tris2.pts[0].z = vec[0].z + 5.0f;
        tris2.pts[1].z = vec[1].z + 5.0f;
        tris2.pts[2].z = vec[2].z + 5.0f;
        struct StorageTriangle tris3;
        tris3.pts[0].x = vec[0].x - 3.0f;
        tris3.pts[1].x = vec[1].x - 3.0f;
        tris3.pts[2].x = vec[2].x - 3.0f;
        tris3.pts[0].y = vec[0].y + 2.2f;
        tris3.pts[1].y = vec[1].y + 2.2f;
        tris3.pts[2].y = vec[2].y + 2.2f;
        tris3.pts[0].z = vec[0].z;
        tris3.pts[1].z = vec[1].z;
        tris3.pts[2].z = vec[2].z;
        tris.mat = mat;
        tris.mat.eval = red;
        tris2.mat = mat;
        tris3.mat = mat;
        help.push_back(tris);
        help.push_back(tris2);
        help.push_back(tris3);
    }
    vector vec;
    struct Effect starter;
    starter.length = 25.4f;
    starter.func = starterf;
    struct Effect starfield3d;
    starfield3d.length = 35.0f;
    starfield3d.func = starfield3df;
    struct Effect morphosphere;
    morphosphere.length = 5.0f;
    struct  Effect inthespotlight;
    inthespotlight.length = 20.0f;
    inthespotlight.func = spotlightf;
    struct Effect chessgame;
    chessgame.length = 1000.0f;
    chessgame.func = chessf;
    struct Effect plasma;
    plasma.length = 5.0f;
    plasma.func = plasmaf;
    struct Effect greets;
    greets.length = 10.0f;
    greets.func = greetsf;
    struct Effect credits;
    credits.length = 6.5f;
    credits.func = creditsf;
    vector_init(&vec);
    //vector_add(&vec, &starter);
    //vector_add(&vec, &starfield3d);
    //vector_add(&vec, &morphosphere);
    //vector_add(&vec, &inthespotlight);
    //vector_add(&vec, &fallapart);
    vector_add(&vec, &chessgame);
    //vector_add(&vec, &plasma);
    //vector_add(&vec, &greets);
    //vector_add(&vec, &credits);
    for(int i = 0 ; i < 1000; i++) {
        field[i].location.x = next() % 1000;
        field[i].location.y = next() % 1000;
        field[i].location.z = next() % 1000;
    }
    for(int x = 0 ; x < 10; x++) {
        for(int y = 0; y < 10; y++) {
            for(int z = 0; z < 10; z++) {
                newfield[x*100 + y*10 + z].location.x = x*10;
                newfield[x*100 + y*10 + z].location.y = y*10;
                newfield[x*100 + y*10 + z].location.z = -z*10;
            }
        }
    }
    glfwInit();
    win = glfwCreateWindow(xscr, yscr, "hi", glfwGetPrimaryMonitor(), NULL);
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
    s.radius   = 0.2f;
    s.origin.z = 0.0f;
    s.mat.diffuse = 1.0f;
    s.mat.reflect = 0.0f;
    s.mat.eval = red;
    t[0].pts[0].x = -4.0f;
    t[0].pts[0].z = -4.0f;
    t[0].pts[0].y =  0.0f;
    t[0].pts[1].x = 4.0f;
    t[0].pts[1].z = 4.0f;
    t[0].pts[1].y =  0.0f;
    t[0].pts[2].x = 4.0f;
    t[0].pts[2].z = -4.0f;
    t[0].pts[2].y =  0.0f;
    t[0].mat.diffuse = 1.0f;
    t[0].mat.reflect = 0.0f;
    t[0].mat.eval = checker;
    t[1].pts[0].x = 4.0f;
    t[1].pts[0].z = -4.0f;
    t[1].pts[0].y =  0.0f;
    t[1].pts[1].x = -4.0f;
    t[1].pts[1].z = 4.0f;
    t[1].pts[1].y =  0.0f;
    t[1].pts[2].x = 4.0f;
    t[1].pts[2].z = 4.0f;
    t[1].pts[2].y =  0.0f;
    t[1].mat.diffuse = 1.0f;
    t[1].mat.reflect = 0.0f;
    t[1].mat.eval = checker;
    help.push_back(t[0]);
    help.back().pts[0].x = -8.0f;
    help.back().pts[0].z = -8.0f;
    help.back().pts[1].x = -8.0f;
    help.back().pts[1].z = 8.0f;
    help.back().pts[2].x = 8.0f;
    help.back().pts[2].z = -8.0f;
    help.push_back(t[1]);
    help.back().pts[0].x = 8.0f;
    help.back().pts[0].z = 8.0f;
    help.back().pts[1].x = 8.0f;
    help.back().pts[1].z = -8.0f;
    help.back().pts[2].x = -8.0f;
    help.back().pts[2].z = 8.0f;
    chess = generateSceneGraphFromStorageAOS(help.data(), NULL, help.size(), 0);
    unsigned char *fb = malloc(xscr*yscr*3);
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    float horizontal = 1.5f;
    float vertical = 0.0f;
    glfwSetCursorPos(win, xres/2, yres/2);
    glfwSetInputMode(win , GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
    //omp_set_num_threads(4);
    float time = glfwGetTime();
    float start = time;
    //scene = generateSceneGraphFromStorage(t, &s, 2, 1);
    sceneaos = generateSceneGraphFromStorageAOS(t, &s, 2, 1);
    //tm.detach();
    effectsforpattern[1][0] = 1;
    effectsforpattern[1][12] = 1;
    effectsforpattern[1][20] = 1;
    effectsforpattern[1][32] = 1;
    effectsforpattern[1][44] = 1;
    effectsforpattern[5][0] = 1;
    effectsforpattern[5][12] = 1;
    effectsforpattern[5][20] = 1;
    effectsforpattern[5][32] = 1;
    effectsforpattern[5][44] = 1;
    while(!glfwWindowShouldClose(win)) {
        float amt = fall/2.0f;
        fall -= amt;
        cmul -= amt;
            if(fall < 0.05) {
                fall = 0.0f;
            }
            if(cmul < 1.0f && fall < 0.05f) {
                cmul += (1.0f-cmul)/2;
            }
            if(cmul > 0.97f && fall < 0.05f) {
                cmul = 1.0f;
            }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if(!vec.total) {
            printf("greetings to truck and VIKING LINE :^)\n");
            printf("bye! :> \n ");
            return EXIT_SUCCESS;
        }
        float tn = glfwGetTime();
        for(int i = 0; i < 1000; i++) {
            field[i].location.x -= tn-time;
        }
        time = tn;
        struct Effect *efx = vector_get(&vec, 0);
        if(time-start > efx->length) {
            vector_delete(&vec, 0);
            start = time;
            goto skip;
        }
        efx->func(time-start, fb);
skip:
        //        //memset(fb, 0, xres*yres*3);
        //        //struct Scene scene = generateSceneGraphFromStorage(t, &s, 192*4, 0);
        //        struct SceneAOS sceneaos = generateSceneGraphFromStorageAOS(t, &s, 384, 0);
        //        struct AABB aabb = AABBFromSceneAOS(&sceneaos);
        //        struct DACRTPartition part;
        //        part.bounds = aabb;
        //        float t = glfwGetTime();
        //        float x = 5*sin(t);
        //        float z = 10*cos(t);
        //        s.origin.x = x;
        //        s.origin.z = z;
        if(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            return -1;
        }

        //        int threads = omp_get_num_threads();
        //        struct SceneAOS copies[threads];
        //        for(int i = 0; i < threads; i++) {
        //            copies[i] = copySceneAOS(sceneaos);
        //        }

        int error = glGetError();
        if(error != GL_NO_ERROR) {
            printf("%d\n", error);
        }
        glfwSwapBuffers(win);
        glfwPollEvents();
        //deallocScene(scene);
    }
    return 0;
}
