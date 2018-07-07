#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <math.h>
#include <omp.h>

/*  Written in 2016-2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

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
        float xyz[4];
        struct {
            float x, y, z;
        };
    };
};

struct vec3 vec_add(struct vec3 a, struct vec3 b) {
    struct vec3 res = a;
    res.x += b.x;
    res.y += b.y;
    res.z += b.z;
    return res;
}

struct vec3 vec_mul(struct vec3 a, struct vec3 b) {
    struct vec3 res = a;
    res.x *= b.x;
    res.y *= b.y;
    res.z *= b.z;
    return res;
}

struct vec3 vec_sub(struct vec3 a, struct vec3 b) {
    struct vec3 res = a;
    res.x -= b.x;
    res.y -= b.y;
    res.z -= b.z;
    return res;
}

struct vec3 vec_dup(float f) {
    struct vec3 res;
    res.x = f;
    res.y = f;
    res.z = f;
    return res;
}

struct vec3 vec_cross(struct vec3 a, struct vec3 b) {
    struct vec3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

float vec_length(struct vec3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

struct vec3 vec_norm(struct vec3 a) {
    struct vec3 res;
    float length = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    res.x = a.x/length;
    res.y = a.y/length;
    res.z = a.z/length;
    return res;
}

struct vec3 vec_mid(struct vec3 a, struct vec3 b) {
    struct vec3 res;
    res.x = (a.x + b.x) * 0.5f;
    res.y = (a.y + b.y) * 0.5f;
    res.z = (a.z + b.z) * 0.5f;
    return res;
}

float vec_dot(struct vec3 a, struct vec3 b) {
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
    struct Triangle *tris;
    struct Sphere *spheres;
    size_t numtris;
    size_t numspheres;
};

struct Ray {
    struct vec3 origin;
    struct vec3 direction;
    float t;
    size_t bounces;
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
        struct vec3 pt0 = s->tris[i].pt0;
        struct vec3 pt1 = vec_add(s->tris[i].pt0, s->tris[i].u);
        struct vec3 pt2 = vec_add(s->tris[i].pt0, s->tris[i].v);
        minimum.x = fmin(fmin(fmin(pt0.x, pt1.x), pt2.x), minimum.x);
        minimum.y = fmin(fmin(fmin(pt0.y, pt1.y), pt2.y), minimum.y);
        minimum.z = fmin(fmin(fmin(pt0.z, pt1.z), pt2.z), minimum.z);
        maximum.x = fmax(fmax(fmax(pt0.x, pt1.x), pt2.x), maximum.x);
        maximum.y = fmax(fmax(fmax(pt0.y, pt1.y), pt2.y), maximum.y);
        maximum.z = fmax(fmax(fmax(pt0.z, pt1.z), pt2.z), maximum.z);
    }
    for(size_t i = 0; i < s->numspheres; i++) {
        struct vec3 pt0 = vec_sub(s->spheres[i].origin, vec_dup(s->spheres[i].radius));
        struct vec3 pt1 = vec_add(s->spheres[i].origin, vec_dup(s->spheres[i].radius));
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

int intersectSphere(struct Sphere *s, struct Ray *r, float *t, struct vec3 *normal) {
    const struct vec3 o = vec_sub(s->origin, r->origin);
    const float tca = vec_dot(o, r->direction);
    float d2 = vec_dot(o, o) - tca*tca;
    if(d2 > s->radius*s->radius)
        return 0;
    const float tc = tca - sqrtf(s->radius*s->radius - d2);
    if(tc < 0)
        return 0;
    if(tc > *t)
        return 0;
    *t = tc;
    *normal = vec_norm(vec_sub(vec_add(r->origin, vec_mul(vec_dup(*t), r->direction)), s->origin));
    return 1;
}

int intersectTriangle(struct Triangle *tri, struct Ray *r, float *t, struct vec3 *normal) {
    const float eps = 0.0001f;
    const struct vec3 h = vec_cross(r->direction, tri->v);
    const float a = vec_dot(tri->u, h);
    if(a > -eps && a < eps) {
        return 0;
    }
    const float f = 1.0f / a;
    const struct vec3 s = vec_sub(r->origin, tri->pt0);
    const float u = f * vec_dot(s, h);
    if(u < 0.0f || u > 1.0f) {
        return 0;
    }
    const struct vec3 q = vec_cross(s, tri->u);
    const float v = f * vec_dot(r->direction, q);
    if(v < 0.0f || u+v > 1.0f) {
        return 0;
    }
    const float ts = f * vec_dot(tri->v, q);
    if(ts < eps) {
        return 0;
    }
    if(ts > *t) {
        return 0;
    }
    *t = ts;
    *normal = tri->normal;
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
    }
    for(size_t i = 0; i < scene.numspheres; i++) {
        scene.spheres[i].origin = spheres->origin;
        scene.spheres[i].radius = spheres->radius;
    }
    return scene;
}

void deallocScene(struct Scene s) {
    free(s.spheres);
    free(s.tris);
}

enum DivisionAxis {
    X,
    Y,
    Z
};

int AABBintersection(struct AABB b, struct Ray r, float *t) {
    float tx1 = (b.min.x - r.origin.x) / r.direction.x;
    float tx2 = (b.max.x - r.origin.x) / r.direction.x;

    float tmin = fmin(tx1, tx2);
    float tmax = fmax(tx1, tx2);

    float ty1 = (b.min.y - r.origin.y) / r.direction.y;
    float ty2 = (b.max.y - r.origin.y) / r.direction.y;

    tmin = fmax(tmin, fmin(ty1, ty2));
    tmax = fmin(tmax, fmax(ty1, ty2));

    float tz1 = (b.min.z - r.origin.z) / r.direction.z;
    float tz2 = (b.max.z - r.origin.z) / r.direction.z;

    tmin = fmax(tmin, fmin(tz1, tz2));
    tmax = fmin(tmax, fmax(tz1, tz2));

    *t = tmin;

    return tmax >= tmin;
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
    float t;
    AABBintersection(part.bounds, r, &t);
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
}

struct DuoPartition {
    struct DACRTPartition part[2];
};

struct DuoPartition averageSpaceCut(struct DACRTPartition part, enum DivisionAxis axis) {
    struct DuoPartition duo;
    struct vec3 diff = vec_sub(part.bounds.max, part.bounds.min);
    struct vec3 m1 = part.bounds.min;
    struct vec3 m2;
    m2.x = m1.x + diff.x * (axis == X) ? 0.5f : 1.0f;
    m2.y = m1.y + diff.y * (axis == Y) ? 0.5f : 1.0f;
    m2.z = m1.z + diff.z * (axis == Z) ? 0.5f : 1.0f;
    struct vec3 m3;
    m3.x = m1.x + diff.x * (axis == X) ? 0.5f : 0.0f;
    m3.y = m1.y + diff.y * (axis == Y) ? 0.5f : 0.0f;
    m3.z = m1.z + diff.z * (axis == Z) ? 0.5f : 0.0f;
    struct vec3 m4 = part.bounds.max;
    duo.part[0].bounds.min = m1;
    duo.part[0].bounds.max = m2;
    duo.part[1].bounds.min = m3;
    duo.part[1].bounds.max = m4;
    return duo;
}

void list_swap(restrict void *i1, restrict void *i2, size_t size) {
    unsigned char object[size];
    memcpy(object, i1, size);
    memcpy(i1, i2, size);
    memcpy(i2, object, size);
}

struct PivotPair {
    int firstEnd;
    int secondStart;
};

struct PivotPair findTrianglePivots(struct DuoPartition duo, struct DACRTPartition p, struct Scene *scene, enum DivisionAxis axis) {
    int sharedStart = p.triStart;
    int knownSorted = p.triStart;
    float bboxsplit = duo.part[0].bounds.max.xyz[axis];
    for(int i = p.triStart; i < p.triEnd; i++) {
        struct vec3 pt0 = scene->triangles[i].pt0;
        struct vec3 pt1 = vec_add(pt0, scene->triangles[i].u);
        struct vec3 pt2 = vec_add(pt0, scene->triangles[i].v);
        float v = fmin(pt0.xyz[axis], fmin(pt1.xyz[axis], pt2.xyz[axis]));
	if(v < bboxsplit) {
            list_swap(&scene->triangles[i], &scene->triangles[knownSorted], sizeof(Triangle));
            knownSorted++;
        }
    }
    sharedStart = knownSorted;
    for(int i = p.triStart; i < knownSorted; i++) {
        struct vec3 pt0 = scene->triangles[i].pt0;
        struct vec3 pt1 = vec_add(pt0, scene->triangles[i].u);
        struct vec3 pt2 = vec_add(pt0, scene->triangles[i].v);
        float v = fmax(pt0.xyz[axis], fmax(pt1.xyz[axis], pt2.xyz[axis]));
	if(v > bboxsplit) {
            list_swap(&scene->triangles[i], &scene->triangles[sharedStart], sizeof(Triangle));
            sharedStart--;
        }
    }
    struct PivotPair p;
    p.firstEnd = knownSorted;
    p.secondStart = sharedStart;
    return p;
}

struct DuoPartition subdivideSpace(struct DACRTPartition part, enum DivisionAxis axis, struct Camera cam, struct Scene *scene) {
    struct DuoPartition duo;
    duo = averageSpaceCut(part, axis);
    float rlength = vec_length(vec_sub(vec_mid(duo.part[1].bounds.min, duo.part[1].bounds.max), cam.center));
    float llength = vec_length(vec_sub(vec_mid(duo.part[0].bounds.min, duo.part[0].bounds.max), cam.center));
    if(rlength < llength) {
        struct DACRTPartition p = duo.part[0];
        duo.part[0] = duo.part[1];
        duo.part[1] = p;
    }
    struct PivotPair pT = findTrianglePivots(duo, part, scene, axis);
}

void DACRTNonPacketParallel(struct Camera cam, struct AABB space, struct Scene *scene, struct Ray *rays, size_t nthreads, size_t numrays) {
    struct DACRTPartition part;
    part.bounds = space;
    part.terminatedRay = 0;
    part.rayStart = 0;
    part.triStart = 0;
    part.sphereStart = 0;
    part.triEnd = scene->numtris - 1;
    part.rayEnd = numrays-1;
    part.sphereEnd = scene->numspheres - 1;
    enum DivisionAxis axis = bestAxis(cam, part);

}

struct vec3 RT(struct Ray *r, struct Scene *scene, int noshade) {
    struct vec3 color;
    struct vec3 normal;
    float t = 65537.0f;
    for(size_t i = 0; i < scene->numspheres; i++) {
        intersectSphere(&scene->spheres[i], r, &t, &normal);
    }
    for(size_t i = 0; i < scene->numtris; i++) {
        intersectTriangle(&scene->tris[i], r, &t, &normal);
    }
    if(t > 65536.0f) {
        return vec_dup(0.04f);
    }
    //color = vec_dup(1.0f);
    r->t = t;
    if(noshade)
        return vec_dup(0.0f);
    float accum = 0.0f;
    if(vec_dot(normal, vec_mul(r->direction, vec_dup(-1.0f))) < 0.0f) {
        normal = vec_mul(normal, vec_dup(-1.0f));
    }
    for(int i = 0; i < 1; i++) {
        float x = (((float)(next()&0xFFFF) - 0xFFFF/2.0f)/32767.5f) * 0.8;
        float y = (((float)(next()&0xFFFF) - 0xFFFF/2.0f)/32767.5f) * 0.8;
        float z = (((float)(next()&0xFFFF) - 0xFFFF/2.0f)/32767.5f) * 0.8;
        struct vec3 randvec;
        randvec.x = x;
        randvec.y = y;
        randvec.z = z;
        randvec = randvec;
        struct vec3 dir = normal;
        struct vec3 org = vec_add(vec_add(r->origin, vec_mul(r->direction, vec_dup(t))), vec_mul(normal, vec_dup(0.02f)));
        struct Ray r2;
        dir = vec_norm(vec_add(dir, randvec));
        r2.direction = dir;
        r2.origin = org;
        r2.t = 65537.0f;
        RT(&r2, scene, 1);
        accum += (1/1.0f)*(r2.t/65537.0f);
    }
    struct vec3 lloc;
    lloc.x = 0.0f;
    lloc.z = 0.0f;
    lloc.y = 10.0f;
    struct vec3 ldir = vec_norm(vec_sub(lloc, vec_add(r->origin, vec_mul(r->direction, vec_dup(t)))));
    color = vec_mul(vec_dup(1.0f*accum), vec_dup(fmax((vec_dot(normal, ldir)+1.0f)/2.0f, 0.0f)));
    return color;
}

inline double fastPow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

size_t xres = 1024;
size_t yres = 768;

int main(int argc, char* argv[])
{
    glfwInit();
    GLFWwindow *win = glfwCreateWindow(1024, 768, "hi", NULL, NULL);
    glfwMakeContextCurrent(win);
    glewInit();
    glutInit(&argc, argv);
    struct Camera camera;
    struct StorageTriangle t[2];
    struct StorageSphere s;
    camera.center.x = 25.0f;
    camera.center.y = 1.0f;
    camera.center.z = 25.0f;
    camera.lookat.x = 0.0f;
    camera.lookat.y = 0.0f;
    camera.lookat.z = 0.0f;
    camera.lookat = vec_norm(vec_sub(camera.lookat, camera.center));
    camera.up.x = 0.0f;
    camera.up.y = 1.0f;
    camera.up.z = 0.0f;
    s.origin.x = 0.0f;
    s.origin.y = 1.0f;
    s.radius   = 0.8f;
    s.origin.z = 0.0f;
    t[0].pts[0].x = -20.0f;
    t[0].pts[0].z = -20.0f;
    t[0].pts[0].y =  0.0f;
    t[0].pts[1].x = 20.0f;
    t[0].pts[1].z = 20.0f;
    t[0].pts[1].y =  0.0f;
    t[0].pts[2].x = 20.0f;
    t[0].pts[2].z = -20.0f;
    t[0].pts[2].y =  0.0f;
    t[1].pts[0].x = -20.0f;
    t[1].pts[0].z = -20.0f;
    t[1].pts[0].y =  0.0f;
    t[1].pts[1].x = -20.0f;
    t[1].pts[1].z = 20.0f;
    t[1].pts[1].y =  0.0f;
    t[1].pts[2].x = 20.0f;
    t[1].pts[2].z = 20.0f;
    t[1].pts[2].y =  0.0f;
    unsigned char *fb = malloc(1024*768*3);
    struct vec3 right = vec_cross(camera.up, camera.lookat);

    float horizontal = 0.0f;
    float vertical = 0.0f;
    glfwSetCursorPos(win, xres/2, yres/2);
    glfwSetInputMode(win , GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
    omp_set_num_threads(28);
    while(!glfwWindowShouldClose(win)) {
        struct Scene scene = generateSceneGraphFromStorage(t, &s, 2, 1);
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
        if(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            return -1;
        }
#pragma omp parallel for
        for(size_t x = 0; x < xres; x++) {
            float xf = ((float)x/(float)xres - 0.5) * ((float)xres/yres);
            for(size_t y = 0; y < yres; y++) {
                float yf = (float)y/(float)yres - 0.5;
                struct vec3 rightm = vec_mul(right, vec_dup(xf));
                struct vec3 upm = vec_mul(vec_mul(camera.up, vec_dup(1.0f)), vec_dup(yf));
                struct vec3 direction = vec_norm(vec_add(vec_add(upm, rightm), camera.lookat));
                struct Ray r;
                r.direction = direction;
                r.origin = camera.center;
                r.bounces = 0;
                r.t = -INFINITY;
                struct vec3 color[1];
                color[0] = RT(&r, &scene, 0);
                //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
                fb[y*3*xres + x*3] = fastPow(color[0].x, 1 / 2.2f)*255;
                fb[y*3*xres + x*3 + 1] = fastPow(color[0].y, 1 / 2.2f)*255;
                fb[y*3*xres + x*3 + 2] = fastPow(color[0].z, 1 / 2.2f)*255;
            }
        }
        glMatrixMode(GL_PROJECTION);
        gluOrtho2D(0, 100, 0, 100);
        glRasterPos2i(0, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        glDrawPixels(1024, 768, GL_RGB, GL_UNSIGNED_BYTE, fb);
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
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glfwSwapBuffers(win);
        glfwPollEvents();
        deallocScene(scene);
    }
    return 0;
}
