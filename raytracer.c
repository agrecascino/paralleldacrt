#include "raytracer.h"
#include "dacrt.h"
#include "scene.h"
#include "ray_structs.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#define depthc 3
#define dconst 1 + 2 + 4 + 8
struct RayTree {
    struct Ray *tree[dconst];
    int nvalid[dconst];
};

static inline double fastPow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

static inline void traceRays(struct SceneAOS sceneaos, struct Ray *rays, size_t nrays, struct AABB aabb, struct SceneIndirect si) {
    struct DACRTPartition p;
    p.bounds = aabb;
    p.rayStart = 0;
    p.rayEnd = nrays;
    p.sphereEnd = sceneaos.numspheres;
    p.sphereStart = 0;
    p.triEnd = sceneaos.numtris;
    p.triStart = 0;
    //NRTIndirectAOS(r, &sceneaos, &p, &si);
    //NewDACRTIndirect(p, r, scene, camera, bestAxis(camera, p), si);
    DACRTWorkingNoEarlyTermAOSIndirect2(&p, rays, &sceneaos, NULL, 0, &si);
    //DACRTWorkingNoEarlyTermAOS(&p, r, &sc, &camera, 0);
    //DACRTWorkingNoEarlyTermIndirectAOS(&p, r, &sceneaos, &camera, bestAxis(camera, p), &si);
    //DACRTWorkingNoEarlyTerm(p, r, sc, camera, bestAxis(camera, p));
    //printf("%d\n", dnum);
    //printf("%d\n", inum);
    //exit(-1);
    //NRT(r, &scene, p);
    for(int i = 0; i < nrays; i++) {
        struct Ray rx = rays[i];
        struct Ray rxs = rays[rx.id];
        rays[rx.id] = rx;
        rays[i] = rxs;
    }
}

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

float randfloat() {
    return ((next() % 65537) / 32768.0f) - 1.0f;
}

struct vec3 hemipoint() {
    float x, y, z, d;
    do {
        x = randfloat();
        y = randfloat();
        z = randfloat();
        d = sqrtf(x*x + y*y + z*z);
    } while (d > 1);
    x = x/d;
    y = y/d;
    z = z/d;
    struct vec3 pt;
    pt.x = x;
    pt.y = fabs(y);
    pt.z = z;
    return pt;
}

void light(struct SceneAOS sceneaos, struct Ray *rays, size_t nrays, struct AABB aabb, struct SceneIndirect si) {
    struct Ray r[nrays];
    for(int i = 0; i < nrays; i++) {
        struct Ray rl = rays[i];
        struct Ray rli;
        rli.origin = vec_add(vec_mul(vec_dup(0.001f), rl.normal), rl.origin);
        rli.direction = vec_norm(vec_add(hemipoint(), rl.normal));
        rli.inv_dir.x = 1.0f/rl.direction.x;
        rli.inv_dir.y = 1.0f/rl.direction.y;
        rli.inv_dir.z = 1.0f/rl.direction.z;
        rli.id = rl.id;
        rli.t = INFINITY;
        rli.m.emit = 0.0f;
        r[i] = rli;
    }
    traceRays(sceneaos, r, nrays, aabb, si);
    for(int i = 0; i < nrays; i++) {
        if(r[i].rli)
    }
}

static float clip(float n, float lower, float upper) {
    return fmax(lower, fmin(n, upper));
}

static inline struct vec3 vec_make(float x, float y, float z) {
    struct vec3 xyz;
    xyz.x = x;
    xyz.y = y;
    xyz.z = z;
    return xyz;
}

struct vec3 refract(const struct vec3 *I, const struct vec3 *N, const float ior)
{
    float cosi = clip(-1, 1, vec_dot(*I, *N));
    float etai = 1, etat = ior;
    struct vec3 n = *N;
    if (cosi < 0) { cosi = -cosi; } else { float cpy = etai; etai = etat; etat = cpy; n= vec_mul(vec_dup(-1.0f), n); }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? vec_make(0.0f, 0.0f, 0.0f) : vec_add(vec_mul(vec_dup(eta), *I), vec_mul(vec_dup(eta * cosi - sqrtf(k)), n));
}

void trace(struct SceneAOS sceneaos, struct Texture *screen, struct Camera camera) {
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    struct AABB aabb = AABBFromSceneAOS(&sceneaos);
    size_t xres = screen->x;
    size_t yres = screen->y;
#pragma omp parallel for
    for(size_t y = 0; y < yres; y++) {
        struct SceneAOS sc = copySceneAOS(sceneaos);
        struct SceneIndirect si = genIndirectAOS(sceneaos, xres);
        float yf = (float)y/(float)yres - 0.5;
        struct Ray rays[dconst][xres];
        struct RayTree r;
        memset(r.nvalid, 0, 16*sizeof(int));
        r.tree[0] = rays[0];
        r.nvalid[0] = xres;
        for(size_t x = 0; x < xres; x++) {
            float xf = (float)x/(float)xres - 0.5;
            struct vec3 rightm = vec_mul(right, vec_dup(xf));
            struct vec3 upm = vec_mul(vec_mul(camera.up, vec_dup(0.5625f)), vec_dup(yf));
            struct vec3 direction = vec_norm(vec_add(vec_add(upm, rightm), camera.lookat));
            struct vec3 inv_dir;
            inv_dir.x = 1.0f/direction.x;
            inv_dir.y = 1.0f/direction.y;
            inv_dir.z = 1.0f/direction.z;
            r.tree[0][x].inv_dir = inv_dir;
            r.tree[0][x].direction = direction;
            r.tree[0][x].origin = camera.center;
            r.tree[0][x].bounces = 0;
            r.tree[0][x].t = INFINITY;
            r.tree[0][x].id = x;
            r.tree[0][x].m.reflect = 0.0f;
            r.tree[0][x].m.refract = 0.0f;
        }
        traceRays(sc, r.tree[0], xres, aabb, si);
        for(int d = 0; d < depthc; d++) {
            for(int i = 0; i < (1 << (d+1)); i++) {
                int ct = 0;
                int item = ((1 << d) + (d == 0)) + i;
                int parent = (item-1)/2;
                for(size_t x = 0; x < r.nvalid[item]; x++) {
                    if(r.tree[parent][x].m.reflect > 0.00001f && !(i & 0x01)) {
                        struct Ray rc = r.tree[parent][x];
                        struct Ray ri;
                        ri.id = x;
                        ri.origin = vec_add(vec_mul(vec_dup(0.001f), rc.normal),
                                            vec_add(rc.origin, vec_mul(
                                            vec_dup(rc.t), rc.direction)));
                        ri.direction = vec_sub(rc.direction, vec_mul(vec_mul(vec_dup(2.0f), rc.normal), vec_dup(vec_dot(rc.direction, rc.normal))));
                        ri.t = INFINITY;
                        ri.inv_dir.x = 1.0f/ri.direction.x;
                        ri.inv_dir.y = 1.0f/ri.direction.y;
                        ri.inv_dir.z = 1.0f/ri.direction.z;
                        ri.m.reflect = 0.0f;
                        ri.m.refract = 0.0f;
                        r.tree[item][ct] = ri;
                        ct++;
                    } else if(r.tree[parent][x].m.refract > 0.00001f && (i & 0x01)) {
                        struct Ray rc = r.tree[parent][x];
                        struct Ray ri;
                        ri.id = x;
                        ri.origin = vec_add(vec_mul(vec_dup(0.001f), rc.normal),
                                            vec_add(rc.origin, vec_mul(
                                            vec_dup(rc.t), rc.direction)));
                        ri.direction = refract(&rc.direction, &rc.normal, rc.m.ior);
                        ri.origin = vec_add(ri.origin,vec_mul(vec_dup(0.001f), ri.direction));
                        ri.t = INFINITY;
                        ri.inv_dir.x = 1.0f/ri.direction.x;
                        ri.inv_dir.y = 1.0f/ri.direction.y;
                        ri.inv_dir.z = 1.0f/ri.direction.z;
                        ri.m.reflect = 0.0f;
                        ri.m.refract = 0.0f;
                        r.tree[item][ct] = ri;
                        ct++;
                    }
                }
                r.nvalid[item] = ct;
                traceRays(sceneaos, r.tree[item], ct, aabb, si);
            }
        }
#pragma omp simd
        for(size_t x = 0; x < xres; x++) {
            if(vec_dot(r.tree[0][x].normal, vec_mul(vec_dup(-1.0f), r.tree[0][x].direction)) < 0) {
                r.tree[0][x].normal = vec_mul(vec_dup(-1.0f),r.tree[0][x].normal);
            }
            float fact = (r.tree[0][x].t == INFINITY) ? 0.0f : vec_dot(r.tree[0][x].direction, vec_mul(vec_dup(-1.0f), r.tree[0][x].normal));
            struct vec3 color;
            if(fact > 0.0f) {
                color = vec_mul(vec_dup(fact), r.tree[0][x].m.eval(r.tree[0][x].u, r.tree[0][x].v, r.tree[0][x].t));
            } else {
                color.x = 0.0f;
                color.y = 0.0f;
                color.z = 0.0f;
            }
            //color[0] = RT(&r, &scene, 0);
            //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
            screen->data[y*3*xres + r.tree[0][x].id*3] = fastPow(color.x, 1 / 2.2f)*255;
            screen->data[y*3*xres + r.tree[0][x].id*3 + 1] = fastPow(color.y, 1 / 2.2f)*255;
            screen->data[y*3*xres + r.tree[0][x].id*3 + 2] = fastPow(color.z, 1 / 2.2f)*255;
        }
        destroyIndirect(si);
        deallocSceneAOS(sc);
    }
}
