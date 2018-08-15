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
    //NRT(rays, &sceneaos, &p);
    //NRTIndirectAOS(rays, &sceneaos, &p, &si);
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
    return ((next() % 16777217) / 16777216.0f);
}

struct vec3 hemipoint() {
    float r1 = randfloat();
    float r2 = randfloat();
    float st = sqrtf(1 - r1*r1);
    float phi = 2 * 3.14159 * r2;
    float x = st * cosf(phi);
    float z = st * sinf(phi);
    struct vec3 v;
    v.x = x;
    v.z = z;
    v.y =r1;
    return v;
}

static inline struct vec3 vec_make(float x, float y, float z) {
    struct vec3 xyz;
    xyz.x = x;
    xyz.y = y;
    xyz.z = z;
    return xyz;
}

void orient(const struct vec3 n, struct vec3 *nt, struct vec3 *nb) {
    float lxz = 1.0f/sqrtf(n.x * n.x + n.z * n.z);
    float lyz = 1.0f/sqrtf(n.y * n.y + n.z * n.z);
    if(fabs(n.x) > fabs(n.y)) {
        nt->x = n.z * lxz;
        nt->y = 0.0f;
        nt->z = -n.x * lxz;
    } else {
        nt->x = 0.0f;
        nt->y = -n.z * lyz;
        nt->z = n.y * lyz;
    }
    *nb = vec_cross(n, *nt);
}

struct vec3 randomSpherePoint()
{
    float s = randfloat() * 3.1415926 * 2.0;
    float t = randfloat() * 2.0 - 1.0;
    float m = sqrt(1.0 - t * t);
    return vec_make(sin(s) * m, cos(s) * m, t);
}

struct vec3 randomHemispherePoint(struct vec3 dir)
{
    struct vec3 v = randomSpherePoint();
    return vec_mul(v, vec_dup(copysign(1.0,vec_dot(v, dir))));
}

void light(struct SceneAOS sceneaos, struct Ray *rays, size_t nrays, struct AABB aabb, struct SceneIndirect si) {
    struct Ray r[nrays];
    int count = 0;
    for(int s = 0; s < 50; s++) {
        for(int i = 0; i < nrays; i++) {
            struct Ray rl = rays[i];
            if(rl.t != INFINITY) {
                if(vec_dot(rl.normal, vec_mul(vec_dup(-1.0f), rl.direction)) < 0) {
                    rl.normal = vec_mul(vec_dup(-1.0f),rl.normal);
                }
                struct Ray rli;
                rli.origin = vec_add(vec_mul(vec_dup(0.001f), rl.normal), vec_add(rl.origin, vec_mul(rl.direction, vec_dup(rl.t))));
                struct vec3 nt;
                struct vec3 nb;
                orient(rl.normal, &nt, &nb);
                struct vec3 h = hemipoint();
                rli.direction = randomHemispherePoint(rl.normal);
                //            rli.direction = vec_norm(vec_make(
                //                                         h.x * nb.x + h.y * rl.normal.x + h.z * nt.x,
                //                                         h.x * nb.y + h.y * rl.normal.y + h.z * nt.y,
                //                                         h.x * nb.z + h.y * rl.normal.z + h.z * nt.z));
                rli.inv_dir.x = 1.0f/rli.direction.x;
                rli.inv_dir.y = 1.0f/rli.direction.y;
                rli.inv_dir.z = 1.0f/rli.direction.z;
                rli.id = rl.id;
                rli.t = INFINITY;
                rli.m.emit = 0.0f;
                rli.lit = vec_make(0.0f, 0.0f, 0.0f);
                r[count] = rli;
                count++;
            }
        }
        if(!count)
            return;
        for(int i = count; i < nrays; i++) {
            r[i].id = -1;
        }
        traceRays(sceneaos, r, count, aabb, si);
        for(int i = 0; i < nrays; i++) {
            if(r[i].m.emit > 0.0001f && r[i].id != -1) {
                rays[i].lit = vec_add(rays[i].lit, vec_mul(vec_dup(r[i].m.emit * vec_dot(r[i].direction, r[i].normal)), r[i].m.eval(r[i].u, r[i].v, 0)));
            }
        }
    }
    for(int i = 0; i < nrays; i++) {
        rays[i].lit = vec_mul(rays[i].lit, vec_dup(1.0f/50.0f));
    }
}

static float clip(float n, float lower, float upper) {
    return fmax(lower, fmin(n, upper));
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
            r.tree[0][x].lit = vec_make(0.0f, 0.0f, 0.0f);
            r.tree[0][x].m.emit = 0.0f;
            r.tree[0][x].m.reflect = 0.0f;
            r.tree[0][x].m.refract = 0.0f;
        }
        traceRays(sc, r.tree[0], xres, aabb, si);
        light(sc, r.tree[0], xres, aabb, si);
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
                light(sceneaos, r.tree[item], ct, aabb, si);
            }
        }
#pragma omp simd
        for(size_t x = 0; x < xres; x++) {
            if(vec_dot(r.tree[0][x].normal, vec_mul(vec_dup(-1.0f), r.tree[0][x].direction)) < 0) {
                r.tree[0][x].normal = vec_mul(vec_dup(-1.0f),r.tree[0][x].normal);
            }
            int fact = (r.tree[0][x].t != INFINITY);
            struct vec3 color;
            if(fact) {
                //r.tree[0][x].lit = vec_make(1.0f, 1.0f, 1.0f);
                struct vec3 m = r.tree[0][x].m.eval(r.tree[0][x].u, r.tree[0][x].v, r.tree[0][x].t);
                color = vec_mul(r.tree[0][x].lit, m);
                color = vec_add(color, vec_mul(m, vec_dup(r.tree[0][x].m.emit)));
            } else {
                color.x = 0.1f;
                color.y = 0.1f;
                color.z = 0.1f;
            }
            //color[0] = RT(&r, &scene, 0);
            //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
            color.x = clip(color.x, 0, 1);
            color.y = clip(color.y, 0, 1);
            color.z = clip(color.z, 0, 1);
            screen->data[y*3*xres + r.tree[0][x].id*3] = fastPow(color.x, 1 / 2.2f)*255;
            screen->data[y*3*xres + r.tree[0][x].id*3 + 1] = fastPow(color.y, 1 / 2.2f)*255;
            screen->data[y*3*xres + r.tree[0][x].id*3 + 2] = fastPow(color.z, 1 / 2.2f)*255;
        }
        destroyIndirect(si);
        deallocSceneAOS(sc);
    }
}
