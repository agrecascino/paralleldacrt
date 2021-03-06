#include "raytracer.h"
#include "dacrt.h"
#include "scene.h"
#include "ray_structs.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#define depthc 1
#define dconst 1 + 2
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
        if(rays[i].id == -1)
            continue;
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

float halton_pt(int index, int base) {
    float f = 1;
    float r = 0;
    while(index > 0) {
        f = f/base;
        r = r + f * (index % base);
        index = index/base;
    }
    return r;
}

struct vec3 hemipoint() {
    float r1 = halton_pt(next() % 1024, 2);
    float r2 = halton_pt((next() + next()) % 1024, 2);
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
    float sin_theta = sqrtf(randfloat());
    float cos_theta = sqrtf(1.0f-sin_theta*sin_theta);
    float angle = randfloat()*2.0f*3.14159f;
    return vec_make(sin_theta * cosf(angle), sin_theta*sinf(angle), cos_theta);
}

struct vec3 randomHemispherePoint(struct vec3 dir)
{
    struct vec3 v = randomSpherePoint();
    return vec_mul(v, vec_dup(copysignf(1.0,vec_dot(v, dir))));
}

struct SceneIndirect overwrite(struct SceneIndirect si, struct SceneAOS scene, size_t nrays) {
    for(size_t i = 0; i < nrays; i++) {
        si.rays[i] = i;
    }
}
static float clip(float n, float lower, float upper) {
    return fmax(lower, fmin(n, upper));
}

void light(struct SceneAOS sceneaos, struct Ray *rays, size_t nrays, struct AABB aabb, struct SceneIndirect si) {
    size_t samplecount = 2;
    for(int s = 0; s < samplecount; s++) {
        struct Ray r[nrays];
        for(int i = 0; i < nrays; i++) {
            r[i].id = -1;
        }
        int count = 0;
        for(int i = 0; i < nrays; i++) {
            struct Ray *rl = rays + i;
            if(rl->t != INFINITY) {
                struct Ray rli;
                struct vec3 cnormal = rl->normal;
                if(vec_dot(cnormal, vec_mul(vec_dup(-1.0f), rl->direction)) < 0) {
                    cnormal = vec_mul(vec_dup(-1.0f), cnormal);
                }
                rli.id = rl->id;
                rli.origin = vec_add(rl->origin, vec_mul(vec_dup(rl->t), rl->direction));
                rli.origin = vec_add(rli.origin, vec_mul(vec_dup(0.001f), cnormal));
                rli.direction = randomHemispherePoint(cnormal);
                rli.inv_dir.x = 1.0f/rli.direction.x;
                rli.inv_dir.y = 1.0f/rli.direction.y;
                rli.inv_dir.z = 1.0f/rli.direction.z;
                rli.m.emit = 0.0f;
                rli.m.eval = NULL;
                rli.t = INFINITY;
                rli.bounces = 0;
                r[count] = rli;
                count++;
            }
        }
        if(!count)
            return;
        overwrite(si, sceneaos, count);
        for(int i  = 0; i < count; i++) {
            if(si.rays[i] > count - 1) {
                printf("oh no\n");
                exit(-1);
            }
        }
        traceRays(sceneaos, r, count, aabb, si);
        for(int i = 0; i < nrays; i++) {
            if(r[i].id != -1 && r[i].m.emit > 0.001f) {
                //rays[r[i].id].lit = vec_dup(1.0f);
                //                if(!r[i].bounces) {
                //                    rays[r[i].id].lit = vec_dup(0.0f);
                //                    continue;
                //                }
                //rays[r[i].id].lit = vec_dup(r[i].bounces/16.0f);
                rays[r[i].id].lit = vec_add(rays[r[i].id].lit, vec_mul(vec_dup(r[i].m.emit/* * clip(vec_dot(r[i].direction, r[i].normal), 0.0f, 1.0f)*/), r[i].m.eval(r[i].u, r[i].v, 0)));
                //                if(r[i].m.emit > 0.0001f && r[i].id != -1) {
                //                    rays[i].lit = vec_add(rays[i].lit, vec_mul(vec_dup(r[i].m.emit), r[i].m.eval(r[i].u, r[i].v, 0)));
                //                }
            } else if(r[i].id != -1 && r[i].t == INFINITY) {
                struct vec3 skyup;
                skyup.x = 0.0f;
                skyup.y = 1.0f;
                skyup.z = 0.0f;
                struct vec3 skyblue = vec_make(135/255.0f, 206/255.0f, 250.0f/255.0f);
                struct vec3 orange  = vec_make(255/255.0f, 128/255.0f, 0.0f/255.0f);
                float dp = vec_dot(skyup, r[i].direction);
                float topsky = clip((dp + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                float botsky = clip(((-dp) + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                rays[r[i].id].lit.x += 32.5f/255.0f;
                rays[r[i].id].lit.y += 78.2f/255.0f;
                rays[r[i].id].lit.z += 217.0f/255.0f;
                //rays[r[i].id].lit = vec_add(rays[r[i].id].lit, vec_add(vec_mul(vec_dup(topsky), skyblue), vec_mul(vec_dup(botsky), orange)));
            }
        }
    }
    for(int i = 0; i < nrays; i++) {
        rays[i].lit = vec_mul(rays[i].lit, vec_dup(1.0f/(float)samplecount));
    }
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

void trace(struct SceneAOS sceneaos, struct Texture *screen, struct Camera camera)  {
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    struct AABB aabb = AABBFromSceneAOS(&sceneaos);
    size_t xres = screen->x;
    size_t yres = screen->y;
#pragma omp parallel for
    for(size_t y = 0; y < yres; y++) {
        struct SceneAOS sc = copySceneAOS(sceneaos);
        struct SceneIndirect si = genIndirectAOS(sceneaos, xres);
        struct Ray rays[dconst][xres];
        struct RayTree r;
        memset(r.nvalid, 0, dconst*sizeof(int));
        r.tree[0] = rays[0];
        r.nvalid[0] = xres;
        for(size_t x = 0; x < xres; x++) {
            float yf = (float)y/(float)yres - 0.5f + (1.0f*(randfloat()-0.5f))*(1.0f/yres);
            float xf = (float)x/(float)xres - 0.5f + (1.0f*(randfloat()-0.5f))*(1.0f/xres);
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
            r.tree[0][x].refid = -1;
            r.tree[0][x].rflid = -1;
        }
        traceRays(sc, r.tree[0], xres, aabb, si);
        r.nvalid[0] = xres;
        light(sc, r.tree[0], xres, aabb, si);
        for(int d = 0; d < depthc; d++) {
            for(int i = 0; i < (1 << (d+1)); i++) {
                int ct = 0;
                int item = ((1 << d)) + i;
                //printf("%i\n", item);
                int parent = (item-1)/2;
                r.tree[item] = rays[item];
                for(size_t x = 0; x < r.nvalid[parent]; x++) {
                    //printf("Parent tree: %d, Child tree: %d, Count: %d, Parent Node: %d\n", parent, item, ct, x);
                    if((r.tree[parent][x].m.reflect > 0.00001f)  && (item & 0x01)) {
                        struct Ray rc = r.tree[parent][x];
                        struct Ray ri;
                        ri.id = ct;
                        if(vec_dot(rc.normal, vec_mul(vec_dup(-1.0f), rc.direction)) < 0) {
                            rc.normal = vec_mul(vec_dup(-1.0f), rc.normal);
                        }
                        ri.origin = vec_add(vec_mul(vec_dup(0.001f), rc.normal),
                                                                    vec_add(rc.origin, vec_mul(
                                                                    vec_dup(rc.t), rc.direction)));
                        ri.direction = vec_sub(rc.direction, vec_mul(vec_mul(vec_dup(2.0f), rc.normal), vec_dup(vec_dot(rc.direction, rc.normal))));
                        ri.t = INFINITY;
                        ri.inv_dir.x = 1.0f/ri.direction.x;
                        ri.inv_dir.y = 1.0f/ri.direction.y;
                        ri.inv_dir.z = 1.0f/ri.direction.z;
                        ri.bounces = 0;
                        ri.lit = vec_make(0.0f, 0.0f, 0.0f);
                        ri.m.emit = 0.0f;
                        ri.m.reflect = 0.0f;
                        ri.m.refract = 0.0f;
                        ri.refid = -1;
                        ri.rflid = -1;
                        r.tree[item][ct] = ri;
                        r.tree[parent][x].rflid = ct;
                        ct++;
                    }
//                    if(r.tree[parent][x].m.reflect > 0.00001f && ((item % 2) == 1)) {
//                        struct Ray rc = r.tree[parent][x];
//                        struct Ray ri;
//                        ri.id = x;
//                        ri.origin = vec_add(vec_mul(vec_dup(0.001f), rc.normal),
//                                            vec_add(rc.origin, vec_mul(
//                                                        vec_dup(rc.t), rc.direction)));
//                        ri.direction = vec_sub(rc.direction, vec_mul(vec_mul(vec_dup(2.0f), rc.normal), vec_dup(vec_dot(rc.direction, rc.normal))));
//                        ri.t = INFINITY;
//                        ri.inv_dir.x = 1.0f/ri.direction.x;
//                        ri.inv_dir.y = 1.0f/ri.direction.y;
//                        ri.inv_dir.z = 1.0f/ri.direction.z;
//                        ri.m.reflect = 0.0f;
//                        ri.m.refract = 0.0f;
//                        ri.rflid = -1;
//                        ri.refid = -1;
//                        ri.bounces = 0;
//                        r.tree[item][ct] = ri;
//                        r.tree[parent][x].rflid = ct;
//                        //printf("Parent tree: %d, Child tree: %d, Count: %d, Parent Node: %d\n", parent, item, ct, x);
//                        ct++;
//                    } else if(r.tree[parent][x].m.refract > 0.00001f && !(item & 0x01)) {
////                        struct Ray rc = r.tree[parent][x];
////                        struct Ray ri;
////                        ri.id = x;
////                        ri.origin = vec_add(vec_mul(vec_dup(0.001f), rc.normal),
////                                            vec_add(rc.origin, vec_mul(
////                                                        vec_dup(rc.t), rc.direction)));
////                        ri.direction = refract(&rc.direction, &rc.normal, rc.m.ior);
////                        ri.origin = vec_add(ri.origin,vec_mul(vec_dup(0.001f), ri.direction));
////                        ri.t = INFINITY;
////                        ri.inv_dir.x = 1.0f/ri.direction.x;
////                        ri.inv_dir.y = 1.0f/ri.direction.y;
////                        ri.inv_dir.z = 1.0f/ri.direction.z;
////                        ri.m.reflect = 0.0f;
////                        ri.m.refract = 0.0f;
////                        r.tree[item][ct] = ri;
////                        r.tree[parent][x].refid = ct;
////                        ct++;
//                    }
                }
                r.nvalid[item] = ct;
                overwrite(si, sceneaos, ct);
                traceRays(sceneaos, r.tree[item], ct, aabb, si);
                light(sceneaos, r.tree[item], ct, aabb, si);
            }
        }
//#pragma omp simd
        for(size_t x = 0; x < xres; x++) {
            int fact = (r.tree[0][x].t != INFINITY);
            struct vec3 color;
            color.x  = 0.0f;
            color.y = 0.0f;
            color.z = 0.0f;
            if(fact) {
                if(vec_dot(r.tree[0][x].normal, vec_mul(vec_dup(-1.0f), r.tree[0][x].direction)) < 0) {
                    r.tree[0][x].normal = vec_mul(vec_dup(-1.0f),r.tree[0][x].normal);
                }
                if(r.tree[0][x].rflid != -1) {
                    int id = r.tree[0][x].rflid;
                    if(r.tree[1][id].t != INFINITY) {
                        //printf("%d\n",id);
                        //printf("%d\n", r.nvalid[1]);
                        struct vec3 er = r.tree[1][id].m.eval(r.tree[1][id].u, r.tree[1][id].v, r.tree[1][id].t);
                        color = vec_mul(r.tree[1][id].lit, er);
                        color = vec_add(color, vec_mul(er,vec_dup(r.tree[1][id].m.emit)));
                        color = vec_mul(color, vec_dup(r.tree[0][x].m.reflect));
                        //color = vec_mul(er, vec_dup(r.tree[1][x].m.emit));
                    } else {
                        struct vec3 skyup;
                        skyup.x = 0.0f;
                        skyup.y = 1.0f;
                        skyup.z = 0.0f;
                        struct vec3 skyblue = vec_make(135/255.0f, 206/255.0f, 250.0f/255.0f);
                        struct vec3 orange  = vec_make(255/255.0f, 128/255.0f, 0.0f/255.0f);
                        float dp = vec_dot(skyup, r.tree[1][id].direction) + r.tree[1][id].origin.y/32.0f;
                        float topsky = clip((dp + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                        float botsky = clip(((-dp) + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                        color.x = 32.5f/255.0f;
                        color.y = 78.2f/255.0f;
                        color.z = 217.0f/255.0f;
                        //color = vec_add(vec_mul(vec_dup(topsky), skyblue), vec_mul(vec_dup(botsky), orange));
                    }
                }
                //r.tree[0][x].lit = vec_make(1.0f, 1.0f, 1.0f);
                struct vec3 m = vec_mul(r.tree[0][x].m.eval(r.tree[0][x].u, r.tree[0][x].v, r.tree[0][x].t), vec_dup(r.tree[0][x].m.diffuse));
                struct vec3 litc = vec_mul(r.tree[0][x].lit, m);
                color = vec_add(color, vec_add(litc, vec_mul(m, vec_dup(r.tree[0][x].m.emit))));
            } else {
                struct vec3 skyup;
                skyup.x = 0.0f;
                skyup.y = 1.0f;
                skyup.z = 0.0f;
                struct vec3 skyblue = vec_make(135/255.0f, 206/255.0f, 250.0f/255.0f);
                struct vec3 orange  = vec_make(255/255.0f, 128/255.0f, 0.0f/255.0f);
                float dp = vec_dot(skyup, r.tree[0][x].direction) + r.tree[0][x].origin.y/32.0f;
                float topsky = clip((dp + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                float botsky = clip(((-dp) + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                color.x = 32.5f/255.0f;
                color.y = 78.2f/255.0f;
                color.z = 217.0f/255.0f;
                //color = vec_add(vec_mul(vec_dup(topsky), skyblue), vec_mul(vec_dup(botsky), orange));
            }
            //color[0] = RT(&r, &scene, 0);
            //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
            color.x = color.x / (color.x + 1);
            color.y = color.y / (color.y + 1);
            color.z = color.z / (color.z + 1);
            screen->data[y*3*xres + r.tree[0][x].id*3] = (uint8_t)(color.x*255);
            screen->data[y*3*xres + r.tree[0][x].id*3 + 1] = (uint8_t)(color.y*255);
            screen->data[y*3*xres + r.tree[0][x].id*3 + 2] = (uint8_t)(color.z*255);
        }
        destroyIndirect(si);
        deallocSceneAOS(sc);
    }
}


void tracesplit(struct SceneAOS sceneaos, struct Texture *screen, struct Camera camera)  {
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    struct AABB aabb = AABBFromSceneAOS(&sceneaos);
    size_t xres = screen->x;
    size_t yres = screen->y;
//#pragma omp parallel for
    for(size_t y = 0; y < yres; y++) {
        struct SceneAOS sc = copySceneAOS(sceneaos);
        struct SceneIndirect si = genIndirectAOS(sceneaos, xres);
        struct Ray rays[dconst][xres];
        struct RayTree r;
        memset(r.nvalid, 0, dconst*sizeof(int));
        r.tree[0] = rays[0];
        r.nvalid[0] = xres;
        for(size_t x = 0; x < xres; x++) {
            float yf = (float)y/(float)yres - 0.5 + (1.0f*(randfloat()-0.5f))*(1.0f/yres);
            float xf = (float)x/(float)xres - 0.5 + (1.0f*(randfloat()-0.5f))*(1.0f/xres);
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
            r.tree[0][x].refid = -1;
            r.tree[0][x].rflid = -1;
        }
        traceRays(sc, r.tree[0], xres, aabb, si);
        r.nvalid[0] = xres;
        light(sc, r.tree[0], xres, aabb, si);
        for(int d = 0; d < depthc; d++) {
            for(int i = 0; i < (1 << (d+1)); i++) {
                int ct = 0;
                int item = ((1 << d)) + i;
                //printf("%i\n", item);
                int parent = (item-1)/2;
                r.tree[item] = rays[item];
                for(size_t x = 0; x < r.nvalid[parent]; x++) {
                    //printf("Parent tree: %d, Child tree: %d, Count: %d, Parent Node: %d\n", parent, item, ct, x);
                    if((r.tree[parent][x].m.reflect > 0.00001f)  && (item & 0x01)) {
                        struct Ray rc = r.tree[parent][x];
                        struct Ray ri;
                        ri.id = ct;
                        ri.origin = vec_add(vec_mul(vec_dup(0.001f), rc.normal),
                                                                    vec_add(rc.origin, vec_mul(
                                                                    vec_dup(rc.t), rc.direction)));
                        ri.direction = vec_sub(rc.direction, vec_mul(vec_mul(vec_dup(2.0f), rc.normal), vec_dup(vec_dot(rc.direction, rc.normal))));
                        ri.t = INFINITY;
                        ri.inv_dir.x = 1.0f/ri.direction.x;
                        ri.inv_dir.y = 1.0f/ri.direction.y;
                        ri.inv_dir.z = 1.0f/ri.direction.z;
                        ri.bounces = 0;
                        ri.lit = vec_make(0.0f, 0.0f, 0.0f);
                        ri.m.emit = 0.0f;
                        ri.m.reflect = 0.0f;
                        ri.m.refract = 0.0f;
                        ri.refid = -1;
                        ri.rflid = -1;
                        r.tree[item][ct] = ri;
                        r.tree[parent][x].rflid = ct;
                        ct++;
                    }
//                    if(r.tree[parent][x].m.reflect > 0.00001f && ((item % 2) == 1)) {
//                        struct Ray rc = r.tree[parent][x];
//                        struct Ray ri;
//                        ri.id = x;
//                        ri.origin = vec_add(vec_mul(vec_dup(0.001f), rc.normal),
//                                            vec_add(rc.origin, vec_mul(
//                                                        vec_dup(rc.t), rc.direction)));
//                        ri.direction = vec_sub(rc.direction, vec_mul(vec_mul(vec_dup(2.0f), rc.normal), vec_dup(vec_dot(rc.direction, rc.normal))));
//                        ri.t = INFINITY;
//                        ri.inv_dir.x = 1.0f/ri.direction.x;
//                        ri.inv_dir.y = 1.0f/ri.direction.y;
//                        ri.inv_dir.z = 1.0f/ri.direction.z;
//                        ri.m.reflect = 0.0f;
//                        ri.m.refract = 0.0f;
//                        ri.rflid = -1;
//                        ri.refid = -1;
//                        ri.bounces = 0;
//                        r.tree[item][ct] = ri;
//                        r.tree[parent][x].rflid = ct;
//                        //printf("Parent tree: %d, Child tree: %d, Count: %d, Parent Node: %d\n", parent, item, ct, x);
//                        ct++;
//                    } else if(r.tree[parent][x].m.refract > 0.00001f && !(item & 0x01)) {
////                        struct Ray rc = r.tree[parent][x];
////                        struct Ray ri;
////                        ri.id = x;
////                        ri.origin = vec_add(vec_mul(vec_dup(0.001f), rc.normal),
////                                            vec_add(rc.origin, vec_mul(
////                                                        vec_dup(rc.t), rc.direction)));
////                        ri.direction = refract(&rc.direction, &rc.normal, rc.m.ior);
////                        ri.origin = vec_add(ri.origin,vec_mul(vec_dup(0.001f), ri.direction));
////                        ri.t = INFINITY;
////                        ri.inv_dir.x = 1.0f/ri.direction.x;
////                        ri.inv_dir.y = 1.0f/ri.direction.y;
////                        ri.inv_dir.z = 1.0f/ri.direction.z;
////                        ri.m.reflect = 0.0f;
////                        ri.m.refract = 0.0f;
////                        r.tree[item][ct] = ri;
////                        r.tree[parent][x].refid = ct;
////                        ct++;
//                    }
                }
                r.nvalid[item] = ct;
                overwrite(si, sceneaos, ct);
                traceRays(sceneaos, r.tree[item], ct, aabb, si);
                light(sceneaos, r.tree[item], ct, aabb, si);
            }
        }
//#pragma omp simd
        for(size_t x = 0; x < xres; x++) {
            int fact = (r.tree[0][x].t != INFINITY);
            struct vec3 color;
            color.x  = 0.0f;
            color.y = 0.0f;
            color.z = 0.0f;
            if(fact) {
                if(vec_dot(r.tree[0][x].normal, vec_mul(vec_dup(-1.0f), r.tree[0][x].direction)) < 0) {
                    r.tree[0][x].normal = vec_mul(vec_dup(-1.0f),r.tree[0][x].normal);
                }
                if(r.tree[0][x].rflid != -1) {
                    int id = r.tree[0][x].rflid;
                    if(r.tree[1][id].t != INFINITY) {
                        //printf("%d\n",id);
                        //printf("%d\n", r.nvalid[1]);
                        struct vec3 er = vec_mul(r.tree[1][id].m.eval(r.tree[1][id].u, r.tree[1][id].v, r.tree[1][id].t), vec_dup(r.tree[0][x].m.reflect));
                        //color = vec_mul(r.tree[1][id].lit, er);
                        //color = vec_add(color, vec_mul(er,vec_dup(r.tree[1][x].m.emit)));
                        //color = vec_mul(er, vec_dup(r.tree[1][x].m.emit));
                    } else {
                        struct vec3 skyup;
                        skyup.x = 0.0f;
                        skyup.y = 1.0f;
                        skyup.z = 0.0f;
                        struct vec3 skyblue = vec_make(135/255.0f, 206/255.0f, 250.0f/255.0f);
                        struct vec3 orange  = vec_make(255/255.0f, 128/255.0f, 0.0f/255.0f);
                        float dp = vec_dot(skyup, r.tree[1][id].direction) + r.tree[1][id].origin.y/32.0f;
                        float topsky = clip((dp + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                        float botsky = clip(((-dp) + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                        color = vec_add(vec_mul(vec_dup(topsky), skyblue), vec_mul(vec_dup(botsky), orange));
                    }
                }
                //r.tree[0][x].lit = vec_make(1.0f, 1.0f, 1.0f);
                struct vec3 m = vec_mul(r.tree[0][x].m.eval(r.tree[0][x].u, r.tree[0][x].v, r.tree[0][x].t), vec_dup(r.tree[0][x].m.diffuse));
                struct vec3 litc = vec_mul(r.tree[0][x].lit, m);
                color = vec_add(color, vec_add(litc, vec_mul(m, vec_dup(r.tree[0][x].m.emit))));
            } else {
                struct vec3 skyup;
                skyup.x = 0.0f;
                skyup.y = 1.0f;
                skyup.z = 0.0f;
                struct vec3 skyblue = vec_make(135/255.0f, 206/255.0f, 250.0f/255.0f);
                struct vec3 orange  = vec_make(255/255.0f, 128/255.0f, 0.0f/255.0f);
                float dp = vec_dot(skyup, r.tree[0][x].direction) + r.tree[0][x].origin.y/32.0f;
                float topsky = clip((dp + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                float botsky = clip(((-dp) + 0.1f),0.0f, 1.1f )*10.0f/11.0f;
                color = vec_add(vec_mul(vec_dup(topsky), skyblue), vec_mul(vec_dup(botsky), orange));
            }
            //color[0] = RT(&r, &scene, 0);
            //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
            color.x = 10.0f / (color.x + 10.0f);
            color.y = 10.0f / (color.y + 10.0f);
            color.z = 10.0f / (color.z + 10.0f);
            screen->data[y*3*xres + r.tree[0][x].id*3] = fastPow(color.x, 1 / 2.2f) *255;
            screen->data[y*3*xres + r.tree[0][x].id*3 + 1] = fastPow(color.y, 1 / 2.2f)*255;
            screen->data[y*3*xres + r.tree[0][x].id*3 + 2] = fastPow(color.z, 1 / 2.2f)*255;
        }
        destroyIndirect(si);
        deallocSceneAOS(sc);
    }
}
