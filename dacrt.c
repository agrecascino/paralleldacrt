#include "ray_structs.h"
#include "naive.h"
#include "math.h"
#include "dacrt.h"
#include "intersection_tests.h"

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

uint32_t reduce(uint32_t x, uint32_t N) {
    return ((uint64_t) x * (uint64_t) N) >> 32 ;
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

static inline int AABBinside(struct AABB space, struct vec3 pt) {
    if(pt.x > space.min.x - 0.001f && pt.x < space.max.x + 0.001f) {
        if(pt.y > space.min.y - 0.001f && pt.y < space.max.y + 0.001f) {
            if(pt.z > space.min.z - 0.001f && pt.z < space.max.z + 0.001f) {
                return 1;
            }
        }
    }
    return 0;
    struct vec3 center = vec_mid(space.min, space.max);
    struct vec3 sz;
    sz.x = (space.max.x + 0.001f)- center.x;
    sz.y = (space.max.y + 0.001f)- center.y;
    sz.z = (space.max.z + 0.001f) - center.z;

    if(fabs(center.x - pt.x) < sz.x) {
        if(fabs(center.y - pt.y) < sz.y) {
            if(fabs(center.z - pt.z) < sz.z) {
                return 1;
            }
        }
    }
    return 0;
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
            const int hit = AABBintersection(d2.part[ps].bounds, r + trueitem, &t) || AABBinside(d2.part[ps].bounds, r[trueitem].origin);
            const int terminated = r[trueitem].t < t; /*&& (tother < t) && hitother;*/
            if(hit /*&& !terminated*/) {
                if(i != pivot) {
                    si->rays[i] = si->rays[pivot];
                    si->rays[pivot] = trueitem;
                    //list_swap(r + pivot, r + i, sizeof(struct Ray));
                }
                r[trueitem].bounces++;
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
