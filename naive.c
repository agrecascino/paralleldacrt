#include "naive.h"
#include "intersection_tests.h"

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
void NRTIndirectAOS(struct Ray *r, struct SceneAOS *scene, struct DACRTPartition *part, struct SceneIndirect *si) {
    for(int rx = part->rayStart; rx < part->rayEnd; rx++) {
        r[si->rays[rx]].bounces++;
        for(size_t i = part->sphereStart; i < part->sphereEnd; i++) {
            intersectSphereAOS(scene, si->spheres[i], r + si->rays[rx]);
        }
        const int granularity = 4;
        const int bitmask = 0x03;
        for(size_t i = part->triStart; i < part->triEnd; i++) {
            intersectTriangleAOS(scene->tris + si->tris[i], r + si->rays[rx]);
//            struct Triangle grabx[granularity];
//            int nValid = 0;
//            for(size_t x = i; (x < part->triEnd) && (x < i + granularity); x++) {
//                grabx[x & bitmask] = *(scene->tris + si->tris[i]);
//                nValid++;
//            }
//            //vectorizedTriangleAOS(r, grabx, nValid);
//            for(int x = 0; x < nValid; x++) {
//                intersectTriangleAOS(&grabx[x], r + si->rays[rx]);
//            }
        }
    }
}
