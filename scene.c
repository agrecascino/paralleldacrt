#include "ray_structs.h"
#include "scene.h"
#include "math.h"

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

void deallocSceneAOS(struct SceneAOS s) {
    free(s.spheres);
    free(s.tris);
}

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

