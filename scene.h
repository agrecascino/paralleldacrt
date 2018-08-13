#ifndef SCENE_H
#define SCENE_H

struct AABB AABBFromScene(struct Scene *s);

struct AABB AABBFromSceneAOS(struct SceneAOS *s);

struct SceneAOS generateSceneGraphFromStorageAOS(struct StorageTriangle *tris, struct StorageSphere *spheres, size_t numtris, size_t numspheres);

struct Scene generateSceneGraphFromStorage(struct StorageTriangle *tris, struct StorageSphere *spheres, size_t numtris, size_t numspheres);

void deallocScene(struct Scene s);

void deallocSceneAOS(struct SceneAOS s);

struct Scene copyScene(struct Scene s);

struct SceneAOS copySceneAOS(struct SceneAOS s);

struct SceneIndirect genIndirect(struct Scene s, int numrays);

struct SceneIndirect genIndirectAOS(struct SceneAOS s, int numrays);

void destroyIndirect(struct SceneIndirect si);
#endif // SCENE_H
