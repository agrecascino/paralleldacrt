#ifndef INTERSECTION_TESTS_H
#define INTERSECTION_TESTS_H


int intersectSphere(struct Scene *s, int i, struct Ray *r);

int intersectSphereAOS(struct SceneAOS *s, int i, struct Ray *r);

int intersectTriangle(struct Scene *scene, int i, struct Ray *r);

inline int intersectTriangleAOS(struct Triangle *tri, struct Ray *r);

inline int AABBintersection(struct AABB b, struct Ray *r, float *t);

const vectorizedTriangleAOS(struct Ray *r, struct Triangle *tris, int nValid);

struct vec3 boxNormal(struct AABB box, struct Ray ray, float t);
#endif // INTERSECTION_TESTS_H
