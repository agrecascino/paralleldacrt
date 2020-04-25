#include "veclib.h"
#include "ray_structs.h"
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

extern inline int intersectTriangleAOS(struct Triangle *tri, struct Ray *r) {
    //return 0;
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

extern inline int AABBintersection(struct AABB b, struct Ray *r, float *t) {
    struct Ray raycopy = *r;

    float tx1 = (b.min.x - raycopy.origin.x) * raycopy.inv_dir.x;
    float tx2 = (b.max.x - raycopy.origin.x) * raycopy.inv_dir.x;

    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);

    float ty1 = (b.min.y - raycopy.origin.y) * raycopy.inv_dir.y;
    float ty2 = (b.max.y - raycopy.origin.y) * raycopy.inv_dir.y;

    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (b.min.z - raycopy.origin.z) * raycopy.inv_dir.z;
    float tz2 = (b.max.z - raycopy.origin.z) * raycopy.inv_dir.z;

    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    *t = tmin;

    return tmax >= tmin && (*t > 0);
}

const vectorizedTriangleAOS(struct Ray *r, struct Triangle *tris, int nValid) {
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
