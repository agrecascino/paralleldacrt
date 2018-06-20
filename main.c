#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>


struct vec3 {
    float x, y, z;
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

struct vec3 vec_norm(struct vec3 a) {
    struct vec3 res;
    float length = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    res.x = a.x/length;
    res.y = a.y/length;
    res.z = a.z/length;
    return res;
}

float vec_dot(struct vec3 a, struct vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void vec_print(struct vec3 a) {
    printf("Values of vec3 \"a\": (%f, %f, %f)\n", a.x, a.y, a.z);
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
    const float b = vec_dot(o, r->direction);
    const float c = vec_dot(o, o) - s->radius * s->radius;
    const float disc = b*b -c;
    if(disc < 0.0f)
        return 0;
    const float tc = b - sqrtf(disc);
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
    const struct vec3 s = vec_sub(r->origin, tri->v);
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

enum DivisionAxis {
    X,
    Y,
    Z
};

enum DivisionAxis bestAxis(struct Camera cam) {

}

void DACRTNonPacketParallel(struct Camera cam, struct DACRTPartition part, struct Scene *scene, struct Ray *rays, size_t nthreads) {
    enum DivisionAxis axis = bestAxis(cam);
}

struct vec3 RT(struct Ray *r, struct Scene *scene) {
    struct vec3 color;
    struct vec3 normal;
    float t = 65537.0f;
    for(size_t i = 0; i < scene->numspheres; i++) {
        intersectSphere(&scene->spheres[i], r, &t, &normal);
    }
    for(size_t i = 0; i < scene->numtris; i++) {
        //intersectTriangle(&scene->tris[i], r, &t, &normal);
    }
    if(t > 65536.0f) {
        return vec_dup(0.2f);
    }
    //color = vec_dup(1.0f);
    color = vec_mul(vec_dup(1.0f), vec_dup(vec_dot(normal, vec_mul(vec_dup(-1.0f), r->direction))));
    return color;
}
int main()
{
    glfwInit();
    GLFWwindow *win = glfwCreateWindow(1024, 768, "hi", NULL, NULL);
    glfwMakeContextCurrent(win);
    glewInit();
    struct Camera camera;
    struct StorageTriangle t[2];
    struct StorageSphere s;
    camera.center.x = 5.0f;
    camera.center.y = 1.0f;
    camera.center.z = 5.0f;
    camera.lookat.x = 0.0f;
    camera.lookat.y = 0.0f;
    camera.lookat.z = 0.0f;
    camera.lookat = vec_norm(vec_sub(camera.lookat, camera.center));
    camera.up.x = 0.0f;
    camera.up.y = 1.0f;
    camera.up.z = 0.0f;
    s.origin.x = 0.0f;
    s.origin.y = 1.0f;
    s.radius   = 1.0f;
    s.origin.z = 0.0f;
    t[0].pts[0].x = -1.0f;
    t[0].pts[0].z = -1.0f;
    t[0].pts[0].y =  0.0f;
    t[0].pts[1].x = 1.0f;
    t[0].pts[1].z = 1.0f;
    t[0].pts[1].y =  0.0f;
    t[0].pts[2].x = 1.0f;
    t[0].pts[2].z = -1.0f;
    t[0].pts[2].y =  0.0f;
    struct Scene scene = generateSceneGraphFromStorage(t, &s, 1, 1);
    struct AABB aabb = AABBFromScene(&scene);
    vec_print(aabb.min);
    vec_print(aabb.max);
    printf("%f\n", vec_dot(aabb.min, aabb.min));
    unsigned char *fb = malloc(1024*768*3);
    size_t xres = 1024;
    size_t yres = 768;
    struct vec3 right = vec_cross(camera.up, camera.lookat);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    float horizontal = 3.14f;
    float vertical = 0.0f;
    glfwSetCursorPos(win, xres/2, yres/2);
    glfwSetInputMode(win , GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
    while(!glfwWindowShouldClose(win)) {
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
        float speedup = 1.0f;
        if(glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) {
            camera.center = vec_add(camera.center,vec_mul(camera.lookat,vec_dup(speedup)));
        }
        if(glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) {
            camera.center = vec_sub(camera.center,vec_mul(camera.lookat,vec_dup(speedup)));
        }
        if(glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) {
            camera.center = vec_add(camera.center,vec_mul(right,vec_dup(speedup)));
        }
        if(glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) {
            camera.center = vec_sub(camera.center,vec_mul(right,vec_dup(speedup)));
        }
        if(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            return -1;
        }
#pragma omp parallel for
        for(size_t x = 0; x < xres; x++) {
            float xf = (float)x/(float)xres - 0.5 * (xres/yres);
            for(size_t y = 0; y < yres; y++) {
                float yf = (float)y/(float)yres - 0.5;
                struct vec3 rightm = vec_mul(right, vec_dup(xf));
                struct vec3 upm = vec_mul(vec_mul(camera.up, vec_dup(-1.0f)), vec_dup(yf));
                struct vec3 direction = vec_norm(vec_add(vec_add(upm, rightm), camera.lookat));
                struct Ray r;
                r.direction = direction;
                r.origin = camera.center;
                r.bounces = 0;
                r.t = -INFINITY;
                struct vec3 color[1];
                color[0] = RT(&r, &scene);
                //newDACRT(color, &r, 1, scene.tris, scene.numtris, scene.spheres, scene.numspheres, aabb);
                fb[y*3*xres + x*3] = color[0].x*255;
                fb[y*3*xres + x*3 + 1] = color[0].y*255;
                fb[y*3*xres + x*3 + 2] = color[0].z*255;
            }
        }
        glDrawPixels(1024, 768, GL_RGB, GL_UNSIGNED_BYTE, fb);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }
    return 0;
}
