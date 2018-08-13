#include "veclib.h"
#include "textures.h"

struct vec3 red(float u, float v, float t) {
    struct vec3 r;
    r.x = 1.0f;
    r.y = 0.0f;
    r.z = 0.0f;
    return r;
}

struct vec3 checker(float u, float v, float t) {
    u = u;
    v = v;
    struct vec3 black;
    black.x = 0.0f;
    black.y = 0.0f;
    black.z = 0.0f;
    struct vec3 white;
    white.x = 1.0f;
    white.y = 1.0f;
    white.z = 1.0f;
    int u8 = floor(u*4);
    int v8 = floor(v*4);
    if(((u8+v8) % 2) == 0) {
        return black;
    }
    return white;
}


struct vec3 blueish(float u, float v, float t) {
    struct vec3 blue;
    blue.x = 0.01f;
    blue.y = 0.08f;
    blue.z = 0.433;
    return blue;
}
