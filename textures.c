#include "veclib.h"
#include "textures.h"

struct vec3 red(float u, float v, float t) {
    struct vec3 r;
    r.x = 1.0f;
    r.y = 1.0f;
    r.z = 1.0f;
    return r;
}

struct vec3 checker(float u, float v, float t) {
    u = u;
    v = v;
    struct vec3 black;
    black.x = 0.10363409665f;
    black.y = 0.6253447208f;
    black.z = 0.9573695762f;
    struct vec3 white;
    white.x = 0.91575012927f;
    white.y = 0.40454082256f;
    white.z = 0.48776520187f;
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
