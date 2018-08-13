#ifndef RAYTRACER_H
#define RAYTRACER_H
#include "ray_structs.h"

void trace(struct SceneAOS scene, struct Texture *screen, struct Camera cam);

#endif // RAYTRACER_H
