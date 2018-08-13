#ifndef NAIVE_H
#define NAIVE_H
#include "ray_structs.h"
void NRTIndirect(struct Ray *r, struct Scene *scene, struct DACRTPartition *part, struct SceneIndirect *si) ;

void NRTIndirectAOS(struct Ray *r, struct SceneAOS *scene, struct DACRTPartition *part, struct SceneIndirect *si);

void NRT(struct Ray *r, struct SceneAOS *scene, struct DACRTPartition *part);
#endif // NAIVE_H
