#ifndef DACRT_H
#define DACRT_H

enum DivisionAxis bestAxis(struct Camera cam, struct DACRTPartition part);

const char* axisString(enum DivisionAxis a);

struct DuoPartition averageSpaceCut(struct DACRTPartition part, enum DivisionAxis axis);

void DACRTWorkingNoEarlyTermIndirectAOS(struct DACRTPartition *space, struct Ray *r, struct SceneAOS *s, struct Camera *cam, int depth, struct SceneIndirect *si);

void DACRTWorkingNoEarlyTermAOS(struct DACRTPartition *space, struct Ray *r, struct SceneAOS *s, struct Camera *cam, int depth);

void DACRTWorkingNoEarlyTermAOSIndirect2(struct DACRTPartition *space, struct Ray *r, struct SceneAOS *s, struct Camera *cam, int depth, struct SceneIndirect *si);

void DACRTWorkingNoEarlyTermIndirect(struct DACRTPartition *space, struct Ray *r, struct Scene *s, struct Camera *cam, int depth, struct SceneIndirect *si);

enum DivisionAxis longestAxis(struct AABB a);
#endif // DACRT_H
