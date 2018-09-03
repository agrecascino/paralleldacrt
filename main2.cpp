#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stack>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
extern "C"
{
#include "libfont.h"
#include "veclib.h"
#include <vector.h>
#include "ray_structs.h"
#include "scene.h"
#include "dacrt.h"
#include "textures.h"
#include "raytracer.h"
}
#include "obj.h"
#include <glm/glm.hpp>
#include <glm/gtx/projection.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <libopenmpt/libopenmpt.h>
#include <libopenmpt/libopenmpt_stream_callbacks_file.h>
#include <portaudio.h>
/*  Written in 2016-2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. Th is software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include <stdint.h>

/* This is xoroshiro128+ 1.0, our best and fastest small-state generator
   for floating-point numbers. We suggest to use its upper bits for
   floating-point generation, as it is slightly faster than
   xoroshiro128**. It passes all tests we are aware of except for the four
   lower bits, which might fail linearity tests (and just those), so if
   low linear complexity is not considered an issue (as it is usually the
   case) it can be used to generate 64-bit outputs, too; moreover, this
   generator has a very mild Hamming-weight dependency making our test
   (http://prng.di.unimi.it/hwd.php) fail after 8 TB of output; we believe
   this slight bias cannot affect any application. If you are concerned,
   use xoroshiro128** or xoshiro256+.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s.

   NOTE: the parameters (a=24, b=16, b=37) of this version give slightly
   better results in our test than the 2016 version (a=55, b=14, c=37).
*/

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}


static uint64_t s[2] ={12, 424};

uint64_t next(void) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
    s[1] = rotl(s1, 37); // c

    return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^64 calls to next(); it can be used to generate 2^64
   non-overlapping subsequences for parallel computations. */

void jump(void) {
    static const uint64_t JUMP[] = { 0xdf900294d8f554a5, 0x170865df4b3201fc };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
            }
            next();
        }

    s[0] = s0;
    s[1] = s1;
}


/* This is the long-jump function for the generator. It is equivalent to
   2^96 calls to next(); it can be used to generate 2^32 starting points,
   from each of which jump() will generate 2^32 non-overlapping
   subsequences for parallel distributed computations. */

std::map<int32_t, std::map<int32_t, int32_t>> effectsforpattern;

void long_jump(void) {
    static const uint64_t LONG_JUMP[] = { 0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1 };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
            }
            next();
        }

    s[0] = s0;
    s[1] = s1;
}
float fall  = 0.0f;

void thread() {
    Pa_Initialize();
    int interpol = 1;
    int ss = 100;
    const int BUFFERSIZE = 2000;
    const int SAMPLERATE = 48000;
    static int16_t left[BUFFERSIZE];
    static int16_t right[BUFFERSIZE];
    static int16_t * const buffers[2] = { left, right };
    FILE *file;
    openmpt_module * mod = 0;
    size_t count = 0;
    PaStream * stream = 0;
    Pa_OpenDefaultStream(&stream, 0, 2, paInt16 | paNonInterleaved, SAMPLERATE, paFramesPerBufferUnspecified, NULL, NULL);
    Pa_StartStream(stream);

    file = fopen("song.mptm", "rb");
    if(file != NULL)
    {

        mod = openmpt_module_create(openmpt_stream_get_file_callbacks(), file, NULL, NULL, NULL);
        openmpt_module_set_render_param(mod, OPENMPT_MODULE_RENDER_INTERPOLATIONFILTER_LENGTH, interpol);
        openmpt_module_set_render_param(mod, OPENMPT_MODULE_RENDER_STEREOSEPARATION_PERCENT, ss);
        fclose(file);
        while (1) {

            count = openmpt_module_read_stereo(mod, SAMPLERATE, BUFFERSIZE, left, right);
            if (count == 0) {
                break;
            }
            Pa_WriteStream(stream, buffers, (unsigned long)count);
            if(effectsforpattern[openmpt_module_get_current_pattern(mod)][openmpt_module_get_current_row(mod)]) {
                std::mutex mtx;
                std::lock_guard<std::mutex> g(mtx);
                fall += 1.0f;
            }
            effectsforpattern[openmpt_module_get_current_pattern(mod)][openmpt_module_get_current_pattern(mod)] = 0;
        }

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        openmpt_module_destroy(mod);
    }
}

size_t xres = 320;
size_t yres = 180;

static inline double fastPow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

const GLfloat gv[108] = {
    -1.0f,-1.0f,-1.0f, // triangle 1 : begin
    -1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, // triangle 1 : end
    1.0f, 1.0f,-1.0f, // triangle 2 : begin
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f, // triangle 2 : end
    1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    1.0f,-1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f,-1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f,-1.0f, 1.0f
};

int alignAt(int offset, int alignment) {
    return ((offset + alignment - 1)/alignment) * alignment;
}

void drawtexture(uint8_t *fb, uint16_t xfb, uint16_t yfb, uint16_t offsetx, uint16_t offsety, struct Texture t) {
    for(int y = 0; y < t.y; y++) {
        for(int x = 0; x < t.x; x++) {
            float alpha = t.data[y*t.x*4 + t.x*4 + 3]/255.0f;
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3)] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3)] + t.data[y*t.x*4 + t.x*4]*(1-alpha);
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 1] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 1] + t.data[y*t.x*4 + t.x*4 + 1]*(1-alpha);
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 2] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 2] + t.data[y*t.x*4 + t.x*4 + 2]*(1-alpha);

        }
    }
}
struct Effect {
    float length;
    void (*func)(float, uint8_t*);
};

struct Star {
    struct vec3 location;
    float speed;
};

float clip(float n, float lower, float upper) {
    return fmax(lower, fmin(n, upper));
}

size_t yscr = 1080;
size_t xscr = 1920;


struct Star field[1000];
struct Star newfield[1000];

float cmul = 1.0f;

struct vec3 vec_make(float x, float y, float z) {
    struct vec3 xyz;
    xyz.x = x;
    xyz.y = y;
    xyz.z = z;
    return xyz;
}

void starterf(float time, uint8_t *fb) {
    glViewport(0, 0, 1920, 1080);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glMatrixMode(GL_PROJECTION);
    gluPerspective(45.0f, 16/9.0f, 0.001f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);
    gluLookAt(0.0, 0.0, 0.0, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    for(int i = 0; i < 1000; i++) {
        glColor3f(1.0f, 1.0f, 1.0f);
        glm::mat4 mat;
        mat = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::vec4 pt(field[i].location.x, field[i].location.y, field[i].location.z, 1.0f);
        glm::mat4 mat2;
        mat2 = glm::perspective(45.0f, 16.0f/9.0f, 0.001f, 1000.0f);
        pt = mat * pt;
        pt = mat2 * pt;
        float z = (pt.z/100.0f);
        glColor3f(z, z, z);
        glPointSize(10.0f);
        glBegin(GL_POINTS);
        glVertex3f(field[i].location.x, field[i].location.y, field[i].location.z);
        glEnd();
    }
    glColor3f(1.0f, 1.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float namefade = clip(time - 0.0f, 0.0f, 0.8f);
    float blitfade = clip(time - 6.0f, 0.0f, 1.0f) * cmul;
    float asmfade  = (clip(time - 4.0f, 0.0f, 0.8f)/2.0f);
    float madewithfade = (clip(time - 18.5f, 0.0f, 0.8f)/2.0f);
    unsigned char *name = drawText("name presents", vec_dup(namefade));
    unsigned char *superblit = drawText("MISSING THE DEADLINE", vec_dup(blitfade));
    unsigned char *assembly = drawText("for assembly 18", vec_dup(asmfade));
    unsigned char *madewith = drawText("made with 100% organic recycled demo parts", vec_dup(madewithfade));

    struct Texture t[4];
    t[0].data = name;
    t[0].x = (9 * strlen("name presents"));
    t[0].y = 15;
    t[0].scale = 1.0f;
    t[0].yoff = 98.0f;
    t[0].xoff = 50.0f;
    t[1].data = superblit;
    t[1].x = (9 * strlen("MISSING THE DEADLINE"));
    t[1].y = 15;
    t[1].scale = 8.0f;
    //t[1].scale = 4.0f;
    t[1].xoff = 50.0f + 25.0f;
    t[1].yoff = 75.0f;
    t[2].data = assembly;
    t[2].x = (9 * strlen("for assembly 18"));
    t[2].y = 15;
    t[2].scale = 1.5f;
    t[2].yoff = 2.5f;
    t[2].xoff = 85.0f;
    t[3].data = madewith;
    t[3].x = (9 * strlen("made with 100% organic recycled demo parts"));
    t[3].y = 15;
    t[3].scale = 1.5f;
    t[3].yoff = 65.0f;
    t[3].xoff = 50.0f;
    for(int i = 0; i < 4; i++) {
        GLuint tex;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
        float xsz = ((t[i].x)/(float)xscr)*100;
        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
        glEnd();
        glDeleteTextures(1, &tex);
        glDisable(GL_TEXTURE_2D);
    }
    float flash = clip(time-25.2f, 0.0f, 0.2f)*5.0f;
    if(flash != 0.0f && flash < 1.0f) {
        uint8_t fbs[4];
        fbs[0] = flash*255;
        fbs[1] = flash*255;
        fbs[2] = flash*255;
        fbs[3] = flash*255;
        glPixelZoom(xscr, yscr);
        glDrawPixels(1,1, GL_RGBA, GL_UNSIGNED_BYTE, fbs);
    }
    glDisable(GL_DEPTH_TEST);
}

void plasmaf(float time, uint8_t *fb) {
}

void starfield3df(float time, uint8_t *fb) {
    glViewport(0, 0, xscr, yscr);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, 4/3.0f, 0.1f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glm::mat4 mat;
    gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
    //mat = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    //glLoadMatrixf(&mat[0][0]);
    ///gluLookAt(1.0, 0.0, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    //glScalef(10.0f, 10.0f, 10.0f);
//    for(int i = 0; i < 1000; i++) {
//        glColor3f(1.0f, 1.0f, 1.0f);
//        glm::vec4 pt(newfield[i].location.x, newfield[i].location.y, newfield[i].location.z, 1.0f);
//        glm::mat4 mat2;
//        mat2 = glm::perspective(45.0f, 4.0f/3.0f, 0.001f, 1000.0f);
//        pt = mat * pt;
//        pt = mat2 * pt;
//        float z = pt.z/100.0f;
//        glScalef(1.0f, 1.0f, 1.0f);
//        glColor3f(newfield[i].location.x/100.0f, newfield[i].location.y/100.0f, newfield[i].location.z/100.0f);
//        glPointSize(10.0f);
//        glBegin(GL_POINTS);
//        glVertex3f(newfield[i].location.x, newfield[i].location.y, newfield[i].location.z);
//        glEnd();
//    }
    glPointSize(10.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS);
    glVertex3f(0, 0, 0);
    glEnd();
    glColor3f(1.0f, 1.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

struct Scene scene;
struct SceneAOS sceneaos;
float velocity = 0.0f;
float acceleration = -0.098f;
float lastdiff = 0.0f;
void spotlightf(float time, uint8_t *fb) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
    float diff = time - lastdiff;
    lastdiff = time;
    velocity += acceleration;
    sceneaos.spheres[0].origin.y += velocity*diff;
    if(sceneaos.spheres[0].origin.y < 0.2) {
        velocity = 4.0f;
    }
    float want = clip(time - 5.0f, 0.0f, 1.0f);
    const char *makeitend;
    if(time > 5.0f && time < 10.0f) {
        makeitend = "right?";
    } else if (time > 10.0f) {
        makeitend = "are you saying no?";
    }else {
        makeitend = "raytracing?";
    }
    const char *stop;
    if(time < 10.0f) {
        stop = "this is what you wanted, right?";
    } else {
        stop = "you hate me, don't you?";
    }
    unsigned char *name = drawText(stop, vec_dup(want));
    unsigned char *raytracing = drawText(makeitend, vec_dup(1.0f));
    const int amt = 150;
    struct Texture t[amt];
    t[0].data = name;
    t[0].x = (9 * strlen(stop));
    t[0].y = 15;
    t[0].scale = 1.0f;
    t[0].yoff = 98.0f;
    t[0].xoff = 50.0f;
    for(int i = 1; i < 50 + clip(time - 12.0f, 0.0f, 3.0f)*33; i++) {
        t[i].data = raytracing;
        t[i].x = 9*strlen(makeitend);
        t[i].y = 15;
        t[i].scale = 1.0f + ((next() % 9)-4)/8.0f;
        t[i].xoff = next() % 100;
        t[i].yoff = next() % 100;
    }
    for(int i = 0; i < 50 + clip(time - 12.0f, 0.0f, 3.0f)*33; i++) {
        GLuint tex;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
        float xsz = ((t[i].x)/(float)xscr)*100;
        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
        glEnd();
        glDeleteTextures(1, &tex);
        glDisable(GL_TEXTURE_2D);
    }
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    struct Camera camera;
    camera.center.x = -0.4f-time;
    camera.center.y = 1.0f;
    camera.center.z = 0.0f;
    struct vec3 a;
    a.x = 1.0f;
    a.y = 0.0f;
    a.z = 0.0f;
    camera.lookat = a;
    camera.up.x = 0.0f;
    camera.up.y = 1.0f;
    camera.up.z = 0.0f;
    glRasterPos2i(0, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glPixelZoom((float)xscr/xres, (float)yscr/yres);
    glDrawPixels(xres, yres, GL_RGB, GL_UNSIGNED_BYTE, fb);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    float flash = clip(time-19.8f, 0.0f, 0.2f)*5.0f;
    if(flash != 0.0f && flash < 1.0f) {
        uint8_t fbs[4];
        fbs[0] = 0;
        fbs[1] = 0;
        fbs[2] = 0;
        fbs[3] = flash*255;
        glPixelZoom(xscr, yscr);
        glDrawPixels(1,1, GL_RGBA, GL_UNSIGNED_BYTE, fbs);
    }
    //        char *s = vec_sprint(camera.center);
}

void creditsf(float time, uint8_t *fb) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float px = 19.0f/yscr * 200.0f;
    const int len = 4;
    const char *text[len] = { "code: name", "music: name", "design: name's insecurities", "bye now :>"};
    struct Texture t[len];
    for(int i = 0; i < len; i++) {
        t[i].data = drawText(text[i], vec_dup(0.99f));
        t[i].yoff = 100 -(px*(i+1)) + sin(time+i);
        t[i].x = 9*strlen(text[i]);
        t[i].y = 15;
        if(i == 3) {
            if(time > 5.5f)
                t[i].xoff = 50.0f + t[i].x/((float)yscr) * 30.0f;
            else
                t[i].xoff = 9000.0f;
        } else {
            t[i].xoff = 50.0f + t[i].x/((float)yscr) * 25.0f;
        }
        t[i].scale = 2.0f;
    }
    for(int i = 0; i < len; i++) {
        GLuint tex;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
        float xsz = ((t[i].x)/(float)xscr)*100;
        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
        glEnd();
        glDeleteTextures(1, &tex);
        glDisable(GL_TEXTURE_2D);
    }
}

void greetsf(float time, uint8_t *fb) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float px = 19.0f/400.0f * 100.0f;
    const int len = 7;
    const char *text[len] = { "greetings to:", "i don't know anybody", "i guess #scenelounge", "truck", "that one guy who got this to be actually shown", "nonsceners:", "neuralspaz"};
    struct Texture t[len];
    for(int i = 0; i < len; i++) {
        t[i].data = drawText(text[i], vec_dup(0.99f));
        t[i].yoff = 100 -(px*(i+1)) + sin(time+i);
        t[i].x = 9*strlen(text[i]);
        t[i].y = 15;
        t[i].xoff = 50.0f;
        t[i].scale = 1.0f;
    }
    for(int i = 0; i < len; i++) {
        GLuint tex;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
        float xsz = ((t[i].x)/(float)xscr)*100;
        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
        glEnd();
        glDeleteTextures(1, &tex);
        glDisable(GL_TEXTURE_2D);
    }
}
objl::Loader loader;
struct SceneAOS chess;

int frame = 0;
int fps = 0;
GLFWwindow *win;
float horizontal = 0.0f, vertical = 0.0f;
float otime = 0.0f;

struct Camera camera;
uint8_t *db;
float *fb;
uint16_t *pxc;
void chessf(float time, float *fb) {

    //    glEnable(GL_BLEND);
    //    glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
    //    float want = 1.0f;
    //    const char *makeitend;
    //    if(time < 5.0f) {
    //        makeitend = "is this better?";
    //    } else {
    //        makeitend = "is it ever enough for you?";
    //    }
    //    const char *stop;
    //    if(time < 10.0f) {
    //        stop = "this is what you wanted, right?";
    //    } else {
    //        stop = "you hate me, don't you?";
    //    }
    //    unsigned char *name = drawText(stop, vec_dup(want));
    //    unsigned char *raytracing = drawText(makeitend, vec_dup(1.0f));
    //    const int amt = 150;
    //    struct Texture t[amt];
    //    t[0].data = name;
    //    t[0].x = (9 * strlen(stop));
    //    t[0].y = 15;
    //    t[0].scale = 1.0f;
    //    t[0].yoff = 98.0f;
    //    t[0].xoff = 50.0f;
    //    for(int i = 1; i < 50 + clip(time - 12.0f, 0.0f, 3.0f)*33; i++) {
    //        t[i].data = raytracing;
    //        t[i].x = 9*strlen(makeitend);
    //        t[i].y = 15;
    //        t[i].scale = 1.0f + ((next() % 9)-4)/8.0f;
    //        t[i].xoff = next() % 100;
    //        t[i].yoff = next() % 100;
    //    }
    //    for(int i = 0; i < 50 + clip(time - 12.0f, 0.0f, 3.0f)*33; i++) {
    //        GLuint tex;
    //        glEnable(GL_TEXTURE_2D);
    //        glGenTextures(1, &tex);
    //        glBindTexture(GL_TEXTURE_2D, tex);
    //        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t[i].x, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, t[i].data);
    //        glBindTexture(GL_TEXTURE_2D, tex);
    //        glBegin(GL_QUADS);
    //        float xsz = ((t[i].x)/(float)xscr)*100;
    //        struct vec3 v1 = vec_make(t[i].xoff - xsz/2, (t[i].yoff - (7.5/yscr)*100), 0);
    //        struct vec3 v2 = vec_make(t[i].xoff - xsz/2, (t[i].yoff + (7.5/yscr)*100), 0);
    //        struct vec3 v3 = vec_make((t[i].xoff + xsz/2), (t[i].yoff + (7.5/yscr)*100), 0);
    //        struct vec3 v4 = vec_make((t[i].xoff + xsz/2), (t[i].yoff - (7.5/yscr)*100), 0);
    //        struct vec3 vorigin = vec_make((t[i].xoff + xsz/2), t[i].yoff, 0);
    //        v1 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v1,vorigin)), vorigin);
    //        v2 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v2,vorigin)), vorigin);
    //        v3 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v3,vorigin)), vorigin);
    //        v4 = vec_add(vec_mul(vec_dup(t[i].scale), vec_sub(v4,vorigin)), vorigin);
    //        glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
    //        glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
    //        glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
    //        glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
    //        glEnd();
    //        glDeleteTextures(1, &tex);
    //        glDisable(GL_TEXTURE_2D);
    //    }
    glViewport(0,0, xscr, yscr);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //camera;
    struct vec3 a;
    a = vec_mul(vec_dup(-1.0f), camera.center);
    camera.lookat = vec_norm(a);
    camera.up.x = 0.0f;
    camera.up.y = 1.0f;
    camera.up.z = 0.0f;
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    int w,h;
    glfwGetWindowSize(win, &w, &h);
    double xpos = w/2, ypos = h/2;
    float mspeed = 0.005f;
    glfwGetCursorPos(win, &xpos, &ypos);
    glfwSetCursorPos(win, w/2, h/2);
    horizontal += mspeed * -(w/2- xpos);
    vertical += mspeed * (h/2 - ypos);
    int changed = 0;

    if(-(w/2- xpos) || (h/2 - ypos)) {
        changed = 1;
    }
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
    right = vec_mul(right, vec_dup(-1.0f));
    float speedup = 0.1f;
    if(glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) {
        changed = 1;
        camera.center = vec_add(camera.center,vec_mul(camera.lookat,vec_dup(speedup)));
    }
    if(glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) {
        changed = 1;
        camera.center = vec_sub(camera.center,vec_mul(camera.lookat,vec_dup(speedup)));
    }
    if(glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) {
        changed = 1;
        camera.center = vec_sub(camera.center,vec_mul(right,vec_dup(speedup)));
    }
    if(glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) {
        changed = 1;
        camera.center = vec_add(camera.center,vec_mul(right,vec_dup(speedup)));
    }
    Texture tex;
    tex.data = db;
    tex.y = yres;
    tex.x = xres;
    if(changed) {
        for(size_t x = 0; x < xres; x++) {
            for(size_t y = 0; y < yres; y++) {
                pxc[y * xres + x] = 0;
            }
        }
        memset(fb, 0, xres*yres*3*4);
    }
    trace(chess, &tex, camera);
    for(size_t x = 0; x < xres; x++) {
        for(size_t y = 0; y < yres; y++) {
           fb[y * xres*3 + x*3] = (((fb[y * xres*3 + x*3]) * pxc[y * xres + x]) + (db[y * xres*3 + x*3]/255.0f)) / (pxc[y * xres + x] + 1);
           fb[y * xres*3 + x*3 + 1] = (((fb[y * xres*3 + x*3 + 1]) * pxc[y * xres + x]) + (db[y * xres*3 + x*3 + 1]/255.0f)) / (pxc[y * xres + x] + 1);
           fb[y * xres*3 + x*3 + 2] = (((fb[y * xres*3 + x*3 + 2]) * pxc[y * xres + x]) + (db[y * xres*3 + x*3 + 2]/255.0f)) / (pxc[y * xres + x] + 1);
           pxc[y * xres + x]++;
        }
    }
    glRasterPos2i(0, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelZoom(6.0f, 6.0f);
    //glPixelZoom((float)xscr/xres, (float)yscr/yres);
    glDrawPixels(xres, yres, GL_RGB, GL_FLOAT, fb);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    float flash = clip(time-19.8f, 0.0f, 0.2f)*5.0f;
    if(flash != 0.0f && flash < 1.0f) {
        uint8_t fbs[4];
        fbs[0] = 0;
        fbs[1] = 0;
        fbs[2] = 0;
        fbs[3] = flash*255;
        glPixelZoom(xscr, yscr);
        glDrawPixels(1,1, GL_RGBA, GL_UNSIGNED_BYTE, fbs);
    }
    const char *str = vec_sprint(right);
    glRasterPos2i(1, 1);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free(str);
    str = vec_sprint(camera.lookat);
    glRasterPos2i(1, 3);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free(str);
    glRasterPos2i(1, 5);
    str = vec_sprint(camera.up);
    glRasterPos2i(1, 7);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free(str);
    frame++;
    float ctime = glfwGetTime();
    if(ctime > otime+1) {
        float tdiff = ctime-otime;
        otime = ctime;
        fps = frame/tdiff;
        frame = 0;
    }
    char fp[(int)((ceil(log10(fps+1))+6)*sizeof(char))];
    sprintf(fp, "FPS: %d", fps);
    glRasterPos2i(1, 9);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)fp);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

int main(int argc, char* argv[])
{
    //std::thread tm(thread);
    camera.center.x = 0.4f;
    camera.center.y = 1.0f;
    camera.center.z = 0.0f;
    loader.LoadFile("pawn.obj");
    std::stack<glm::vec3> vertices;
    std::vector<unsigned int> &v = loader.LoadedMeshes[0].Indices;
    objl::Vertex v0 = loader.LoadedMeshes[0].Vertices[v[0]];
    glm::vec3 minimum = glm::vec3(v0.Position.X/6.0f, v0.Position.Y/6.0f, v0.Position.Z/3.6f);
    glm::vec3 maximum = minimum;
    for(size_t m = 0; m < loader.LoadedMeshes.size(); m++) {
        for(size_t i = 0; i < loader.LoadedMeshes[m].Indices.size(); i++) {
            v = loader.LoadedMeshes[m].Indices;
            objl::Vertex a = loader.LoadedMeshes[m].Vertices[v[i]];
            minimum.x = std::min(a.Position.Y/6.0f, minimum.x);
            minimum.y = std::min(a.Position.Z/3.6f, minimum.y);
            minimum.z = std::min(a.Position.X/6.0f, minimum.z);
            maximum.x = std::max(a.Position.Y/3.6f, maximum.x);
            maximum.y = std::max(a.Position.Z/6.0f, maximum.y);
            maximum.z = std::max(a.Position.X/6.0f, maximum.z);

            vertices.push(glm::vec3(a.Position.Y/6.0f, a.Position.Z/4.0f, a.Position.X/6.0f));
        }
    }
    glm::vec3 origin = (maximum + minimum)/2.0f;
    std::vector<StorageTriangle> help;
    while(!vertices.empty()) {
        glm::vec3 v1 = vertices.top(); vertices.pop();
        glm::vec3 v2 = vertices.top(); vertices.pop();
        glm::vec3 v3 = vertices.top(); vertices.pop();
        glm::vec3 vec[3];
        vec[0] = v1 - origin;
        vec[1] = v2 - origin;
        vec[2] = v3 - origin;
        Material mat;
        mat.diffuse = 1.0f;
        mat.eval = blueish;
        mat.reflect = 0.0f;
        mat.emit = 0.0f;
        struct StorageTriangle tris;
        tris.pts[0].x = vec[0].x;
        tris.pts[1].x = vec[1].x;
        tris.pts[2].x = vec[2].x;
        tris.pts[0].y = vec[0].y + 2.2f;
        tris.pts[1].y = vec[1].y + 2.2f;
        tris.pts[2].y = vec[2].y + 2.2f;
        tris.pts[0].z = vec[0].z;
        tris.pts[1].z = vec[1].z;
        tris.pts[2].z = vec[2].z;
        struct StorageTriangle tris2;
        tris2.pts[0].x = vec[0].x - 2.5f;
        tris2.pts[1].x = vec[1].x - 2.5f;
        tris2.pts[2].x = vec[2].x - 2.5f;
        tris2.pts[0].y = vec[0].y + 2.2f;
        tris2.pts[1].y = vec[1].y + 2.2f;
        tris2.pts[2].y = vec[2].y + 2.2f;
        tris2.pts[0].z = vec[0].z + 5.0f;
        tris2.pts[1].z = vec[1].z + 5.0f;
        tris2.pts[2].z = vec[2].z + 5.0f;
        struct StorageTriangle tris3;
        tris3.pts[0].x = vec[0].x - 3.0f;
        tris3.pts[1].x = vec[1].x - 3.0f;
        tris3.pts[2].x = vec[2].x - 3.0f;
        tris3.pts[0].y = vec[0].y + 2.2f;
        tris3.pts[1].y = vec[1].y + 2.2f;
        tris3.pts[2].y = vec[2].y + 2.2f;
        tris3.pts[0].z = vec[0].z;
        tris3.pts[1].z = vec[1].z;
        tris3.pts[2].z = vec[2].z;
        tris.mat = mat;
        tris.mat.eval = red;
        tris.mat.emit = 1000.0f;
        tris2.mat = mat;
        tris3.mat = mat;
        help.push_back(tris);
        help.push_back(tris2);
        help.push_back(tris3);
    }
    vector vec;
    struct Effect starter;
    starter.length = 25.4f;
    starter.func = starterf;
    struct Effect starfield3d;
    starfield3d.length = 35.0f;
    starfield3d.func = starfield3df;
    struct Effect morphosphere;
    morphosphere.length = 5.0f;
    struct  Effect inthespotlight;
    inthespotlight.length = 20.0f;
    inthespotlight.func = spotlightf;
    struct Effect chessgame;
    chessgame.length = 1000.0f;
    chessgame.func = chessf;
    struct Effect plasma;
    plasma.length = 5.0f;
    plasma.func = plasmaf;
    struct Effect greets;
    greets.length = 10.0f;
    greets.func = greetsf;
    struct Effect credits;
    credits.length = 6.5f;
    credits.func = creditsf;
    vector_init(&vec);
    //vector_add(&vec, &starter);
    //vector_add(&vec, &starfield3d);
    //vector_add(&vec, &morphosphere);
    //vector_add(&vec, &inthespotlight);
    //vector_add(&vec, &fallapart);
    vector_add(&vec, &chessgame);
    //vector_add(&vec, &plasma);
    //vector_add(&vec, &greets);
    //vector_add(&vec, &credits);
    for(int i = 0 ; i < 1000; i++) {
        field[i].location.x = next() % 1000;
        field[i].location.y = next() % 1000;
        field[i].location.z = next() % 1000;
    }
    for(int x = 0 ; x < 10; x++) {
        for(int y = 0; y < 10; y++) {
            for(int z = 0; z < 10; z++) {
                newfield[x*100 + y*10 + z].location.x = x*10;
                newfield[x*100 + y*10 + z].location.y = y*10;
                newfield[x*100 + y*10 + z].location.z = -z*10;
            }
        }
    }
    glfwInit();
    win = glfwCreateWindow(xscr, yscr, "hi", glfwGetPrimaryMonitor(), NULL);
    glfwGetWindowSize(win, (int*)&xscr, (int*)&yscr);
    glfwMakeContextCurrent(win);
    glewInit();
    glutInit(&argc, argv);
    struct Camera camera;
    struct StorageTriangle t[192*4];
    for(int i = 0; i < 16; i++) {
        float xoff = 3*i;
        for(int k = 0; k < 12; k++) {
            struct vec3 p1;
            struct vec3 p2;
            struct vec3 p3;
            p1.x = gv[k*9] + xoff;
            p1.y = gv[k*9 + 1];
            p1.z = gv[k*9 + 2];
            p2.x = gv[k*9 + 3] + xoff;
            p2.y = gv[k*9 + 4];
            p2.z = gv[k*9 + 5];
            p3.x = gv[k*9 + 6] + xoff;
            p3.y = gv[k*9 + 7];
            p3.z = gv[k*9 + 8];

            t[i*12 + k].pts[0] = p1;
            t[i*12 + k].pts[1] = p2;
            t[i*12 + k].pts[2] = p3;
        }
    }
    for(int i = 0; i < 16; i++) {
        float xoff = 3*i;
        for(int k = 0; k < 12; k++) {
            struct vec3 p1;
            struct vec3 p2;
            struct vec3 p3;
            p1.x = gv[k*9] + xoff;
            p1.y = gv[k*9 + 1] + 3.0f;
            p1.z = gv[k*9 + 2];
            p2.x = gv[k*9 + 3] + xoff;
            p2.y = gv[k*9 + 4] + 3.0f;
            p2.z = gv[k*9 + 5];
            p3.x = gv[k*9 + 6] + xoff;
            p3.y = gv[k*9 + 7] + 3.0f;
            p3.z = gv[k*9 + 8];

            t[192 + i*12 + k].pts[0] = p1;
            t[192 + i*12 + k].pts[1] = p2;
            t[192 + i*12 + k].pts[2] = p3;
        }
    }
    for(int i = 0; i < 16; i++) {
        float xoff = 3*i;
        for(int k = 0; k < 12; k++) {
            struct vec3 p1;
            struct vec3 p2;
            struct vec3 p3;
            p1.x = gv[k*9] + xoff;
            p1.y = gv[k*9 + 1] + 3.0f;
            p1.z = gv[k*9 + 2] + 3.0f;
            p2.x = gv[k*9 + 3] + xoff;
            p2.y = gv[k*9 + 4] + 3.0f;
            p2.z = gv[k*9 + 5] + 3.0f;
            p3.x = gv[k*9 + 6] + xoff;
            p3.y = gv[k*9 + 7] + 3.0f;
            p3.z = gv[k*9 + 8] + 3.0f;

            t[384 + i*12 + k].pts[0] = p1;
            t[384 + i*12 + k].pts[1] = p2;
            t[384 + i*12 + k].pts[2] = p3;
        }
    }
    for(int i = 0; i < 16; i++) {
        float xoff = 3*i;
        for(int k = 0; k < 12; k++) {
            struct vec3 p1;
            struct vec3 p2;
            struct vec3 p3;
            p1.x = gv[k*9] + xoff;
            p1.y = gv[k*9 + 1];
            p1.z = gv[k*9 + 2] + 3.0f;
            p2.x = gv[k*9 + 3] + xoff;
            p2.y = gv[k*9 + 4];
            p2.z = gv[k*9 + 5] + 3.0f;
            p3.x = gv[k*9 + 6] + xoff;
            p3.y = gv[k*9 + 7];
            p3.z = gv[k*9 + 8] + 3.0f;

            t[576 + i*12 + k].pts[0] = p1;
            t[576 + i*12 + k].pts[1] = p2;
            t[576 + i*12 + k].pts[2] = p3;
        }
    }
    struct StorageSphere s;
    camera.center.x = 47.867912f;
    camera.center.y = -0.693855f;
    camera.center.z = -2.437953f;
    camera.lookat.x = -0.704917f;
    camera.lookat.y = 0.281158f;
    camera.lookat.z = 0.651185f;
    //camera.lookat = vec_norm(vec_sub(camera.lookat, camera.center));
    camera.up.x = 0.206372f;
    camera.up.y = 0.959661f;
    camera.up.z = -0.190946f;
    s.origin.x = 0.0f;
    s.origin.y = 1.0f;
    s.radius   = 0.2f;
    s.origin.z = 0.0f;
    s.mat.diffuse = 1.0f;
    s.mat.reflect = 0.0f;
    s.mat.eval = red;
    s.mat.emit = 1.0f;
    t[0].pts[0].x = -4.0f;
    t[0].pts[0].z = -4.0f;
    t[0].pts[0].y =  0.0f;
    t[0].pts[1].x = 4.0f;
    t[0].pts[1].z = 4.0f;
    t[0].pts[1].y =  0.0f;
    t[0].pts[2].x = 4.0f;
    t[0].pts[2].z = -4.0f;
    t[0].pts[2].y =  0.0f;
    t[0].mat.diffuse = 0.8f;
    t[0].mat.reflect = 0.2f;
    t[0].mat.eval = checker;
    t[0].mat.emit = 0.0f;
    t[1].pts[0].x = 4.0f;
    t[1].pts[0].z = -4.0f;
    t[1].pts[0].y =  0.0f;
    t[1].pts[1].x = -4.0f;
    t[1].pts[1].z = 4.0f;
    t[1].pts[1].y =  0.0f;
    t[1].pts[2].x = 4.0f;
    t[1].pts[2].z = 4.0f;
    t[1].pts[2].y =  0.0f;
    t[1].mat.diffuse = 0.8f;
    t[1].mat.reflect = 0.2f;
    t[1].mat.eval = checker;
    t[1].mat.emit = 0.0f;
    help.push_back(t[0]);
    help.back().pts[0].x = -8.0f;
    help.back().pts[0].z = -8.0f;
    help.back().pts[1].x = -8.0f;
    help.back().pts[1].z = 8.0f;
    help.back().pts[2].x = 8.0f;
    help.back().pts[2].z = -8.0f;
    help.push_back(t[1]);
    help.back().pts[0].x = 8.0f;
    help.back().pts[0].z = 8.0f;
    help.back().pts[1].x = 8.0f;
    help.back().pts[1].z = -8.0f;
    help.back().pts[2].x = -8.0f;
    help.back().pts[2].z = 8.0f;
    chess = generateSceneGraphFromStorageAOS(help.data(), NULL, help.size(), 0);
    unsigned char *fb = calloc(xscr*yscr*3*4, 1);
    db = calloc(xscr*yscr*3, 1);
    pxc = calloc(xscr*yscr*2, 1);
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    float horizontal = 1.5f;
    float vertical = 0.0f;
    glfwSetCursorPos(win, xscr/2, yscr/2);
    glfwSetInputMode(win , GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
    omp_set_num_threads(1);
    float time = glfwGetTime();
    float start = time;
    //scene = generateSceneGraphFromStorage(t, &s, 2, 1);
    sceneaos = generateSceneGraphFromStorageAOS(t, &s, 2, 1);
    //tm.detach();
    effectsforpattern[1][0] = 1;
    effectsforpattern[1][12] = 1;
    effectsforpattern[1][20] = 1;
    effectsforpattern[1][32] = 1;
    effectsforpattern[1][44] = 1;
    effectsforpattern[5][0] = 1;
    effectsforpattern[5][12] = 1;
    effectsforpattern[5][20] = 1;
    effectsforpattern[5][32] = 1;
    effectsforpattern[5][44] = 1;
    while(!glfwWindowShouldClose(win)) {
        float amt = fall/2.0f;
        fall -= amt;
        cmul -= amt;
            if(fall < 0.05) {
                fall = 0.0f;
            }
            if(cmul < 1.0f && fall < 0.05f) {
                cmul += (1.0f-cmul)/2;
            }
            if(cmul > 0.97f && fall < 0.05f) {
                cmul = 1.0f;
            }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if(!vec.total) {
            printf("greetings to truck and VIKING LINE :^)\n");
            printf("bye! :> \n ");
            return EXIT_SUCCESS;
        }
        float tn = glfwGetTime();
        for(int i = 0; i < 1000; i++) {
            field[i].location.x -= tn-time;
        }
        time = tn;
        struct Effect *efx = vector_get(&vec, 0);
        if(time-start > efx->length) {
            vector_delete(&vec, 0);
            start = time;
            goto skip;
        }
        efx->func(time-start, fb);
skip:
        //        //memset(fb, 0, xres*yres*3);
        //        //struct Scene scene = generateSceneGraphFromStorage(t, &s, 192*4, 0);
        //        struct SceneAOS sceneaos = generateSceneGraphFromStorageAOS(t, &s, 384, 0);
        //        struct AABB aabb = AABBFromSceneAOS(&sceneaos);
        //        struct DACRTPartition part;
        //        part.bounds = aabb;
        //        float t = glfwGetTime();
        //        float x = 5*sin(t);
        //        float z = 10*cos(t);
        //        s.origin.x = x;
        //        s.origin.z = z;
        if(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            return -1;
        }

        //        int threads = omp_get_num_threads();
        //        struct SceneAOS copies[threads];
        //        for(int i = 0; i < threads; i++) {
        //            copies[i] = copySceneAOS(sceneaos);
        //        }

        int error = glGetError();
        if(error != GL_NO_ERROR) {
            printf("%d\n", error);
        }
        glfwSwapBuffers(win);
        glfwPollEvents();
        //deallocScene(scene);
    }
    return 0;
}
