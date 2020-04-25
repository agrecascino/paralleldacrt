#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <math.h>
//#include <omp.h>
#include <string.h>
#include <stack>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <mutex>
#define GLM_ENABLE_EXPERIMENTAL
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
#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"
#include <glm/glm.hpp>
#include <glm/gtx/projection.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <libopenmpt/libopenmpt.h>
#include <libopenmpt/libopenmpt_stream_callbacks_file.h>
#include <portaudio.h>
#include <condition_variable>
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
std::mutex morp;
std::condition_variable cv;

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

    file = fopen("andalucia.mod", "rb");
    if(file != NULL)
    {

        mod = openmpt_module_create(openmpt_stream_get_file_callbacks(), file, NULL, NULL, NULL);
        openmpt_module_set_render_param(mod, OPENMPT_MODULE_RENDER_INTERPOLATIONFILTER_LENGTH, interpol);
        openmpt_module_set_render_param(mod, OPENMPT_MODULE_RENDER_STEREOSEPARATION_PERCENT, ss);
        fclose(file);
        cv.notify_one();
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

size_t xres = 3200/4;
size_t yres = 1800/4;

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
            float alpha = t.data[y*((int)t.x)*4 + ((int)t.x)*4 + 3]/255.0f;
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3)] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3)] + t.data[y*((int)t.x)*4 + ((int)t.x)*4]*(1-alpha);
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 1] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 1] + t.data[y*((int)t.x)*4 + ((int)t.x)*4 + 1]*(1-alpha);
            fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 2] = alpha*fb[(offsety*yfb*3 + y*yfb*3) + ((offsetx + x)* 3) + 2] + t.data[y*((int)t.x)*4 + ((int)t.x)*4 + 2]*(1-alpha);

        }
    }
}
struct Effect {
    float length;
    void (*func)(float, float*);
};

struct Star {
    struct vec3 location;
    float speed;
};

float clip(float n, float lower, float upper) {
    return fmax(lower, fmin(n, upper));
}

size_t yscr = 1800;
size_t xscr = 3200;

float cmul = 1.0f;

struct vec3 vec_make(float x, float y, float z) {
    struct vec3 xyz;
    xyz.x = x;
    xyz.y = y;
    xyz.z = z;
    return xyz;
}

static struct SceneAOS chess;

int frame = 0;
int fps = 0;
GLFWwindow *win;
float horizontal = 0.0f, vertical = 0.0f;
float otime = 0.0f;

struct Camera camera;
uint8_t *db;
float *fb;
uint16_t *pxc;
float *gfb;
float lastframe;

void error(){
    int err = glGetError();
    if(err != GL_NO_ERROR)
        throw std::runtime_error("hi");
}

bool frchange = 0;
int changed = 0;
float rendertime = 0;
void renderWorker() {
    while(true) {
        Texture tex;
        tex.data = db;
        tex.y = yres;
        tex.x = xres;
        std::mutex mtx;
        mtx.lock();
        bool chcopy = changed;
        struct Camera ccopy = camera;
        changed = 0;
        mtx.unlock();
        float itime = glfwGetTime();
        if(chcopy) {
            for(size_t x = 0; x < xres; x++) {
                for(size_t y = 0; y < yres; y++) {
                    pxc[y * xres + x] = 0;
                }
            }
            memset(fb, 0, xres*yres*3*4);
        }
        trace(chess, &tex, ccopy);
        rendertime = glfwGetTime()-itime;


        for(size_t x = 0; x < xres; x++) {
            for(size_t y = 0; y < yres; y++) {
                fb[y * xres*3 + x*3] = (((fb[y * xres*3 + x*3]) * pxc[y * xres + x]) + (db[y * xres*3 + x*3]/255.0f)) / (pxc[y * xres + x] + 1);
                fb[y * xres*3 + x*3 + 1] = (((fb[y * xres*3 + x*3 + 1]) * pxc[y * xres + x]) + (db[y * xres*3 + x*3 + 1]/255.0f)) / (pxc[y * xres + x] + 1);
                fb[y * xres*3 + x*3 + 2] = (((fb[y * xres*3 + x*3 + 2]) * pxc[y * xres + x]) + (db[y * xres*3 + x*3 + 2]/255.0f)) / (pxc[y * xres + x] + 1);
                pxc[y * xres + x]++;
            }
        }
        frchange = 1;
    }
}
void getLatestFrame(bool noscale, bool nodraw = false) {
    std::mutex mtx;
    mtx.lock();
    if(frchange) {
        for(size_t x = 0; x < xres; x++) {
            for(size_t y = 0; y < yres; y++) {
                gfb[y * xres*3 + x*3] = fastPow(fb[y * xres*3 + x*3], 1 / 2.2f);
                gfb[y * xres*3 + x*3 + 1] = fastPow(fb[y * xres*3 + x*3 + 1], 1 / 2.2f);
                gfb[y * xres*3 + x*3 + 2] = fastPow(fb[y * xres*3 + x*3 + 2], 1 / 2.2f);
            }
        }
        gfb[0 * xres*3 + 0*3] = 1.0f;
        gfb[0 * xres*3 + 0*3 + 1] = 0.0f;
        gfb[0 * xres*3 + 0*3 + 2] = 0.0f;
        gfb[0 * xres*3 + (xres-1)*3] = 1.0f;
        gfb[0 * xres*3 + (xres-1)*3 + 1] = 0.0f;
        gfb[0 * xres*3 + (xres-1)*3 + 2] = 0.0f;
        gfb[(yres-1) * xres*3 + 0*3] = 1.0f;
        gfb[(yres-1) * xres*3 + 0*3 + 1] = 0.0f;
        gfb[(yres-1) * xres*3 + 0*3 + 2] = 0.0f;
        gfb[(yres-1) * xres*3 + (xres-1)*3] = 1.0f;
        gfb[(yres-1) * xres*3 + (xres-1)*3 + 1] = 0.0f;
        gfb[(yres-1) * xres*3 + (xres-1)*3 + 2] = 0.0f;
        frchange = 0;
    }
    mtx.unlock();
    if(!nodraw) {
        glRasterPos2i(0, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        if(noscale) {
            glPixelZoom(4, 4);
        } else
            glPixelZoom(((float)xscr)/xres, ((float)yscr)/yres);
        //glPixelZoom((float)xscr/xres, (float)yscr/yres);
        glDrawPixels(xres, yres, GL_RGB, GL_FLOAT, gfb);
    }
}

void drawLine(float x1, float x2, float y1, float y2, struct vec3 color) {
    glLineWidth(2.5f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glBegin(GL_LINES);
    glColor3f(color.x, color.y, color.z);
    glVertex3f(x1*100*(9/16.0f), (y1*100), 0);
    glVertex3f(x2*100*(9/16.0f), (y2*100), 0);
    glColor3f(1.0f, 1.0f, 1.0f);
    glEnd();
    glViewport(0,0, xscr, yscr);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void drawTextToFB(float scl, const char *t, float xposc, float yposc, float brightness) {
    return;
    glViewport(0, 0, xscr, yscr);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho( 0, xscr, 0, yscr, -1, 1);
    //gluOrtho2D(0, xscr, 0, yscr);
    glMatrixMode(GL_MODELVIEW);
    glRasterPos2i(0,0);
    glLoadIdentity();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    struct vec3 alpha;
    alpha.x = 1.0f;
    alpha.y = 1.0f;
    alpha.z = 1.0f;
    alpha.w = 1.0f;
    unsigned char *tdata = drawText(t, alpha);
    struct Texture tex;
    tex.data = tdata;
    tex.scale = scl;
    tex.x = xposc;
    tex.y = yposc;
    tex.xoff = 9 * strlen(t);
    tex.yoff = 15;
    float truex = tex.xoff*scl;
    float truey = tex.yoff*scl;
    unsigned int tx = truex;
    unsigned int ty = truey;
    GLuint gtex;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gtex);
    glBindTexture(GL_TEXTURE_2D, gtex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.xoff, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
    glBindTexture(GL_TEXTURE_2D, gtex);
    glBegin(GL_QUADS);
    float xinscr = xscr*tex.x;
    float yinscr = yscr*tex.y;
    float xadj = xinscr - truex/2;
    float yadj = yinscr - truey/2;
    struct vec3 v1 = vec_make(xadj,  yadj, 0);
    struct vec3 v2 = vec_make(xadj + truex, yadj, 0);
    struct vec3 v3 = vec_make(xadj + truex, yadj + truey, 0);
    struct vec3 v4 = vec_make(xadj, yadj + truey, 0);
    glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, -1);
    glTexCoord2f(1, 0); glVertex3f(v2.x, v2.y, -1);
    glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, -1);
    glTexCoord2f(0, 1); glVertex3f(v4.x, v4.y, -1);
    glEnd();
    glDeleteTextures(1, &gtex);
    glDisable(GL_TEXTURE_2D);
    free(tdata);
    glDisable(GL_BLEND);
    glViewport(0,0, xscr, yscr);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
static float randfloat() {
    return ((next() % 16777217) / 16777216.0f);
}
struct Line {
    float x1;
    float x2;
    float y1;
    float y2;
    struct vec3 color;
};
bool test = 0;
bool resetfb = 0;
bool resetfb2 = 0;
std::vector<struct Line> slines;

void extract(float *fb, float *tex, int xoff, int yoff, int xs, int ys, int fbx, int fby) {
    for(int i = 0; i < xs; i++) {
        for(int j = 0; j < ys; j++) {
            tex[(j)*xs*3 + (i)*3] = fb[(j+yoff)*fbx*3 + (xoff+i)*3];
            tex[(j)*xs*3 + (i)*3 + 1] = fb[(j+yoff)*fbx*3 + (xoff+i)*3 + 1];
            tex[(j)*xs*3 + (i)*3 + 2] = fb[(j+yoff)*fbx*3 + (xoff+i)*3 + 2];
        }
    }
}
void chessf(float time, float *fb) {
    lastframe = glfwGetTime();glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.xoff, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
    glBindTexture(GL_TEXTURE_2D, gtex);
    glBegin(GL_QUADS);
    float xinscr = xscr*tex.x;
    float yinscr = yscr*tex.y;
    float xadj = xinscr - truex/2;
    float yadj = yinscr - truey/2;
    struct vec3 v1 = vec_make(xadj,  yadj, 0);
    struct vec3 v2 = vec_make(xadj + truex, yadj, 0);
    struct vec3 v3 = vec_make(xadj + truex, yadj + truey, 0);
    struct vec3 v4 = vec_make(xadj, yadj + truey, 0);
    glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, -1);
    glTexCoord2f(1, 0); glVertex3f(v2.x, v2.y, -1);
    glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, -1);
    glTexCoord2f(0, 1); glVertex3f(v4.x, v4.y, -1);
    glEnd();
    glDeleteTextures(1, &gtex);
    glDisable(GL_TEXTURE_2D);
    free(tdata);
    glDisable(GL_BLEND);
    glViewport(0,0, xscr, yscr);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
static float randfloat() {
    return ((next() % 16777217) / 16777216.0f);
}
struct Line {
    float x1;
    float x2;
    float y1;
    float y2;
    struct vec3 color;
};
bool test = 0;
bool resetfb = 0;
bool resetfb2 = 0;
std::vector<struct Line> slines;

void extract(float *fb, float *tex, int xoff, int yoff, int xs, int ys, int fbx, int fby) {
    for(int i = 0; i < xs; i++) {
        for(int j = 0; j < ys; j++) {
            tex[(j)*xs*3 + (i)*3] = fb[(j+yoff)*fbx*3 + (xoff+i)*3];
            tex[(j)*xs*3 + (i)*3 + 1] = fb[(j+yoff)*fbx*3 + (xoff+i)*3 + 1];
            tex[(j)*xs*3 + (i)*3 + 2] = fb[(j+yoff)*fbx*3 + (xoff+i)*3 + 2];
        }
    }
}
void chessf(float time, float *fb) {
    lastframe = glfwGetTime();
    glViewport(0,0, xscr, yscr);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    std::mutex mtx;
    mtx.lock();
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    int w,h;
    glfwGetWindowSize(win, &w, &h);
    double xpos = w/2, ypos = h/2;
    float mspeed = 0.005f;
    glfwGetCursorPos(win, &xpos, &ypos);
    glfwSetCursorPos(win, w/2, h/2);
    if(-(w/2- xpos) || (h/2 - ypos)) {
        changed = true;
    }
    horizontal += mspeed * -(w/2- xpos);
    vertical += mspeed * (h/2 - ypos);
    getLatestFrame(false);
    //    if(-(w/2- xpos) || (h/2 - ypos)) {
    //        changed = 1;
    //    }s
    //    if (vertical > 1.5f) {
    //        vertical = 1.5f;
    //    }
    //    else if (vertical < -1.5f) {
    //        vertical = -1.5f;
    //    }
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
    mtx.unlock();
    //        camera.up.x = -0.131286;
    //        camera.up.y = 0.984726;
    //        camera.up.z = 0.114358;
    //        camera.lookat.x = -0.743046;
    //        camera.lookat.y = -0.174108;
    //        camera.lookat.z = 0.646196;
    //        right.x = 0.656819;
    //        right.y = 0.0f;
    //        right.z  = 0.754048;
    if(!test) {
        float r = .1;
        //float a=  3.14159 - 3.14159*(1/4.0f);
        float a = ((randfloat()) * 3.14159) - 3.14159*(1/4.0f);
        float x = r * cos(a);
        float y = r * sin(a);
        drawTextToFB(1.5, "3. Sample a direction in the in the hemisphere of the normal", .4, .3, 1.0);
        struct Line l;
        l.x1 = .85;
        l.x2 = .85 + x;
        l.y1 = .85;
        l.y2 = .85 + y;
        l.color = vec_make(randfloat(), randfloat(), randfloat());
        slines.push_back(l);
        changed = 1;
        std::thread t(renderWorker);
        t.detach();
        test = 1;
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#ifdef CAMERA_PRINT
    char *str = vec_sprint(camera.center);
    glRasterPos2i(1, 1);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free((void*)str);
    str = vec_sprint(camera.lookat);
    glRasterPos2i(1, 3);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free((void*)str);
    glRasterPos2i(1, 5);
    str = vec_sprint(camera.up);
    glRasterPos2i(1, 7);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    free((void*)str);
#endif
    frame++;
    float ctime = glfwGetTime();
    if(ctime > otime+1) {
        float tdiff = ctime-otime;
        otime = ctime;
        fps = frame/tdiff;
        frame = 0;
    }
    float dist = glfwGetTime()-lastframe;
    char spf[(int)((ceil(log10(dist+1))+11)*sizeof(char))];
    sprintf(spf, "frametime: %f", rendertime);
    char fp[(int)((ceil(log10(fps+1))+6)*sizeof(char))];
    sprintf(fp, "FPS: %f", 1.0f/rendertime);
    char hangle[(int)((ceil(log10(fps+1))+6)*sizeof(char))];
    char vangle[(int)((ceil(log10(fps+1))+6)*sizeof(char))];

    sprintf(hangle,"HAS: %f", horizontal);
    sprintf(vangle,"VAS: %f", vertical);

    glRasterPos2i(1, 10);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)fp);
    glRasterPos2i(1, 13);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)spf);
    glRasterPos2i(1, 16);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)hangle);
    glRasterPos2i(1, 19);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)vangle);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

int main(int argc, char* argv[])
{
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
        tris.mat.eval = blueish;
        tris.mat.emit = 0.0f;
        tris2.mat = mat;
        tris3.mat = mat;
        help.push_back(tris);
        help.push_back(tris2);
        help.push_back(tris3);
    }
    vector vec;
    struct Effect chessgame;
    chessgame.length = 954354354.0f;
    chessgame.func = chessf;
    struct Effect credits;
    credits.length = 6.5f;
    vector_init(&vec);
    vector_add(&vec, &chessgame);
    vector_add(&vec, &credits);
    glfwInit();
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);;
    win = glfwCreateWindow(xscr, yscr, "hi", /*glfwGetPrimaryMonitor()*/ NULL, NULL);
    glfwGetFramebufferSize(win, (int*)&xscr, (int*)&yscr);
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
    s.radius   = 0.4f;
    s.origin.z = 0.0f;
    s.mat.diffuse = 1.0f;
    s.mat.reflect = 0.0f;
    s.mat.eval = red;
    s.mat.emit = 0.0f;
    t[0].pts[0].x = -4.0f;
    t[0].pts[0].z = -4.0f;
    t[0].pts[0].y =  0.0f;
    t[0].pts[1].x = 4.0f;
    t[0].pts[1].z = 4.0f;
    t[0].pts[1].y =  0.0f;
    t[0].pts[2].x = 4.0f;
    t[0].pts[2].z = -4.0f;
    t[0].pts[2].y =  0.0f;
    t[0].mat.diffuse = 0.2f;
    t[0].mat.reflect = 0.8f;
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
    t[1].mat.diffuse = 0.2f;
    t[1].mat.reflect = 0.8f;
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
    struct Material l;
    l.diffuse = 1.0f;
    l.reflect = 0.0f;
    l.refract = 0.0f;
    l.emit = 50.0f;
    l.eval = red;
    l.ior = 0.0f;
    struct StorageTriangle light[2];
    float closer = 10.0f;
    float pushaway = 2.0f;
    light[0].pts[0] = vec_make(-17 + closer, 9, -15 + closer);
    light[0].pts[1] = vec_make(-17 + closer -pushaway, 0, -15 + closer -pushaway);
    light[0].pts[2] = vec_make(-15 + closer -pushaway, 0, -15 + closer - 2 -pushaway);
    light[0].mat = l;
    light[1].pts[0] = vec_make(-17 + closer, 9, -15 + closer);
    light[1].pts[1] = vec_make(-15 + closer, 9, -15 + closer - 2);
    light[1].pts[2] = vec_make(-15 + closer-pushaway, 0, -15 + closer - 2 -pushaway);
    light[1].mat = l;
    help.push_back(light[0]);
    help.push_back(light[1]);

    chess = generateSceneGraphFromStorageAOS(help.data(), &s, help.size(), 0);
    fb = (float*)calloc(xres*yres*3*4, 1);
    db = (unsigned char*)calloc(xres*yres*3, 1);
    pxc = (unsigned short*)calloc(xres*yres*2, 1);
    gfb = (float*)calloc(xres*yres*3*4, 1);
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    float horizontal = 1.5f;
    float vertical = 0.0f;
    glfwSetCursorPos(win, xscr/2, yscr/2);
    glfwSetInputMode(win , GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
    //omp_set_num_threads(8);
    float time = glfwGetTime();
    float start = time;
    sceneaos = generateSceneGraphFromStorageAOS(t, &s, 2, 1);
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
    float velocity = 0.0f;
    while(!glfwWindowShouldClose(win)) {
        glfwGetFramebufferSize(win, (int*)&xscr, (int*)&yscr);
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
        float tn = (float)glfwGetTime();
        for(int i = 0; i < 1000; i++) {
            field[i].location.x -= tn-time;
        }
        time = tn;
        struct Effect *efx = (Effect*)vector_get(&vec, 0);
        if(time-start > efx->length) {
            vector_delete(&vec, 0);
            start = time;
            goto skip;
        }
        efx->func(time-start, fb);
skip:
        if(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            return -1;
        }

        unsigned int error = glGetError();
        if(error != GL_NO_ERROR) {
            printf("%d\n", error);
        }
        glfwSwapBuffers(win);
        glfwPollEvents();
        deallocScene(scene);
    }
    return 0;
}

    glViewport(0,0, xscr, yscr);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 100, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    std::mutex mtx;
    mtx.lock();
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    int w,h;
    glfwGetWindowSize(win, &w, &h);
    double xpos = w/2, ypos = h/2;
    float mspeed = 0.005f;
    glfwGetCursorPos(win, &xpos, &ypos);
    glfwSetCursorPos(win, w/2, h/2);
    if(-(w/2- xpos) || (h/2 - ypos)) {
        changed = true;
    }
    horizontal += mspeed * -(w/2- xpos);
    vertical += mspeed * (h/2 - ypos);
    getLatestFrame(false);
    //    if(-(w/2- xpos) || (h/2 - ypos)) {
    //        changed = 1;
    //    }s
    //    if (vertical > 1.5f) {
    //        vertical = 1.5f;
    //    }
    //    else if (vertical < -1.5f) {
    //        vertical = -1.5f;
    //    }
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
    mtx.unlock();
    //        camera.up.x = -0.131286;
    //        camera.up.y = 0.984726;
    //        camera.up.z = 0.114358;
    //        camera.lookat.x = -0.743046;
    //        camera.lookat.y = -0.174108;
    //        camera.lookat.z = 0.646196;
    //        right.x = 0.656819;
    //        right.y = 0.0f;
    //        right.z  = 0.754048;
    if(!test) {
        float r = .1;
        //float a=  3.14159 - 3.14159*(1/4.0f);
        float a = ((randfloat()) * 3.14159) - 3.14159*(1/4.0f);
        float x = r * cos(a);
        float y = r * sin(a);
        drawTextToFB(1.5, "3. Sample a direction in the in the hemisphere of the normal", .4, .3, 1.0);
        struct Line l;
        l.x1 = .85;
        l.x2 = .85 + x;
        l.y1 = .85;
        l.y2 = .85 + y;
        l.color = vec_make(randfloat(), randfloat(), randfloat());
        slines.push_back(l);
        changed = 1;
        std::thread t(renderWorker);
        t.detach();
        test = 1;
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    //    char *str = vec_sprint(camera.center);
    //    glRasterPos2i(1, 1);
    //    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    //    free((void*)str);
    //    str = vec_sprint(camera.lookat);
    //    glRasterPos2i(1, 3);
    //    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    //    free((void*)str);
    //    glRasterPos2i(1, 5);
    //    str = vec_sprint(camera.up);
    //    glRasterPos2i(1, 7);
    //    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)str);
    //    free((void*)str);
    frame++;
    float ctime = glfwGetTime();
    if(ctime > otime+1) {
        float tdiff = ctime-otime;
        otime = ctime;
        fps = frame/tdiff;
        frame = 0;
    }
    float dist = glfwGetTime()-lastframe;
    char spf[(int)((ceil(log10(dist+1))+11)*sizeof(char))];
    sprintf(spf, "frametime: %f", rendertime);
    char fp[(int)((ceil(log10(fps+1))+6)*sizeof(char))];
    sprintf(fp, "FPS: %f", 1.0f/rendertime);
    char hangle[(int)((ceil(log10(fps+1))+6)*sizeof(char))];
    char vangle[(int)((ceil(log10(fps+1))+6)*sizeof(char))];

    sprintf(hangle,"HAS: %f", horizontal);
    sprintf(vangle,"VAS: %f", vertical);

    glRasterPos2i(1, 10);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)fp);
    glRasterPos2i(1, 13);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)spf);
    glRasterPos2i(1, 16);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)hangle);
    glRasterPos2i(1, 19);
    glutBitmapString(GLUT_BITMAP_8_BY_13, (unsigned char*)vangle);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

int main(int argc, char* argv[])
{
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
        tris.mat.eval = blueish;
        tris.mat.emit = 0.0f;
        tris2.mat = mat;
        tris3.mat = mat;
        help.push_back(tris);
        help.push_back(tris2);
        help.push_back(tris3);
    }
    vector vec;
    struct Effect chessgame;
    chessgame.length = 954354354.0f;
    chessgame.func = chessf;
    struct Effect credits;
    credits.length = 6.5f;
    vector_init(&vec);
    vector_add(&vec, &chessgame);
    vector_add(&vec, &credits);
    glfwInit();
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);;
    win = glfwCreateWindow(xscr, yscr, "hi", /*glfwGetPrimaryMonitor()*/ NULL, NULL);
    glfwGetFramebufferSize(win, (int*)&xscr, (int*)&yscr);
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
    s.radius   = 0.4f;
    s.origin.z = 0.0f;
    s.mat.diffuse = 1.0f;
    s.mat.reflect = 0.0f;
    s.mat.eval = red;
    s.mat.emit = 0.0f;
    t[0].pts[0].x = -4.0f;
    t[0].pts[0].z = -4.0f;
    t[0].pts[0].y =  0.0f;
    t[0].pts[1].x = 4.0f;
    t[0].pts[1].z = 4.0f;
    t[0].pts[1].y =  0.0f;
    t[0].pts[2].x = 4.0f;
    t[0].pts[2].z = -4.0f;
    t[0].pts[2].y =  0.0f;
    t[0].mat.diffuse = 0.2f;
    t[0].mat.reflect = 0.8f;
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
    t[1].mat.diffuse = 0.2f;
    t[1].mat.reflect = 0.8f;
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
    struct Material l;
    l.diffuse = 1.0f;
    l.reflect = 0.0f;
    l.refract = 0.0f;
    l.emit = 50.0f;
    l.eval = red;
    l.ior = 0.0f;
    struct StorageTriangle light[2];
    float closer = 10.0f;
    float pushaway = 2.0f;
    light[0].pts[0] = vec_make(-17 + closer, 9, -15 + closer);
    light[0].pts[1] = vec_make(-17 + closer -pushaway, 0, -15 + closer -pushaway);
    light[0].pts[2] = vec_make(-15 + closer -pushaway, 0, -15 + closer - 2 -pushaway);
    light[0].mat = l;
    light[1].pts[0] = vec_make(-17 + closer, 9, -15 + closer);
    light[1].pts[1] = vec_make(-15 + closer, 9, -15 + closer - 2);
    light[1].pts[2] = vec_make(-15 + closer-pushaway, 0, -15 + closer - 2 -pushaway);
    light[1].mat = l;
    help.push_back(light[0]);
    help.push_back(light[1]);

    chess = generateSceneGraphFromStorageAOS(help.data(), &s, help.size(), 0);
    fb = (float*)calloc(xres*yres*3*4, 1);
    db = (unsigned char*)calloc(xres*yres*3, 1);
    pxc = (unsigned short*)calloc(xres*yres*2, 1);
    gfb = (float*)calloc(xres*yres*3*4, 1);
    struct vec3 right = vec_cross(camera.up, camera.lookat);
    float horizontal = 1.5f;
    float vertical = 0.0f;
    glfwSetCursorPos(win, xscr/2, yscr/2);
    glfwSetInputMode(win , GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
    //omp_set_num_threads(8);
    float time = glfwGetTime();
    float start = time;
    sceneaos = generateSceneGraphFromStorageAOS(t, &s, 2, 1);
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
    float velocity = 0.0f;
    while(!glfwWindowShouldClose(win)) {
        glfwGetFramebufferSize(win, (int*)&xscr, (int*)&yscr);
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
        float tn = (float)glfwGetTime();
        for(int i = 0; i < 1000; i++) {
            field[i].location.x -= tn-time;
        }
        time = tn;
        struct Effect *efx = (Effect*)vector_get(&vec, 0);
        if(time-start > efx->length) {
            vector_delete(&vec, 0);
            start = time;
            goto skip;
        }
        efx->func(time-start, fb);
skip:
        if(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            return -1;
        }

        unsigned int error = glGetError();
        if(error != GL_NO_ERROR) {
            printf("%d\n", error);
        }
        glfwSwapBuffers(win);
        glfwPollEvents();
        deallocScene(scene);
    }
    return 0;
}
