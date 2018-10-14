TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    vector.c \
    libfont.c \
    main2.cpp \
    textures.c \
    scene.c \
    intersection_tests.c \
    naive.c \
    dacrt.c \
    raytracer.c

LIBS += -lglfw -lGLEW -lGL -lgomp -lglut -lGLU -lm -lportaudio -Wl,-Bdynamic -lopenmpt -lpthread -lXxf86vm -lXi -lXrandr -lGL -lGLU -ldl -lX11 -lasound
QMAKE_CXXFLAGS += -O2 -fopenmp -fpermissive -ffast-math -mtune=power9 -mcpu=power9 -std=c++11
QMAKE_CFLAGS += -lm -O2 -std=gnu11 -fopenmp -ffast-math -mtune=power9 -mcpu=power9

HEADERS += \
    vector.h \
    libfont.h \
    veclib.h \
    obj.h \
    ray_structs.h \
    textures.h \
    scene.h \
    intersection_tests.h \
    naive.h \
    dacrt.h \
    raytracer.h

