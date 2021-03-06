TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    main2.cpp \
    vector.c \
    libfont.c \
    textures.c \
    scene.c \
    intersection_tests.c \
    naive.c \
    dacrt.c \
    raytracer.c

LIBS += -lglfw -lGLEW -lGL -lgomp -lglut -lGLU -lm -lportaudio -Wl,-Bdynamic -lopenmpt -lpthread -lXxf86vm -lXi -lXrandr -lGL -lGLU -ldl -lX11 -lasound -lgomp
QMAKE_CXXFLAGS += -O2 -fopenmp -fpermissive -ffast-math -std=c++11 -Wno-old-style-casts
QMAKE_CFLAGS += -lm -O2 -std=gnu11 -fopenmp -ffast-math

HEADERS += \
    tinyobj_loader_c.h \
    vector.h \
    libfont.h \
    veclib.h \
    ray_structs.h \
    textures.h \
    scene.h \
    intersection_tests.h \
    naive.h \
    dacrt.h \
    raytracer.h

