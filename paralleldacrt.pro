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
QMAKE_CXXFLAGS += -O0 -std=gnu11 -fopenmp -ffast-math -mtune=native -fpermissive
QMAKE_CFLAGS += -O0 -std=gnu11 -fopenmp -ffast-math -mtune=native -fpermissive

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

