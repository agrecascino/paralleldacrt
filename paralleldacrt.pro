TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    vector.c \
    libfont.c \
    main2.cpp

LIBS += -Wl,-Bstatic -lglfw3 -lGLEW -lGL -lgomp -lglut -lGLU -lm -lportaudio_static -Wl,-Bdynamic -lopenmpt -lpthread -lXxf86vm -lXi -lXrandr -lGL -lGLU -ldl -lX11 -lasound
QMAKE_CXXFLAGS += -O3 -std=gnu11 -fopenmp -ffast-math -mtune=native -fpermissive -fPIC
QMAKE_CFLAGS += -O3 -std=gnu11 -fopenmp -ffast-math -mtune=native -fpermissive -fPIC

HEADERS += \
    vector.h \
    libfont.h \
    veclib.h \
    obj.h

