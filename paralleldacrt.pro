TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    main2.c \
    vector.c

LIBS += -lglfw -lGLEW -lGL -lgomp -lglut -lGLU -lm
QMAKE_CFLAGS += -O0 -fopenmp -ffast-math

HEADERS += \
    vector.h

