TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.c

LIBS += -lglfw -lGLEW -lGL -lgomp
QMAKE_CFLAGS += -O2 -fopenmp

HEADERS += \
    linmath.h
