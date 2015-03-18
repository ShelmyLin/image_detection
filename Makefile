CC = g++
CFLAGS = -Wall -g
INCLUDES = -I/usr/local/include/opencv -I/usr/include/opencv2
LFLAGS = -L/usr/local/lib/
SRCS = door_detection.cpp
TARGET = run
DEPEND = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching -std=c++11
all:
	$(CC) $(INCLUDES) $(LFLAGS) $(CFLAGS) -o $(TARGET) $(SRCS) $(DEPEND)
clean:
	rm *.o
