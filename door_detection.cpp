/* Written by Xiongmin Lin <linxiongmin@gmail.com>, ISIMA, Clermont-Ferrand  *
 * (c++) 2015. All rights reserved.                                          */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"
#include <iostream>
#include <stdio.h>
#include <fstream>  // ofstream
#include <time.h>   // clock_t
#include <math.h>
using namespace cv;
using namespace std;

ofstream log_file;

bool mouse_keep = false;
Point mouse_start, mouse_end;
Rect  mouse_rect;
Mat src;
Mat src_HSV;
Mat dst_thresh;
Mat dst_door;
int lowerV = 0;
int upperV = 256;
int lowerS = 0;
int upperS = 256;
int lowerH = 0;
int upperH = 256;

void onMouse( int event, int x, int y, int, void* );
void getSimilarColor(Rect &rect, Mat &m_src, Mat &m_dst);
void getDoor(Mat &m_src, Mat &m_dst);

int main(int argc, char** argv)
{
  log_file.open("log.txt");
  char src_name[50];
  sprintf(src_name, "%s", argv[1]);

  /* read the image file*/
  src = imread(src_name);
  if(src.empty())
  {
    log_file << "load image error\n" << endl;
    cout     << "load image error\n" << endl;
    return -1;
  }

  //zoom out 1/8
  pyrDown(src, src);
  pyrDown(src, src);
  pyrDown(src, src);
  imshow("source", src);
  
  /* covert the source image from BGR to HSV  */
  cvtColor(src, src_HSV, CV_BGR2HSV);
  
  /* set mouse event*/
  setMouseCallback("source", onMouse, NULL);
 
  /* show image after thresh*/ 
  dst_thresh = src;
  imshow("color detection", dst_thresh);
  
  /* show the door*/
  dst_door = src;
  imshow("door detection", dst_door);

  waitKey(); 
  log_file.close();
}

/* responsible to mouse event */
void onMouse( int event, int x, int y, int, void* )
{
  if(event == EVENT_LBUTTONDOWN)
  {
    if(mouse_keep == true)
    {
      mouse_keep = false;
      /* store the trainning information */
      mouse_end = Point(x, y);
      mouse_rect = Rect(mouse_start, mouse_end);
      //cout << "end: " << mouse_end.x << ", " << mouse_end.y << endl;
      cout << "end: " << mouse_end << endl;
      /* for a choosen rectangle, filter the similar color to  dst_thresh */
      //showMatInfo(mouse_rect, src_HSV);
      getSimilarColor(mouse_rect, src_HSV, dst_thresh);
      imshow("color detection", dst_thresh);  
      /* for a given image, find its max contour*/
      getDoor(dst_thresh, dst_door);
      imshow("door detection", dst_door);
    }
    else
    {
      mouse_keep = true;
      mouse_start = Point(x, y);
      //cout << "start: " << mouse_start.x << ", " << mouse_start.y<< endl;
      cout << "start: " << mouse_start << endl;
    }
    
  }
  else if(event == EVENT_LBUTTONUP)
  {
  }
  else if(event == EVENT_RBUTTONDOWN)
  {
    
  }
  else if(event == EVENT_MOUSEMOVE)
  {
    if(mouse_keep)
    {
      mouse_end = Point(x, y);
      mouse_rect = Rect(mouse_start, mouse_end);
      Vec3b color = src_HSV.at<Vec3b>(Point(x,y));  
      cout << "moving: " << Point(x, y) << " pixel: "<< color << endl;
      //rectangle(src_HSV, mouse_rect,  Scalar(0,255, 255), 3, 8,0);// why the rectangle was not displayed
    }
  }
}

void getSimilarColor(Rect &rect, Mat &m_src, Mat &m_dst)
{
  cout <<"rectangle: "<<rect << endl;
  vector<Vec3b> color_array;
  for(int x = rect.x; x < rect.width + rect.x; x++)
  {
   for(int y = rect.y; y < rect.height + rect.y; y++)
   { 
     Vec3b color = m_src.at<Vec3b>(Point(x,y));
     color_array.push_back(color);  
   }
  }
  Vec3b average;
  Vec3b min = color_array.at(0);
  Vec3b max = color_array.at(0);
  int thresh_min[3] = {0};
  int thresh_max[3] = {0};
  int variance[3] = {0};
  int sum[3] = {0};
  int size = color_array.size();

  /* compute piexl average */
  for(int i = 0; i < size; i++)
  {
    Vec3b v = color_array.at(i);
    for(int j = 0; j < 3; j++)
    {
      sum[j] += v[j];
      min[j] = min[j] < v[j] ? min[j] : v[j];
      max[j] = max[j] > v[j] ? max[j] : v[j];
    }
  }
  
  for(int j = 0; j < 3; j++)
  {
    average[j] = sum[j] / size;
  }
  cout << "pixel average: " << average << ", min " << min << ", max: " << max << endl;   

  /* compute pixel variance */  
  for(int i = 0; i < size; i++)
  {
    Vec3b v = color_array.at(i);
    for(int j = 0; j < 3; j++)
    {
      variance[j] += pow(v[j] - average[j], 2);
    }
  }
  for(int j = 0; j < 3; j++)
  {
    variance[j] = variance[j] / size;
    int K = 3;
    thresh_min[j] = (average[j] - K * variance[j]) < 0   ? 0   : (average[j] - K * variance[j]);
    thresh_max[j] = (average[j] + K * variance[j]) > 256 ? 256 : (average[j] + K * variance[j]);
    cout << j << " thresh: [" << thresh_min[j] << ", " << thresh_max[j] << "]" << endl;
  }
  cout << "pixel variance: " << variance[0] <<", " <<variance[1] <<", " << variance[2] << endl;
  lowerH = thresh_min[0];
  upperH = thresh_max[0];
  lowerS = thresh_min[1];
  upperS = thresh_max[1];
  lowerV = thresh_min[2];
  upperV = thresh_max[2];
  
  inRange(m_src, Scalar(lowerH, lowerS, lowerV), Scalar(upperH, upperS, upperV), m_dst);  

}
/* for a given image, find its max contour*/
void getDoor(Mat &m_src, Mat &m_dst)
{

  m_dst = src.clone();
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(m_src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));  // try other mode and method
  //Scalar color( 255, 255, 255 );
  log_file << "get contours: " << contours.size() << endl;

  /* find the max area is not a good way to detect the door edge */
  double max_area = 0;
  int max_index   = 0;
  for(unsigned int i = 0; i < contours.size(); i++)
  {
    double area = contourArea(contours[i], false);
    log_file <<"NO." << i << endl <<"size: " << contours[i].size() << endl <<  "-> area: " <<  area << endl <<  " -> content: " << contours[i] << endl;
    if(area > max_area)
    {
      max_index = i;
      max_area  = area;
    }
  }
  Rect rect;
  rect = boundingRect(contours[max_index]);
  Scalar color( rand()&255, rand()&255, rand()&255 );
  //drawContours(m_dst, contours, max_index, color, 3, 8, hierarchy, 0, Point());
  rectangle(m_dst, rect, color, 3, 8,0);
  

}


