#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"
#include <iostream>
#include <stdio.h>
#include <fstream>  // ofstream
#include <time.h>   // clock_t, time_t
#include <math.h>
using namespace cv;
using namespace std;

ofstream training_file;
ofstream log_file;
char src_name[50];
int file_index; 
bool mouse_keep = false;
Point mouse_start, mouse_end;
Rect  mouse_rect;
vector<Vec3b> color_array;
Mat src;
Mat src_HSV;
Mat dst_thresh;
Mat dst_door;
Scalar hsv_Threshhold_Lower;
Scalar hsv_Threshhold_Upper;
Scalar hsv_Average;
Scalar hsv_Variance;

void onMouse( int event, int x, int y, int, void* );
int  getRectColor(Rect &rect, Mat &m_src, vector<Vec3b> &color_hsv);
int  getColorThresh(vector<Vec3b> &color_hsv, const Mat &m_src, Mat &m_dst);
int  getDoor(const Mat &m_src, Mat &m_dst);
void saveResult(int index);


int main(int argc, char** argv)
{
  training_file.open("training_data.txt", ios_base::app | ios_base::out);
  log_file.open("log.txt");
  sprintf(src_name, "%s", argv[1]);
  file_index = 0;
  /* read the image file*/
  src = imread(src_name);
  
  if(src.empty())
  {
    log_file << "load image error" << endl;
    cout     << "load image error" << endl;
    return -1;
  }

  log_file << "load image success" << endl;
  time_t timer;
  struct tm * timeinfo;
  time(&timer);
  timeinfo = localtime(&timer);
  training_file << asctime(timeinfo);
  
  //zoom out 1/8
  pyrDown(src, src);
  pyrDown(src, src);
  pyrDown(src, src);
  imshow("source", src);
 
  // Reduce noise with a kernel 3x3
  GaussianBlur(src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
 
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
  training_file.close();
  log_file.close();
}

/* responsible to mouse event */
void onMouse( int event, int x, int y, int, void* )
{
  if(event == EVENT_LBUTTONDOWN)
  {
    log_file << "left button down" << endl;
    if(mouse_keep == true)
    {
      mouse_keep = false;
      /* store the trainning information */
      mouse_end = Point(x, y);
      mouse_rect = Rect(mouse_start, mouse_end);
      cout << "end: " << mouse_end << endl;
      
      /* for a choosen rectangle, filter the similar color to  dst_thresh */
      if(getRectColor(mouse_rect, src_HSV, color_array) == 0)
      {
      
        getColorThresh(color_array, src_HSV, dst_thresh);
        imshow("color detection", dst_thresh);  
      
        /* for a given image, find its max contour*/
        getDoor(dst_thresh, dst_door);
        imshow("door detection", dst_door);
      }
    }
    else
    {
      mouse_keep = true;
      mouse_start = Point(x, y);
      cout << "start: " << mouse_start << endl;
    }
    
  }
  else if(event == EVENT_RBUTTONDOWN)  /* when find a good threshhold, just press the right mouse button and the threshes will be recored*/
  {
    saveResult(file_index);
    file_index++;
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


int getRectColor(Rect &rect, Mat &m_src, vector<Vec3b> &color_hsv)
{
  cout <<"rectangle: "<<rect << endl;
  color_hsv.clear();
  for(int x = rect.x; x < rect.width + rect.x; x++)
  {
   for(int y = rect.y; y < rect.height + rect.y; y++)
   { 
     Vec3b color = m_src.at<Vec3b>(Point(x,y));
     color_hsv.push_back(color);  
   }
  }
  
  int size = color_hsv.size();
  if(size == 0) return -1;
  
  return 0;
}

int getColorThresh(vector<Vec3b> &color_hsv, const Mat &m_src, Mat &m_dst)
{
  Vec3b min = color_hsv.at(0);
  Vec3b max = color_hsv.at(0);
  int sum[3] = {0};
  int size = color_hsv.size();

  /* compute piexl hsv_Average */
  for(int i = 0; i < size; i++)
  {
    Vec3b v = color_hsv.at(i);
    for(int j = 0; j < 3; j++)
    {
      sum[j] += v[j];
      min[j] = min[j] < v[j] ? min[j] : v[j];
      max[j] = max[j] > v[j] ? max[j] : v[j];
    }
  }
  
  for(int j = 0; j < 3; j++)
  {
    hsv_Average[j] = (int) sum[j] / size;
  }

  /* compute pixel hsv_Variance */  
  for(int i = 0; i < size; i++)
  {
    Vec3b v = color_hsv.at(i);
    for(int j = 0; j < 3; j++)
    {
      hsv_Variance[j] += pow(v[j] - hsv_Average[j], 2);
    }
  }
  for(int j = 0; j < 3; j++)
  {
    hsv_Variance[j] = (int) hsv_Variance[j] / size;
    int K = 1;
    hsv_Threshhold_Lower[j] = (hsv_Average[j] - K * hsv_Variance[j]) < 0   ? 0   : (hsv_Average[j] - K * hsv_Variance[j]);
    hsv_Threshhold_Upper[j] = (hsv_Average[j] + K * hsv_Variance[j]) > 256 ? 256 : (hsv_Average[j] + K * hsv_Variance[j]);
  }
  cout << " [pixel hsv_Average]  " << hsv_Average          << endl
       << " [min piexl]          " << min                  << endl
       << " [max pixel]          " << max                  << endl
       << " [pixel hsv_Variance] " << hsv_Variance         << endl
       << " [Lower threshhold]   " << hsv_Threshhold_Lower << endl 
       << " [Upper threshold]    " << hsv_Threshhold_Upper << endl << endl;
  
  inRange(m_src, hsv_Threshhold_Lower, hsv_Threshhold_Upper, m_dst);  

  return 0;
}

/* for a given image, find its max contour*/
int getDoor(const Mat &m_src, Mat &m_dst)
{

  m_dst = src.clone();
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(m_src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));  // try other mode and method

  /* find the max area is not a good way to detect the door edge */
  double max_area = 0;
  int max_index   = 0;
  for(unsigned int i = 0; i < contours.size(); i++)
  {
    double area = contourArea(contours[i], false);
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
  
  return 0;
}

void saveResult(int index)
{
  char thresh_name[50];
  char door_name[50];
  training_file << " [file]             " << src_name             << endl 
                << " [hsv average ]     " << hsv_Average          << endl
                << " [hsv variance]     " << hsv_Variance         << endl
                << " [Low HSV thresh]   " << hsv_Threshhold_Lower << endl
                << " [Upper HSV thresh] " << hsv_Threshhold_Upper << endl;

  cout          << " [file]             " << src_name             << endl 
                << " [hsv average ]     " << hsv_Average          << endl
                << " [hsv variance]     " << hsv_Variance         << endl
                << " [Low HSV thresh]   " << hsv_Threshhold_Lower << endl
                << " [Upper HSV thresh] " << hsv_Threshhold_Upper << endl;
  
  sprintf(thresh_name,"%s_%d_thresh.jpg", src_name, index);
  imwrite(thresh_name, dst_thresh);
  
  sprintf(door_name,"%s_%d_result.jpg", src_name, index);
  imwrite(door_name, dst_door);


  ofstream file_H, file_S, file_V;
  file_H.open("door/color_H.txt"); 
  file_S.open("door/color_S.txt");
  file_V.open("door/color_S.txt");

  for(int i = 0; i < color_array.size(); i++)
  {
    Vec3b v = color_array.at(i);
    file_H << (int)v[0] << endl;
    file_S << (int)v[1] << endl;
    file_V << (int)v[2] << endl;
  }
    
  file_H.close();
  file_S.close();
}
