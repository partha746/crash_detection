#!/usr/bin/env python
"""OpenCV Lane detection system"""

import cv2
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import peakutils
import warnings

scale_ratio = 1
prev_leftx = None
prev_lefty = None
prev_rightx = None
prev_righty = None
prev_left_fit = []
prev_right_fit = []

prev_leftx2 = None
prev_lefty2 = None
prev_rightx2 = None
prev_righty2 = None
prev_left_fit2 = []
prev_right_fit2 = []


class Lane:
  def __init__(self, orig_frame):
    self.orig_frame = orig_frame
    self.lane_line_markings = None
    self.warped_frame = None
    self.transformation_matrix = None
    self.inv_transformation_matrix = None
    self.orig_image_size = self.orig_frame.shape[::-1][1:]

    width = self.orig_image_size[0]
    height = self.orig_image_size[1]
    self.width = width
    self.height = height
    self.roi_points = np.float32([
      (int(0.456*width),int(0.544*height)), # Top-left corner
      (0, height-1), # Bottom-left corner     
      (int(0.94*width),height-1), # Bottom-right corner
      (int(0.53*width),int(0.544*height)) # Top-right corner
    ])
    self.padding = int(0.25 * width) # padding from side of the image in pixels
    self.desired_roi_points = np.float32([
      [self.padding, 0], # Top-left corner
      [self.padding, self.orig_image_size[1]], # Bottom-left corner     
      [self.orig_image_size[
        0]-self.padding, self.orig_image_size[1]], # Bottom-right corner
      [self.orig_image_size[0]-self.padding, 0] # Top-right corner
    ]) 
    self.histogram = None
    self.no_of_windows = 10
    self.margin = int((1/12) * width)  # Window width is +/- margin
    self.minpix = int((1/24) * width)  # Min no. of pixels to recenter window
    
    self.left_fit = None
    self.right_fit = None
    self.left_lane_inds = None
    self.right_lane_inds = None
    self.ploty = None
    self.left_fitx = None
    self.right_fitx = None
    self.leftx = None
    self.rightx = None
    self.lefty = None
    self.righty = None
    
    self.YM_PER_PIX = 7.0 / 400 # meters per pixel in y dimension
    self.XM_PER_PIX = 3.7 / 255 # meters per pixel in x dimension
    
    self.left_curvem = None
    self.right_curvem = None
    self.center_offset = None

  def calculate_car_position(self, print_to_terminal=False):
    car_location = self.orig_frame.shape[1] / 2

    height = self.orig_frame.shape[0]
    bottom_left = self.left_fit[0]*height**2 + self.left_fit[
      1]*height + self.left_fit[2]
    bottom_right = self.right_fit[0]*height**2 + self.right_fit[
      1]*height + self.right_fit[2]

    center_lane = (bottom_right - bottom_left)/2 + bottom_left 
    center_offset = (np.abs(car_location) - np.abs(
      center_lane)) * self.XM_PER_PIX * 100

    if print_to_terminal == True:
      print(str(round(center_offset, 1)))
      
    self.center_offset = round(center_offset, 1)
      
    return round(center_offset, 1)

  def calculate_curvature(self, print_to_terminal=False):
    y_eval = np.max(self.ploty)    

    with warnings.catch_warnings():
      warnings.simplefilter('ignore', np.RankWarning)

      left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (
        self.XM_PER_PIX), 2)
      right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (
        self.XM_PER_PIX), 2)
      
    left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*self.YM_PER_PIX + left_fit_cr[
                    1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvem = ((1 + (2*right_fit_cr[
                    0]*y_eval*self.YM_PER_PIX + right_fit_cr[
                    1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
  
    if print_to_terminal == True:
      print(left_curvem, 'm', right_curvem, 'm')
      
    self.left_curvem = round(left_curvem, 1)
    self.right_curvem = round(right_curvem, 1)

    return left_curvem, right_curvem    
    
  def calculate_histogram(self,frame=None,plot=True):
    if frame is None:
      frame = self.warped_frame
      
    self.histogram = np.sum(frame[int(
          frame.shape[0]/2):,:], axis=0)

    if plot == True:
      figure, (ax1, ax2) = plt.subplots(2,1) # 2 row, 1 columns
      figure.set_size_inches(10, 5)
      ax1.imshow(frame, cmap='gray')
      ax1.set_title("Warped Binary Frame")
      ax2.plot(self.histogram)
      ax2.set_title("Histogram Peaks")
      plt.show()
      
    return self.histogram

  def display_curvature_offset(self, frame=None, plot=False):
    image_copy = None
    if frame is None:
      image_copy = self.orig_frame.copy()
    else:
      image_copy = frame

    # cv2.putText(image_copy,'Curve Radius: '+str((
    #   self.left_curvem+self.right_curvem)/2)[:7]+' m', (int((
    #   5/600)*self.width), int((
    #   20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
    #   0.5/600)*self.width)),(
    #   255,255,255),2,cv2.LINE_AA)
    cv2.putText(image_copy,'Center Offset: '+str(
      self.center_offset)[:7], (int((
      5/600)*self.width), int((
      20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
      0.5/600)*self.width)),(
      255,255,255),2,cv2.LINE_AA)
      
    if plot==True:       
      cv2.imshow("Image with Curvature and Offset", image_copy)
      
    return image_copy
    
  def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
    margin = self.margin
    nonzero = self.warped_frame.nonzero()  
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(
      nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
      nonzerox < (left_fit[0]*(
      nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(
      nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
      nonzerox < (right_fit[0]*(
      nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))       
    self.left_lane_inds = left_lane_inds
    self.right_lane_inds = right_lane_inds

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]  

    global prev_leftx2
    global prev_lefty2 
    global prev_rightx2
    global prev_righty2
    global prev_left_fit2
    global prev_right_fit2
    
    if len(leftx)==0 or len(lefty)==0 or len(rightx)==0 or len(righty)==0:
      leftx = prev_leftx2
      lefty = prev_lefty2
      rightx = prev_rightx2
      righty = prev_righty2

    self.leftx = leftx
    self.rightx = rightx
    self.lefty = lefty
    self.righty = righty
    
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', np.RankWarning)
      left_fit = np.polyfit(lefty, leftx, 2)
      right_fit = np.polyfit(righty, rightx, 2) 

    prev_left_fit2.append(left_fit)
    prev_right_fit2.append(right_fit)

    if len(prev_left_fit2) > 15:
      prev_left_fit2.pop(0)
      prev_right_fit2.pop(0)
      left_fit = sum(prev_left_fit2) / len(prev_left_fit2)
      right_fit = sum(prev_right_fit2) / len(prev_right_fit2)

    self.left_fit = left_fit
    self.right_fit = right_fit
    
    prev_leftx2 = leftx
    prev_lefty2 = lefty 
    prev_rightx2 = rightx
    prev_righty2 = righty
    
    ploty = np.linspace(
      0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0]) 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    self.ploty = ploty
    self.left_fitx = left_fitx
    self.right_fitx = right_fitx
    
    if plot==True:
      out_img = np.dstack((self.warped_frame, self.warped_frame, (
                           self.warped_frame)))*255
      window_img = np.zeros_like(out_img)
      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                                                                     0, 0, 255]
      margin = self.margin
      left_line_window1 = np.array([np.transpose(np.vstack([
                                    left_fitx-margin, ploty]))])
      left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                    left_fitx+margin, ploty])))])
      left_line_pts = np.hstack((left_line_window1, left_line_window2))
      right_line_window1 = np.array([np.transpose(np.vstack([
                                     right_fitx-margin, ploty]))])
      right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                     right_fitx+margin, ploty])))])
      right_line_pts = np.hstack((right_line_window1, right_line_window2))
      
      cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
      cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
      result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
      
      figure, (ax1, ax2, ax3) = plt.subplots(3,1)
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(self.warped_frame, cmap='gray')
      ax3.imshow(result)
      ax3.plot(left_fitx, ploty, color='yellow')
      ax3.plot(right_fitx, ploty, color='yellow')
      ax1.set_title("Original Frame")  
      ax2.set_title("Warped Frame")
      ax3.set_title("Warped Frame With Search Window")
      plt.show()
      
  def get_lane_line_indices_sliding_windows(self, plot=False):
    margin = self.margin

    frame_sliding_window = self.warped_frame.copy()

    window_height = np.int(self.warped_frame.shape[0]/self.no_of_windows)   

    nonzero = self.warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1]) 
    
    left_lane_inds = []
    right_lane_inds = []
    
    leftx_base, rightx_base = self.histogram_peak()
    leftx_current = leftx_base
    rightx_current = rightx_base

    no_of_windows = self.no_of_windows
    
    for window in range(no_of_windows):
      win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
      win_y_high = self.warped_frame.shape[0] - window * window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(
        win_xleft_high,win_y_high), (255,255,255), 2)
      cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(
        win_xright_high,win_y_high), (255,255,255), 2)

      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (
                           nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (
                            nonzerox < win_xright_high)).nonzero()[0]
                            
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
        
      minpix = self.minpix
      if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
          
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds] 
    righty = nonzeroy[right_lane_inds]

    left_fit = None
    right_fit = None
    
    global prev_leftx
    global prev_lefty 
    global prev_rightx
    global prev_righty
    global prev_left_fit
    global prev_right_fit

    if len(leftx)==0 or len(lefty)==0 or len(rightx)==0 or len(righty)==0:
      leftx = prev_leftx
      lefty = prev_lefty
      rightx = prev_rightx
      righty = prev_righty
    
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', np.RankWarning)
      left_fit = np.polyfit(lefty, leftx, 2)
      right_fit = np.polyfit(righty, rightx, 2) 

    prev_left_fit.append(left_fit)
    prev_right_fit.append(right_fit)

    if len(prev_left_fit) > 10:
      prev_left_fit.pop(0)
      prev_right_fit.pop(0)
      left_fit = sum(prev_left_fit) / len(prev_left_fit)
      right_fit = sum(prev_right_fit) / len(prev_right_fit)

    self.left_fit = left_fit
    self.right_fit = right_fit
    
    prev_leftx = leftx
    prev_lefty = lefty 
    prev_rightx = rightx
    prev_righty = righty

    if plot==True:
      ploty = np.linspace(
        0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

      out_img = np.dstack((
        frame_sliding_window, frame_sliding_window, (
        frame_sliding_window))) * 255
      
      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
        0, 0, 255]
        
      figure, (ax1, ax2, ax3) = plt.subplots(3,1)
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(frame_sliding_window, cmap='gray')
      ax3.imshow(out_img)
      ax3.plot(left_fitx, ploty, color='yellow')
      ax3.plot(right_fitx, ploty, color='yellow')
      ax1.set_title("Original Frame")  
      ax2.set_title("Warped Frame with Sliding Windows")
      ax3.set_title("Detected Lane Lines with Sliding Windows")
      plt.show()      
      
    return self.left_fit, self.right_fit

  def get_line_markings(self, frame=None):

    def binary_array(array, thresh, value=0):
      if value == 0:
        binary = np.ones_like(array) 
      else:
        binary = np.zeros_like(array)  
        value = 1

      binary[(array >= thresh[0]) & (array <= thresh[1])] = value
     
      return binary
     
    def blur_gaussian(channel, ksize=3):
      return cv2.GaussianBlur(channel, (ksize, ksize), 0)
             
    def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
      sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
      sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))
      mag = np.sqrt(sobelx ** 2 + sobely ** 2)
     
      return binary_array(mag, thresh)
     
    def sobel(img_channel, orient='x', sobel_kernel=3):
      if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
      if orient == 'y':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)
     
      return sobel
     
    def threshold(channel, thresh=(128,255), thresh_type=cv2.THRESH_BINARY):
      return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)

    if frame is None:
      frame = self.orig_frame
      
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    _, sxbinary = threshold(hls[:, :, 1], thresh=(120, 255))
    sxbinary = blur_gaussian(sxbinary, ksize=3)
    sxbinary = mag_thresh(sxbinary, sobel_kernel=3, thresh=(254, 255)) #110, 255

    s_channel = hls[:, :, 2]
    _, s_binary = threshold(s_channel, (0, 255))
    _, r_thresh = threshold(frame[:, :, 2], thresh=(175, 255)) #(150, 255)

    rs_binary = cv2.bitwise_and(s_binary, r_thresh)

    self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(
                              np.uint8))  
    return self.lane_line_markings
    
  def histogram_peak(self):
    midpoint = np.int(self.histogram.shape[0]/2)
    leftx_base = np.argmax(self.histogram[:midpoint])
    rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base
    
  def overlay_lane_lines(self, plot=False):
    warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))   
    
    pts_left = np.array([np.transpose(np.vstack([
                         self.left_fitx, self.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([
                          self.right_fitx, self.ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
                                  self.orig_frame.shape[
                                  1], self.orig_frame.shape[0]))
    
    result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)
    
    if plot==True:
      figure, (ax1, ax2) = plt.subplots(2,1) # 2 rows, 1 column
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
      ax1.set_title("Original Frame")  
      ax2.set_title("Original Frame With Lane Overlay")
      plt.show()   

    return result     
  
  def perspective_transform(self, frame=None, plot=False):
    if frame is None:
      frame = self.lane_line_markings
      
    self.transformation_matrix = cv2.getPerspectiveTransform(
      self.roi_points, self.desired_roi_points)

    self.inv_transformation_matrix = cv2.getPerspectiveTransform(
      self.desired_roi_points, self.roi_points)

    self.warped_frame = cv2.warpPerspective(
      frame, self.transformation_matrix, self.orig_image_size, flags=(
     cv2.INTER_LINEAR)) 

    (thresh, binary_warped) = cv2.threshold(
      self.warped_frame, 127, 255, cv2.THRESH_BINARY)     
    self.warped_frame = binary_warped

    if plot == True:
      warped_copy = self.warped_frame.copy()
      warped_plot = cv2.polylines(warped_copy, np.int32([
                    self.desired_roi_points]), True, (147,20,255), 3)

      while(1):
        cv2.imshow('Warped Image', warped_plot)
        if cv2.waitKey(0):
          break

      cv2.destroyAllWindows() 
      
    return self.warped_frame    
  
  def plot_roi(self, frame=None, plot=False):
    if plot == False:
      return
      
    if frame is None:
      frame = self.orig_frame.copy()

    this_image = cv2.polylines(frame, np.int32([
      self.roi_points]), True, (147,20,255), 3)

    while(1):
      cv2.imshow('ROI Image', this_image)
      if cv2.waitKey(0):
        break

    cv2.destroyAllWindows()
  
def detect_lane(video_file_name, show_video=False):
  car_position_arr = []
  cap = cv2.VideoCapture(video_file_name)
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  while cap.isOpened():
    success, frame = cap.read() 
    if success:
      # width = int(frame.shape[1] * scale_ratio)
      # height = int(frame.shape[0] * scale_ratio)
      frame = cv2.resize(frame, (800, 450))
      
      original_frame = frame.copy()

      lane_obj = Lane(orig_frame=original_frame)

      lane_line_markings = lane_obj.get_line_markings()

      lane_obj.plot_roi(plot=False)

      warped_frame = lane_obj.perspective_transform(plot=False)

      histogram = lane_obj.calculate_histogram(plot=False)  

      left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)

      lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
  
      frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)

      lane_obj.calculate_curvature(print_to_terminal=False)

      car_position_arr.append(lane_obj.calculate_car_position(print_to_terminal=False))
  
      frame_with_lane_lines2 = lane_obj.display_curvature_offset(
        frame=frame_with_lane_lines, plot=False)
        
      if show_video:
        cv2.imshow("Frame", frame_with_lane_lines2)   

        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    else:
      break

  cap.release()
  cv2.destroyAllWindows() 
  return car_position_arr, fps
  

DEF_PARSER = argparse.ArgumentParser(description='Script to detect lane data in a video')
DEF_PARSER.add_argument('--show-video','-s', dest='show_live', default=False, action='store_true', help='Show live detection')
DEF_PARSER.add_argument('--show-graph','-g', dest='show_graph', default=False, action='store_true', help='Show variance graph')
DEF_PARSER.add_argument('--file-path', '-f', dest='file_name', default='videos/vid1.mp4', type=str, help='Provide video path.')
DEF_PARSER.add_argument('--window', '-w', dest='window', default=5, type=int, help='window.')
PARSER_OBJ = DEF_PARSER.parse_args()

if PARSER_OBJ.file_name == None:
    DEF_PARSER.print_help()
    DEF_PARSER.error('Please provide video file path')
else:
  car_pos_arr, fps = detect_lane(PARSER_OBJ.file_name, PARSER_OBJ.show_live)
  for window in range(PARSER_OBJ.window, PARSER_OBJ.window+1):
    var_arr, mean_arr, std_arr = [], [], []
    for i in range(len(car_pos_arr)):
      windows_arr = []
      for j in range(i-1, i+window):
        try:
          windows_arr.append(car_pos_arr[j + 1])
        except:
          pass
      if len(windows_arr) == window+1:
        var_arr.append(np.var(windows_arr))
        mean_arr.append(np.mean(windows_arr))
        std_arr.append(np.std(windows_arr))

    peeks = peakutils.indexes(var_arr)
    if PARSER_OBJ.show_graph:
      # plt.plot(var_arr, label='Variance')
      plt.plot(mean_arr, label='Mean')
      plt.plot(std_arr, label='Standard Deviation')
      # plt.hist(var_arr)
      plt.legend()
      plt.show()
      plt.clf()

  # print(len(peeks))
  if len(peeks) > 2:
    print("Wobble Detected")
    print('Total Movements Detected : ' + str(len(peeks)))
    sys.exit(1)
  else:
    print("Wobble Not Detected")
    print('Total Movements Detected : ' + str(len(peeks)))
    sys.exit(0)
