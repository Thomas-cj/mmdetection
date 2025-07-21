"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def is_moving_right_to_left(self, min_speed_threshold=0.5):
        """
        Checks if the object's estimated velocity indicates movement from right to left.
        """
        vx = self.kf.x[4, 0] # Estimated velocity in x-direction
        vy = self.kf.x[5, 0] # Estimated velocity in y-direction

        # Calculate the magnitude of the velocity
        speed = np.sqrt(vx**2 + vy**2)

        # Check if the x-velocity is negative and the object is actually moving
        if vx < 0 and speed > min_speed_threshold:
            return True
        else:
            return False
    
def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        


    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        
        # get predicted locations from existing trackers.
        trks_predicted_bboxes = np.zeros((len(self.trackers), 5))
        to_del_tracker_indices = []
        
        # Trackers that are predicted to be moving Left-to-Right
        # We will NOT update these with new detections, effectively allowing them to age out.
        predicted_l_to_r_tracker_indices = [] 

        for t_idx, tracker_obj in enumerate(self.trackers):
            pos = tracker_obj.predict()[0] # Predict next state, pos is [x1,y1,x2,y2]
            trks_predicted_bboxes[t_idx, :] = [pos[0], pos[1], pos[2], pos[3], 0] # Score doesn't matter for IOU

            if np.any(np.isnan(pos)):
                to_del_tracker_indices.append(t_idx)
            
            # Check if the *predicted* state of the tracker indicates L->R movement
            # We use a positive speed threshold here, as positive vx is L->R
            predicted_vx = tracker_obj.kf.x[4, 0]
            predicted_speed = np.sqrt(predicted_vx**2 + tracker_obj.kf.x[5, 0]**2)

            if predicted_vx > 0.5 and predicted_speed > 0.5:
                predicted_l_to_r_tracker_indices.append(t_idx)
            
        # Clean up trackers with NaN positions (should ideally not happen with a robust Kalman)
        # Using reversed order for pop to avoid index issues
        trks_predicted_bboxes = np.ma.compress_rows(np.ma.masked_invalid(trks_predicted_bboxes))
        for t_idx in reversed(to_del_tracker_indices):
            self.trackers.pop(t_idx)

        # Perform association between current detections and predicted tracker locations
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks_predicted_bboxes, self.iou_threshold)

        # Update matched trackers with assigned detections
        for m in matched:
            det_idx = m[0]
            tracker_idx = m[1]
            
            # CRITICAL FILTERING STEP:
            # Only update the tracker if its *predicted* movement is NOT left-to-right.
            # If it's predicted to be moving L->R, we let its time_since_update increment,
            # causing it to age out if it continues that movement.
            if tracker_idx not in predicted_l_to_r_tracker_indices:
                self.trackers[tracker_idx].update(dets[det_idx, :])
            # Else: Do nothing. The detection is considered "matched" but the track isn't updated.

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # When creating new trackers, we don't have a reliable velocity estimate yet.
            # So, we create the track, and rely on the filtering in the output loop
            # to only show it once its direction is established (R->L).
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        
        ret = [] # This will store the final filtered output bounding boxes with IDs
        
        # Iterate through trackers in reverse for safe popping
        for i in range(len(self.trackers) - 1, -1, -1):
            trk = self.trackers[i]
            d = trk.get_state()[0] # Get current estimated bbox [x1,y1,x2,y2]

            # Condition 1: Basic SORT eligibility (track is "active" or recently updated)
            is_active_sort_track = (trk.time_since_update < 1) and \
                                   (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)

            # Condition 2: Check if the track is moving right-to-left based on its CURRENT estimated velocity
            is_moving_right_to_left = trk.is_moving_right_to_left(0.5)

            if is_active_sort_track and is_moving_right_to_left:
                # Append to return list if both conditions are met
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1)) # +1 for MOT benchmark

            # Remove dead tracklets (either too old or lost too many updates)
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            # Sort by ID for consistent output
            ret = np.concatenate(ret)
            # You might want to sort by ID here for consistent output, though not strictly necessary.
            # ret = ret[np.argsort(ret[:, 4])] # if you want to sort by ID
            return ret
        return np.empty((0, 5))

#   def update(self, dets=np.empty((0, 5))):
#     """
#     Params:
#       dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#     Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
#     Returns the a similar array, where the last column is the object ID.

#     NOTE: The number of objects returned may differ from the number of detections provided.
#     """
#     self.frame_count += 1
#     # get predicted locations from existing trackers.
#     trks = np.zeros((len(self.trackers), 5))
#     to_del = []
#     ret = []
#     for t, trk in enumerate(trks):
#       pos = self.trackers[t].predict()[0]
#       trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#       if np.any(np.isnan(pos)):
#         to_del.append(t)
#     trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#     for t in reversed(to_del):
#       self.trackers.pop(t)
#     matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

#     # update matched trackers with assigned detections
#     for m in matched:
#       self.trackers[m[1]].update(dets[m[0], :])

#     # create and initialise new trackers for unmatched detections
#     for i in unmatched_dets:
#         trk = KalmanBoxTracker(dets[i,:])
#         self.trackers.append(trk)
#     i = len(self.trackers)
#     for trk in reversed(self.trackers):
#         d = trk.get_state()[0]
#         if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#           ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
#         i -= 1
#         # remove dead tracklet
#         if(trk.time_since_update > self.max_age):
#           self.trackers.pop(i)
#     if(len(ret)>0):
#       return np.concatenate(ret)
#     return np.empty((0,5))

