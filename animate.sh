#!/bin/bash
#This script converts a series of images to a video using ffmpeg
ffmpeg -f image2 -framerate 10 -i 'Kentland_lcs_%04d.tif' -pix_fmt yuv420p -vf scale=1280:-2 'LCS.mp4'
