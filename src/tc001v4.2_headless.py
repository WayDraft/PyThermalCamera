#!/usr/bin/env python3
'''
Les Wright 21 June 2023
https://youtube.com/leslaboratory
A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!
Modified for RTSP streaming and API access in headless mode.
'''
print('Les Wright 21 June 2023')
print('https://youtube.com/leslaboratory')
print('A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!')
print('Modified for RTSP streaming and API access.')
print('')
print('Tested on Debian all features are working correctly')
print('This will work on the Pi However a number of workarounds are implemented!')
print('Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!')
print('')
print('Running in headless mode - no GUI, RTSP streaming and API only.')

import cv2
import numpy as np
import argparse
import time
import io
import threading
from flask import Flask, jsonify, Response
import sys
import os

# Flask app for API
app = Flask(__name__)

# Global variables for HUD data
hud_data = {
    'avg_temp': 0.0,
    'label_threshold': 2,
    'colormap': 'Jet',
    'blur': 0,
    'scaling': 3,
    'contrast': 1.0,
    'snapshot': 'None',
    'recording': 'Not available in headless mode',
    'center_temp': 0.0,
    'max_temp': 0.0,
    'min_temp': 0.0
}

# Global frame for streaming
current_frame = None

@app.route('/api/hud')
def get_hud():
    return jsonify(hud_data)

@app.route('/shutdown')
def shutdown():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    os._exit(0)

@app.route('/video')
def video_feed():
    def generate():
        while True:
            if current_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', current_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)  # Adjust frame rate
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_api():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

#We need to know if we are running on the Pi, because openCV behaves a little oddly on all the builds!
#https://raspberrypi.stackexchange.com/questions/5100/detect-that-a-python-program-is-running-on-the-pi
def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False

isPi = is_raspberrypi()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()
	
if args.device:
	dev = args.device
else:
	dev = 0
	
#init video
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
#cap = cv2.VideoCapture(0)
#pull in the video but do NOT automatically convert to RGB, else it breaks the temperature data!
#https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
if isPi == True:
	cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
	cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

#256x192 General settings
width = 256 #Sensor width
height = 192 #sensor height
scale = 3 #scale multiplier
newWidth = width*scale 
newHeight = height*scale
alpha = 1.0 # Contrast control (1.0-3.0)
colormap = 0
font=cv2.FONT_HERSHEY_SIMPLEX
rad = 0 #blur radius
threshold = 2
hud = True
snaptime = "None"

snaptime = "None"

def snapshot(heatmap):
	#I would put colons in here, but it Win throws a fit if you try and open them!
	now = time.strftime("%Y%m%d-%H%M%S") 
	snaptime = time.strftime("%H:%M:%S")
	cv2.imwrite("TC001"+now+".png", heatmap)
	return snaptime

# Start API server in a separate thread
api_thread = threading.Thread(target=run_api)
api_thread.daemon = True
api_thread.start()

print(f'Video streaming at http://localhost:5000/video')
print('API available at http://localhost:5000/api/hud')
print('Shutdown via http://localhost:5000/shutdown')

while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
		imdata,thdata = np.array_split(frame, 2)
		#now parse the data from the bottom frame and convert to temp!
		#https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
		#Huge props to LeoDJ for figuring out how the data is stored and how to compute temp from it.
		#grab data from the center pixel...
		hi = int(thdata[96][128][0])
		lo = int(thdata[96][128][1]) * 256
		rawtemp = hi + lo
		temp = (rawtemp/64)-273.15
		temp = round(temp,2)

		#find the max temperature in the frame
		lomax = thdata[...,1].max()
		posmax = thdata[...,1].argmax()
		#since argmax returns a linear index, convert back to row and col
		mcol,mrow = divmod(posmax,width)
		himax = int(thdata[mcol][mrow][0])
		lomax=int(lomax)*256
		maxtemp = himax+lomax
		maxtemp = (maxtemp/64)-273.15
		maxtemp = round(maxtemp,2)

		
		#find the lowest temperature in the frame
		lomin = thdata[...,1].min()
		posmin = thdata[...,1].argmin()
		#since argmax returns a linear index, convert back to row and col
		lcol,lrow = divmod(posmin,width)
		himin = int(thdata[lcol][lrow][0])
		lomin=int(lomin)*256
		mintemp = himin+lomin
		mintemp = (mintemp/64)-273.15
		mintemp = round(mintemp,2)

		#find the average temperature in the frame
		loavg = thdata[...,1].mean()
		hiavg = thdata[...,0].mean()
		loavg=int(loavg)*256
		avgtemp = loavg+hiavg
		avgtemp = (avgtemp/64)-273.15
		avgtemp = round(avgtemp,2)		# Update HUD data
		hud_data['avg_temp'] = avgtemp
		hud_data['center_temp'] = temp
		hud_data['max_temp'] = maxtemp
		hud_data['min_temp'] = mintemp
		hud_data['label_threshold'] = threshold
		hud_data['blur'] = rad
		hud_data['scaling'] = scale
		hud_data['contrast'] = alpha
		hud_data['snapshot'] = snaptime

		# Convert the real image to RGB
		bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
		#Contrast
		bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
		#bicubic interpolate, upscale and blur
		bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)
		if rad>0:
			bgr = cv2.blur(bgr,(rad,rad))

		#apply colormap
		if colormap == 0:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
			cmapText = 'Jet'
		if colormap == 1:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_HOT)
			cmapText = 'Hot'
		if colormap == 2:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_MAGMA)
			cmapText = 'Magma'
		if colormap == 3:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)
			cmapText = 'Inferno'
		if colormap == 4:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PLASMA)
			cmapText = 'Plasma'
		if colormap == 5:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_BONE)
			cmapText = 'Bone'
		if colormap == 6:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_SPRING)
			cmapText = 'Spring'
		if colormap == 7:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_AUTUMN)
			cmapText = 'Autumn'
		if colormap == 8:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_VIRIDIS)
			cmapText = 'Viridis'
		if colormap == 9:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PARULA)
			cmapText = 'Parula'
		if colormap == 10:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_RAINBOW)
			heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
			cmapText = 'Inv Rainbow'

		hud_data['colormap'] = cmapText

		# draw crosshairs
		cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
		(int(newWidth/2),int(newHeight/2)-20),(255,255,255),2) #vline
		cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
		(int(newWidth/2)-20,int(newHeight/2)),(255,255,255),2) #hline

		cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
		(int(newWidth/2),int(newHeight/2)-20),(0,0,0),1) #vline
		cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
		(int(newWidth/2)-20,int(newHeight/2)),(0,0,0),1) #hline
		#show temp
		cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
		cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
		cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		if hud==True:
			# display black box for our data
			cv2.rectangle(heatmap, (0, 0),(160, 120), (0,0,0), -1)
			# put text in the box
			cv2.putText(heatmap,'Avg Temp: '+str(avgtemp)+' C', (10, 14),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Label Threshold: '+str(threshold)+' C', (10, 28),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Colormap: '+cmapText, (10, 42),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Blur: '+str(rad)+' ', (10, 56),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Scaling: '+str(scale)+' ', (10, 70),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Contrast: '+str(alpha)+' ', (10, 84),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Snapshot: '+snaptime+' ', (10, 98),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Recording: Not available', (10, 112),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200, 200, 200), 1, cv2.LINE_AA)
		
		#display floating max temp
		if maxtemp > avgtemp+threshold:
			cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,0), 2)
			cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,255), -1)
			cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
			cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		#display floating min temp
		if mintemp < avgtemp-threshold:
			cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (0,0,0), 2)
			cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (255,0,0), -1)
			cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
			cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		current_frame = heatmap

cap.release()
cv2.destroyAllWindows()