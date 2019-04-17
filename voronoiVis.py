# voronoiVis.py
# script to generate a tetrahedra/cone for 2D Voronoi visualization
# Author PKR, UPENN, MARCH 2019

# add loading paths
import sys, os
sys.path.append(os.path.join(sys.path[0],'ONI'))
sys.path.append(os.path.join(sys.path[0],'Legacy'))

import numpy as np
from vispy import app, gloo
from binLoad import binLoad
from loadLocB import loadLocB
from vispy.util.transforms import translate, perspective, rotate
# using cython code here
from polyArea import PolyArea

from scipy import spatial as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

## SET FILE LOCATION HERE
#FileLoc = '//entropy.lgads.lakadamyali.group/Data3/peter/Melina/MelinagfptauA3_gfpnanogerm_1hrDMSOtreatedSTORM.1543953593043.locb'
FileLoc = '//entropy.lgads.lakadamyali.group/Data3/peter/hMSC-MeHA-Glass.bin'
OutDict = binLoad(FileLoc)
#OutDict = loadLocB(FileLoc)
# Pull out X Y coordinates
Pos = OutDict['Localizations'][:,0:2]
#keys = ['rawPosition_x','rawPosition_y'] #,'rawPosition_z']
#rawPosCol = [OutDict['Localization_Matrix_Mapping'].get(key) for key in keys]
#Pos = OutDict['Localization_Results'][:,rawPosCol]

maxes = np.amax(Pos,axis=0)
mines = np.amin(Pos,axis=0)
width = np.max(maxes-mines)
pointz = 2*((Pos-mines)/width)-1
# flip y-axis to match insight3 convention
pointz[:,1] = -pointz[:,1]
#pointz = 2*(np.random.rand(1000000,2)-0.5)
uPos = np.unique(pointz,axis=0) # make sure points are unique
pointz = uPos
nPoints = uPos.shape[0]
Area = PolyArea(uPos*width/2)

# find min/max values for normalization
minima = min(-np.log(Area))
maxima = max(-np.log(Area))
print(minima)
print(maxima)

## SET PARAMETERS HERE:
pix2nm = 116
barWidth = 2000
maxima = 7
minima = 3

## Creating a modified jet color map here
jjet = cm.get_cmap('jet',240)
ojet = jjet(np.linspace(0,1,240))
fadeLen = 16
djet = np.zeros([fadeLen,4])
# create an nx4 array of dark colors
for ii in range(0,fadeLen):
    djet[ii,:] = (ii/fadeLen)*ojet[0,:]
    djet[ii,3] = 1
newjet = np.vstack((djet,ojet))
njetcmp = ListedColormap(newjet)

# normalize chosen colormap
norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=njetcmp)

color = np.zeros((nPoints,4))
color = mapper.to_rgba(-np.log(Area))

# VBO Code for Voronoi Display
# note the 'color' and 'v_color' in vertex
vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
uniform vec4   u_color;         // mask color for edge plotting
attribute vec3 a_position;
attribute vec4 a_color;
varying vec4   v_color;
void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    v_color = a_color * u_color;
}
"""
# note the varying 'v_color', it must has the same name as in the vertex.
fragment = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""

# helper functions for generating polygons, scale bar
def genVIC(points,tri=32,color=None):
    N = points.shape[0]
    V = np.zeros((N*(tri+1),3),dtype=np.float32)
    I = np.zeros((N*(tri),3),dtype=np.uint32)
    C = np.ones((N*(tri+1),4),dtype=np.float32)
    npi = np.pi
    V[::tri+1,0:2] = points
    starts = np.arange(0,N*(tri+1),tri+1)
    ends = np.arange(tri,N*(tri+1),tri+1)
    I[::tri,:] = np.stack((starts,ends,starts+1),axis=-1)
    for ii in range(0,tri):
        adjust = [np.cos(2*npi*ii/(tri-1))/64,np.sin(2*npi*ii/(tri-1))/64]
        V[ii+1::tri+1,0:2] = points+adjust
        V[ii+1::tri+1,2] = -0.1
    for ii in range(0,tri-1):
        I[ii+1::tri,:] = np.stack((starts,starts+ii+1,starts+ii+2),axis=-1)
    # we'll put color logic in later, do random for now
    if color is None:
        color = np.random.rand(N,4)
        color[:,3] = 1
    C[::tri+1,:] = color
    for ii in range(0,tri):
        C[ii+1::tri+1,:] = C[::tri+1,:]
    return V, I, C

def genScaleBar(dim, center):
    # bottom right, bottom left, top left, top right
    rect = {0:[1,-1],1:[-1,-1],2:[-1,1],3:[1,1]}
    N = 4
    V = np.zeros((N,3),dtype=np.float32)
    I = np.zeros((N-2,3),dtype=np.uint32)
    C = np.ones((N,4),dtype=np.float32)
    for ii in range(0,N):
        V[ii,0] = center[0]+rect[ii][0]*dim[0]/2
        V[ii,1] = center[1]+rect[ii][1]*dim[1]/2
    V[:,2] = 0.002 # raise scale bar above voronoi cones
    I[0,:] = [0, 1, 2]
    I[1,:] = [2, 3, 0]
    return V,I,C

class Canvas(app.Canvas):
    """ build canvas class for this demo """

    def __init__(self):
        """ initialize the canvas """
        app.Canvas.__init__(self,
                            size=(512, 512),
                            title='SR Voronoi Visualizer',
                            keys='interactive')        
        self.tri=16 # 16 edges for each point
        self.shrinkage = width/2
        # define vertices, indices, color
        V,I,C = genVIC(pointz,self.tri,color)
        self.BarStart = V.shape[0]
        # set initial scale and center point
        self.centerpos = [0,0]
        self.scale = 1
        # hard-coded bar coordinates in the mean time
        self.BarCenter = [0.9, -0.9]
        self.BarDim = [barWidth/pix2nm/self.shrinkage, 0.05/np.sqrt(self.scale)]
        bV,bI,bC = genScaleBar(self.BarDim,self.BarCenter)
        bI = bI+self.BarStart
        # bind to data
        V = np.vstack((V,bV))
        I = np.vstack((I,bI))
        C = np.vstack((C,bC))
        # shader program
        tet = gloo.Program(vert=vertex, frag=fragment)#, count=V.shape[0])
        self.I = gloo.IndexBuffer(I)
        self.V = gloo.VertexBuffer(V)
        self.C = gloo.VertexBuffer(C)
        tet['a_position'] = self.V
        tet['a_color'] = self.C
        # intialize transformation matrix
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        # set view
        self.view = translate((0, 0, -3))
        tet['u_model'] = self.model
        tet['u_view'] = self.view
        tet['u_projection'] = self.projection
        # bind program
        self.program = tet
        # config and set viewport
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_clear_color('black')
        gloo.set_state('opaque')
        gloo.set_polygon_offset(1.0, 1.0)
        # bind a timer
        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()
        # show the canvas
        self.show()

    def update_scalebar(self):
        self.BarCenter = [0.9, -0.9]
        self.BarDim = [barWidth/pix2nm/self.shrinkage, 0.05/np.sqrt(self.scale)]
        invMat = np.linalg.inv(self.model)
        newCenter = [self.BarCenter[0],self.BarCenter[1],0.002,1]
        newCenter = invMat@newCenter
        self.BarCenter = [newCenter[0]-self.centerpos[0],
                          newCenter[1]-self.centerpos[1]]
        bV,bI,bC = genScaleBar(self.BarDim,self.BarCenter)
        self.V.set_subdata(bV,self.BarStart)

    def on_resize(self, event):
        """ canvas resize callback """
        ratio = event.physical_size[0] / float(event.physical_size[1])
        self.projection = perspective(45.0, ratio, 2.0, 10.0)
        self.program['u_projection'] = self.projection
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        """ canvas update callback """
        gloo.clear()
        gloo.set_state('opaque', blend=True, depth_test=True,
                       polygon_offset_fill=True)
        self.program['u_color'] = [1.0, 1.0, 1.0, 1.0]
        self.program.draw('triangles', self.I)

    def on_mouse_move(self, event):
        """Pan the view based on the change in mouse position."""
        if event.is_dragging and event.buttons[0] == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            X0, Y0 = self.pixel_to_coords(float(x0), float(y0))
            X1, Y1 = self.pixel_to_coords(float(x1), float(y1))
            center = self.centerpos
            center[0] += X1-X0
            center[1] += Y1-Y0
            model = translate((center[0]*self.scale, center[1]*self.scale, 0))
            model[0,0] *= self.scale
            model[1,1] *= self.scale
            self.model = model
            self.centerpos = center
            self.program['u_model'] = self.model

    def pixel_to_coords(self, x, y):
        """Convert pixel coordinates to set coordinates."""
        rx, ry = self.size
        nx = (x / rx - 0.5) / self.scale + self.centerpos[0]
        ny = ((ry - y) / ry - 0.5) / self.scale + self.centerpos[1]
        return [nx, ny]

    def on_mouse_wheel(self, event):
        """Use the mouse wheel to zoom."""
        delta = event.delta[1]
        if delta > 0:  # Zoom in
            factor = 1 / 0.9
        elif delta < 0:  # Zoom out
            factor = 0.9
        for _ in range(int(abs(delta))):
            self.zoom(factor, event.pos)

    def on_key_press(self, event):
        """Use + or - to zoom in and out.

#        The mouse wheel can be used to zoom, but some people don't have mouse
#        wheels :)

#        """
        if event.text == '-':
            self.zoom(0.9)
        elif event.text == '+' or event.text == '=':
            self.zoom(1/0.9)
        elif event.text == 'b':
            self.update_scalebar()

    def zoom(self, factor, mouse_coords=None):
        """Factors less than zero zoom in, and greater than zero zoom out.

        If mouse_coords is given, the point under the mouse stays stationary
        while zooming. mouse_coords should come from MouseEvent.pos.

        """
        if mouse_coords is not None:  # Record the position of the mouse
            x, y = float(mouse_coords[0]), float(mouse_coords[1])
            x0, y0 = self.pixel_to_coords(x, y)
        self.scale *= factor
        model = translate((self.centerpos[0]*self.scale, self.centerpos[1]*self.scale, 0))
        model[0,0] *= self.scale
        model[1,1] *= self.scale
        self.model = model
        self.program['u_model'] = self.model

    def on_timer(self, event):
        """ canvas time-out callback """
        self.update()

# Finally, we show the canvas and we run the application.
c = Canvas()
app.run()
