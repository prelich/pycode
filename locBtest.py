# locBtest.py
# script to test loading ONI locB file and visualizing
# Author: PKR, UPenn, July 2018
# Adapted from Vispy Fireworks demo

from loadLocB import loadLocB

import sys
import numpy as np
from vispy import gloo, app
import copy

VERT_SHADER = """
uniform float u_scale;
uniform vec3 u_centerPosition;
attribute vec3 a_pos;
void main () {
    gl_Position.xyz = u_scale*(a_pos - u_centerPosition);
    gl_Position.w = 1.0;
    gl_PointSize = pow(u_scale,0.334)*2;
}
"""

FRAG_SHADER = """
precision highp float;
uniform sampler2D texture1;
uniform vec4 u_color;
uniform highp sampler2D s_texture;
void main () {
    highp vec4 texColor;
    texColor = texture2D(s_texture, gl_PointCoord);
    gl_FragColor = vec4(u_color) * texColor;
}
"""

# Define the File Location
#FileLoc = 'E:/Data/rdemo/gridcheck.locb'
FileLoc = '//entropy.lgads.lakadamyali.group/Data3/peter/Melina/MelinagfptauA3_gfpnanogerm_1hrDMSOtreatedSTORM.1543951287600.locb'

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))

        # initialize texture and particle count
        self.create_texture()
        self.load_Coordinates(FileLoc)

        # Create Program
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._program.bind(gloo.VertexBuffer(self.data))
        self._program['s_texture'] = gloo.Texture2D(self.im1)

        # draw the points
        self._set_points()

        # enable blending
        gloo.set_state(blend=True, clear_color='black',
                       blend_func=('src_alpha', 'one'))

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])

        self._timer = app.Timer('auto', connect=self.update, start=True)

        self.show()

    def create_texture(self, radius=32):
        self.im1 = np.zeros([radius * 2 + 1, radius * 2 + 1])
        for index,array in np.ndenumerate(self.im1):
            pixelValue = 1 - np.sqrt( (radius-index[0])**2 +
                       (radius-index[1])**2)/ radius
            self.im1[index] = max(pixelValue,0)
            self.im1 = self.im1.astype(np.float32)

    def load_Coordinates(self, FileLoc):
        DictOut = loadLocB(FileLoc)

        self.N = DictOut['Localization_Channels'].size
        self.data = np.zeros(self.N, [('a_pos', np.float32, 3)])
        # load raw coordinates first
        keys = ['rawPosition_x','rawPosition_y','rawPosition_z']
        rawPosCol = [DictOut['Localization_Matrix_Mapping'].get(key) for key in keys]
        rawLocs = DictOut['Localization_Results'][:,rawPosCol]
        # apply drift correction
        # perform a deep copy operation here
        corLocs = copy.deepcopy(rawLocs)
        # figure out frame pointer
        offsets = DictOut['Drift_Offsets']
        frameLoc = DictOut['Localization_Frames'].astype(np.int)
        frameOffset = DictOut['Drift_Frames'].astype(np.int)
        
        for index,element in enumerate(frameLoc):
            corLocs[index,0:2] += offsets[element,:]
        self.corLocIn = corLocs
        boundMax = np.max(corLocs,0)
        boundMin = np.min(corLocs,0)
        stretchLocs = corLocs - boundMin
        stretchLocs /= (boundMax - boundMin)/2
        stretchLocs -= 1
        # take out Z for now
        stretchLocs[:,2] = 0
        # stretch coordinates so they fit in the scene
        self.data['a_pos'] = stretchLocs
        #print(self.data)

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):

        # Clear
        gloo.clear()

        # Draw
        self._program.draw('points')

    def _set_points(self):

        # New centerpos
        self.centerpos = [0,0,0]
        self._program['u_centerPosition'] = self.centerpos

        self.scale = 1.0
        self._program['u_scale'] = self.scale

        # New color, scale alpha with N
        alpha = 1.0 / self.N ** 0.08
        #color = np.random.uniform(0.1, 0.9, (3,))
        color = np.array([0,1,0])

        self._program['u_color'] = tuple(color) + (alpha,)

    def on_mouse_move(self, event):
        """Pan the view based on the change in mouse position."""
        if event.is_dragging and event.buttons[0] == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            X0, Y0 = self.pixel_to_coords(float(x0), float(y0))
            X1, Y1 = self.pixel_to_coords(float(x1), float(y1))
            self.translate_center(X1 - X0, Y1 - Y0)

    def translate_center(self, dx, dy):
        """Translates the center point, and keeps it in bounds."""
        center = self.centerpos
        center[0] -= dx
        center[1] -= dy
        self._program['u_centerPosition'] = center
        self.centerpos = center

    def pixel_to_coords(self, x, y):
        """Convert pixel coordinates to set coordinates."""
        rx, ry = self.size
        #nx = (x / rx - 0.5) * self.scale + self.centerpos[0]
        #ny = ((ry - y) / ry - 0.5) * self.scale + self.centerpos[1]
        nx = (x / rx - 0.5) / self.scale + self.centerpos[0]
        ny = ((ry - y) / ry - 0.5) / self.scale + self.centerpos[1]
        #print(nx, ny)
        #print(self.scale)
        #print(self.centerpos)
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

        The mouse wheel can be used to zoom, but some people don't have mouse
        wheels :)

        """
        if event.text == '-':
            self.zoom(0.9)
        elif event.text == '+' or event.text == '=':
            self.zoom(1/0.9)

    def zoom(self, factor, mouse_coords=None):
        """Factors less than zero zoom in, and greater than zero zoom out.

        If mouse_coords is given, the point under the mouse stays stationary
        while zooming. mouse_coords should come from MouseEvent.pos.

        """
        if mouse_coords is not None:  # Record the position of the mouse
            x, y = float(mouse_coords[0]), float(mouse_coords[1])
            x0, y0 = self.pixel_to_coords(x, y)

        self.scale *= factor
        self._program['u_scale'] = self.scale

        if mouse_coords is not None:  # Translate so the mouse point is stationary
            x1, y1 = self.pixel_to_coords(x, y)
            self.translate_center(x1 - x0, y1 - y0)
            #self.translate_center(x0-x1, y0-y1)

    def on_timer(self, event):
        self.update()

if __name__ == '__main__':
    c = Canvas()
    app.run()

