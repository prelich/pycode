# binLoad.py
# function to load .bin files from Insight 3

import numpy as np
import struct

def binLoad(FileLoc):
    with open(FileLoc, "rb") as binary_file:
        binary_file.seek(0)
        first_word = binary_file.read(4).decode('utf-8')
        nFrames = int.from_bytes(binary_file.read(4),byteorder='little')
        junk_value = int.from_bytes(binary_file.read(4),byteorder='little')
        nMolecules = int.from_bytes(binary_file.read(4),byteorder='little')
        LocalizationResults = np.zeros([nMolecules,18])
        for ii in range(0,nMolecules):
            LocalizationResults[ii,:] = struct.unpack('<'+'f'*11+'I'*5+'f'*2,
                                        binary_file.read(72))
        FrameResults = np.zeros(nFrames)
        for ii in range(0,nFrames):
            FrameResults[ii] = int.from_bytes(binary_file.read(4),byteorder='little')

    LocNames = dict([("X",0),("Y",1),("Xc",2),("Yc",3),("Height",4),("Area",5),
                    ("Width",6),("Phi",7),("Aspect",8),("Bg",9),("I",10),
                    ("Channel",11),("FitIter",12),("Frame",13),("TrackL",14),
                    ("Link",15),("Z",16),("Zc",17)])

    OutDictionary = dict([("Localizations",LocalizationResults), ("Frames",FrameResults),("Localization_Matrix_Mapping",LocNames)])
    return OutDictionary
