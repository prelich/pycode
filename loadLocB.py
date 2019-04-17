# loadLocB.py
# brief: function for loading locB files generated by the NanoImager (ONI)
# Author: Peter K. Relich, UPenn, July 2018

import numpy as np
import struct

def loadLocB(FileLoc):
    with open(FileLoc, "rb") as binary_file:
        # read entire file
        #data = binary_file.read()

        # move the pointer to the correct location in the array
        def set_offset(N,offset=0):
           offset += N
           binary_file.seek(offset)
           return offset

        # start at beginning of data and read out bytes
        offset = set_offset(0)
        header = binary_file.read(8).decode('utf-8')
        version = int.from_bytes(binary_file.read(4),byteorder='little')
        DriftFlag = int.from_bytes(binary_file.read(1),byteorder='little')
        NumDrift = int.from_bytes(binary_file.read(8),byteorder='little')
        offset = set_offset(21,offset)

        # populate drift fields if they exist
        if DriftFlag:
            DriftFrame = np.zeros(NumDrift)
            CoordOffset = np.zeros([NumDrift,2])
            for ii in range(0,NumDrift):
                DriftFrame[ii] = int.from_bytes(binary_file.read(4),byteorder='little')
                CoordOffset[ii,:] = struct.unpack('ff',binary_file.read(8))
            offset = set_offset(12*NumDrift,offset)
        else:
            NumDrift = []
            DriftFrame = []
            CoordOffset = []

        # populate localization results
        NumLoc = int.from_bytes(binary_file.read(8),byteorder='little')
        offset = set_offset(8,offset)
        FrameLocIndex = np.zeros(NumLoc)
        ChannelIndex = np.zeros(NumLoc)
        LocalizationResults = np.zeros([NumLoc,18])
        for ii in range(0,NumLoc):
            FrameLocIndex[ii] = int.from_bytes(binary_file.read(4),byteorder='little')
            ChannelIndex[ii] = int.from_bytes(binary_file.read(1),byteorder='little')
            LocalizationResults[ii,:] = struct.unpack('<'+'f'*7+'?'+'f'*8+'I'*2,binary_file.read(69))
        offset = set_offset(74*NumLoc,offset)

        # populate Acquisition knowledge
        NumAcquisitions = int.from_bytes(binary_file.read(4),byteorder='little')
        offset = set_offset(4,offset)
        offset = set_offset(4,offset) # have to do this twice here!!!
        FrameAcqIndex = np.zeros(NumAcquisitions)
        AcquisitionData = np.zeros([NumAcquisitions,10])
        for ii in range(0,NumAcquisitions):
            FrameAcqIndex[ii] = int.from_bytes(binary_file.read(4),byteorder='little')
            AcquisitionData[ii,:] = struct.unpack('<I?I'+'d'*5+'?d',binary_file.read(58))

        # Dict Pointers for Localization Results and Acquisition Data
        LocNames = dict([("rawPosition_x",0),("rawPosition_y",1),("rawPosition_z",2),
                     ("sigma_x",3),("sigma_y",4),("intensity",5),("background",6),
                     ("is_valid",7),("CRLB_xPosition",8),("CRLB_yPosition",9),
                     ("CRLB_intensity",10),("CRLB_background",11),("CRLB_sigma_x",12),
                     ("CRLB_sigma_y",13),("logLikelihood",14),("logLikelihoodRatio_PValue",15),
                     ("spotDetectionPixelPos_x",16),("spotDetectionPixelPos_y",17)])

        AcqNames = dict([("versionNumber",0),("hasCameraFrameIndex",1),("frameIndex",2),
                     ("stagePositionInMicrons_x",3),("stagePositionInMicrons_y",4),
                     ("stagePositionInMicrons_z",5),("illuminationAngleInDegrees",6),
                     ("temperatureInCelsius",7),("outOfRangeAccelerationDetected",8),
                     ("zFocusOffset",9)])

        # put it all in a dictionary and export!
        outDictionary = dict([("header",header),("version",version),("Drift_On",DriftFlag),
                          ("Drift_Frames",DriftFrame),("Drift_Offsets",CoordOffset),
                          ("Num_Localizations",NumLoc),("Localization_Frames",FrameLocIndex),
                          ("Localization_Channels",ChannelIndex),
                          ("Localization_Results",LocalizationResults),
                          ("Num_Acquisitions",NumAcquisitions),
                          ("Acquisition_Frames",FrameAcqIndex),("Acquisition_Data",AcquisitionData),
                          ("Localization_Matrix_Mapping",LocNames),
                          ("Acquisition_Matrix_Mapping",AcqNames)])

        return outDictionary