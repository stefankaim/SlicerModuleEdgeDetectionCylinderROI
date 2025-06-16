import os
import re
import csv
import slicer
import qt
import vtk, ctk
import numpy as np
import locale
from slicer.ScriptedLoadableModule import *

class AutoCylinderModule(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Cylinder by Edge-Detection"
        parent.categories = ["ROIs"]
        parent.dependencies = []
        parent.contributors = ["Stefan Kaim"]
        parent.helpText = "Detects cylinder centerpoints from two markup planes and generates a 3D cylinder."
        parent.acknowledgementText = "TU Wien / OpenAI Assist"

class AutoCylinderModuleWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        layout = qt.QFormLayout()
        layout.addRow(qt.QLabel("Cylinder Detection"))

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("Select CT-Volume")
        self.volumeSelector.currentNodeChanged.connect(self.disableGenerate)
        layout.addRow("CT Volume:", self.volumeSelector)

        self.plane2Selector = slicer.qMRMLNodeComboBox()
        self.plane2Selector.nodeTypes = ["vtkMRMLMarkupsPlaneNode"]
        self.plane2Selector.setMRMLScene(slicer.mrmlScene)
        self.plane2Selector.setToolTip("Select top plane")
        self.plane2Selector.currentNodeChanged.connect(self.disableGenerate)
        layout.addRow("Top Plane:", self.plane2Selector)

        self.plane1Selector = slicer.qMRMLNodeComboBox()
        self.plane1Selector.nodeTypes = ["vtkMRMLMarkupsPlaneNode"]
        self.plane1Selector.setMRMLScene(slicer.mrmlScene)
        self.plane1Selector.setToolTip("Select bottom plane")
        self.plane1Selector.currentNodeChanged.connect(self.disableGenerate)
        layout.addRow("Bottom Plane:", self.plane1Selector)

        self.detectButton = qt.QPushButton("Detect Center Points and Radius")
        self.detectButton.clicked.connect(self.detectCentersAndRadius)
        layout.addRow(self.detectButton)

        line = qt.QFrame()
        line.setFrameShape(qt.QFrame.HLine)
        line.setFrameShadow(qt.QFrame.Sunken)
        layout.addRow(line)
        layout.addRow(qt.QLabel(""))
        layout.addRow(qt.QLabel(""))
        layout.addRow(qt.QLabel("Generate Cylinder"))

        self.nameInput = qt.QLineEdit()
        self.nameInput.setPlaceholderText("PLACEHOLDER")
        layout.addRow("Name:", self.nameInput)

        self.radiusSpinBox = qt.QDoubleSpinBox()
        self.radiusSpinBox.setMinimum(0.1)
        self.radiusSpinBox.setMaximum(100.0)
        self.radiusSpinBox.setSingleStep(0.1)
        self.radiusSpinBox.setValue(5.0)
        layout.addRow("Radius [mm] (editable):", self.radiusSpinBox)

        self.heightDisplay = qt.QLineEdit()
        self.heightDisplay.setReadOnly(True)
        layout.addRow("Height [mm]:", self.heightDisplay)

        self.topCoord = qt.QLineEdit()
        self.topCoord.setReadOnly(True)
        layout.addRow("Center-Top (X,Y,Z):", self.topCoord)

        self.botCoord = qt.QLineEdit()
        self.botCoord.setReadOnly(True)
        layout.addRow("Center-Bottom (X,Y,Z):", self.botCoord)

        self.generateButton = qt.QPushButton("Generate Cylinder")
        self.generateButton.clicked.connect(self.generateCylinder)
        self.generateButton.setEnabled(False)
        layout.addRow(self.generateButton)

        line = qt.QFrame()
        line.setFrameShape(qt.QFrame.HLine)
        line.setFrameShadow(qt.QFrame.Sunken)
        layout.addRow(line)
        layout.addRow(qt.QLabel(""))
        layout.addRow(qt.QLabel(""))
        layout.addRow(qt.QLabel("Export Cylinders"))

        # Volume Selector
        self.volumeSelectorExport = slicer.qMRMLNodeComboBox()
        self.volumeSelectorExport.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelectorExport.selectNodeUponCreation = True
        self.volumeSelectorExport.setMRMLScene(slicer.mrmlScene)

        # Segment Lists
        self.availableSegmentsList = qt.QListWidget()
        self.selectedSegmentsList = qt.QListWidget()
        self.addSegmentButton = qt.QPushButton("→")
        self.removeSegmentButton = qt.QPushButton("←")
        self.addSegmentButton.clicked.connect(self.addSelectedSegments)
        self.removeSegmentButton.clicked.connect(self.removeSelectedSegments)

        # Lists Button Layout
        buttonLayout = qt.QVBoxLayout()
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.addSegmentButton)
        buttonLayout.addWidget(self.removeSegmentButton)
        buttonLayout.addStretch(1)

        segmentSelectionLayout = qt.QHBoxLayout()
        segmentSelectionLayout.addWidget(self.availableSegmentsList)
        segmentSelectionLayout.addLayout(buttonLayout)
        segmentSelectionLayout.addWidget(self.selectedSegmentsList)

        # Export Path
        self.outputPathButton = ctk.ctkDirectoryButton()
        self.outputPathButton.directory = qt.QDir.homePath()

        self.exportButton = qt.QPushButton("Export")
        self.exportButton.clicked.connect(self.exportCSV)

        self.statusLabel = qt.QLabel("")

        layout.addRow("CT Volume (Export):", self.volumeSelectorExport)
        layout.addRow("Segments:", segmentSelectionLayout)
        layout.addRow("Export Folder:", self.outputPathButton)
        layout.addRow(self.exportButton)
        layout.addRow(self.statusLabel)

        self.layout.addLayout(layout)
        self.layout.addStretch(1)

        self.basePoint = None
        self.topPoint = None

    def extractCenterFromPlane(self, volumeNode, planeNode):
        ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRAS)
        rasToIJK = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ijkToRAS, rasToIJK)

        centerRAS = planeNode.GetOrigin()
        size = planeNode.GetSize()
        spacing = volumeNode.GetSpacing()
        width = int(size[0] / spacing[0] / 2)
        height = int(size[1] / spacing[1] / 2)

        centerIJK = [0, 0, 0, 1.0]
        rasToIJK.MultiplyPoint(list(centerRAS) + [1.0], centerIJK)
        centerIJK = np.array(centerIJK[:3])
        z = int(round(centerIJK[2]))

        array = slicer.util.arrayFromVolume(volumeNode)
        if z < 0 or z >= array.shape[0]:
            return None, None

        slice2D = array[z]
        x0 = int(round(centerIJK[0] - width))
        x1 = int(round(centerIJK[0] + width))
        y0 = int(round(centerIJK[1] - height))
        y1 = int(round(centerIJK[1] + height))
        roi = slice2D[y0:y1, x0:x1]

        if roi.size == 0 or np.max(roi) == np.min(roi):
            return None, None

        center_x = int(round(centerIJK[0])) - x0
        center_y = int(round(centerIJK[1])) - y0
        if center_x < 0 or center_x >= roi.shape[1] or center_y < 0 or center_y >= roi.shape[0]:
            return None, None

        hu_center = roi[center_y, center_x]
        lower = hu_center * 0.9
        upper = hu_center * 1.1
        mask = (roi >= lower) & (roi <= upper)
        if not np.any(mask):
            return None, None

        y_coords, x_coords = np.nonzero(mask)
        cx = np.mean(x_coords) + x0
        cy = np.mean(y_coords) + y0
        radius = np.median(np.sqrt((x_coords - np.mean(x_coords)) ** 2 + (y_coords - np.mean(y_coords)) ** 2))
        radius_mm = radius * spacing[0]

        centerIJK = np.array([cx, cy, z, 1.0])
        centerRAS = [0.0, 0.0, 0.0, 1.0]
        ijkToRAS.MultiplyPoint(centerIJK, centerRAS)

        # DEBUG, creates MarkupPoints in the found center of the planes
        #name = self.nameInput.text.strip()
        #markupsLogic = slicer.modules.markups.logic()
        #listNode = slicer.util.getNode(f"{name}_CenterPoints") if slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode") else slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{name}_CenterPoints")
        #listNode.AddFiducialFromArray(centerRAS[:3], planeNode.GetName())

        return centerRAS[:3], radius_mm

    def detectCentersAndRadius(self):
        volumeNode = self.volumeSelector.currentNode()
        plane1 = self.plane1Selector.currentNode()
        plane2 = self.plane2Selector.currentNode()
        if not (volumeNode and plane1 and plane2):
            slicer.util.errorDisplay("Please select a volume and two plane markups.")
            return

        base, r1 = self.extractCenterFromPlane(volumeNode, plane1)
        top, r2 = self.extractCenterFromPlane(volumeNode, plane2)
        if base is None or top is None:
            slicer.util.errorDisplay("Could not detect valid content in ROI.")
            return

        self.basePoint = base
        self.topPoint = top
        avgRadius = (r1 + r2) / 2.0
        self.radiusSpinBox.setValue(avgRadius)
        height = np.linalg.norm(np.array(top) - np.array(base))
        self.heightDisplay.setText(f"{height:.2f}")
        self.topCoord.setText(f"{top[0]:.2f}, {top[1]:.2f}, {top[2]:.2f}")
        self.botCoord.setText(f"{base[0]:.2f}, {base[1]:.2f}, {base[2]:.2f}")

        #slicer.util.infoDisplay(f"Center base: {base}, top: {top}, radius: {avgRadius:.2f}, height: {height:.2f}")
        slicer.util.infoDisplay(f"Center found. Cylinder can be created")

        self.generateButton.setEnabled(True)

    def generateCylinder(self):
        if not (self.basePoint and self.topPoint):
            slicer.util.errorDisplay("No center-points found. Detection must be run first.")
            return
        
        if not (self.nameInput.text):
            slicer.util.errorDisplay("No name set for Cylinder. Please set a name.")
            return
        
        name = self.nameInput.text.strip()

        p1 = np.array(self.basePoint)
        p2 = np.array(self.topPoint)
        axis = p2 - p1
        height = np.linalg.norm(axis)
        direction = axis / height if height != 0 else np.array([0, 0, 1])

        center = (p1 + p2) / 2

        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(self.radiusSpinBox.value)
        cylinder.SetHeight(height)
        cylinder.SetResolution(200)
        cylinder.Update()

        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.RotateX(90)

        baseDirection = [0, 0, 1]
        rotationAxis = np.cross(baseDirection, direction)
        angle = np.degrees(np.arccos(np.clip(np.dot(baseDirection, direction), -1.0, 1.0)))

        if np.linalg.norm(rotationAxis) > 0.01:
            transform.RotateWXYZ(angle, *rotationAxis)

        transform.Translate(*center)

        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{name}_Transform")
        transformNode.SetMatrixTransformToParent(transform.GetMatrix())

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', f"{name}")
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.SetAndObserveTransformNodeID(transformNode.GetID())
        segmentation = segmentationNode.GetSegmentation()

        segment = slicer.vtkSegment()
        segment.SetName(name)
        segment.AddRepresentation(
            slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName(),
            cylinder.GetOutput()
        )
        segmentation.AddSegment(segment)
        segmentationNode.Modified()

        self.nameInput.text = ""
        self.heightDisplay.text = ""
        self.topCoord.text = ""
        self.botCoord.text = ""
        self.generateButton.setEnabled(False)

        self.updateAvailableSegments()

        slicer.util.infoDisplay(f"Cylinder {name} Generated.")

    def disableGenerate(self):
        self.generateButton.setEnabled(False)

    def updateSegmentDropdown(self):
        self.segmentDropdown.clear()
        segmentationNode = self.segmentationSelector.currentNode()
        if segmentationNode:
            segmentation = segmentationNode.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                name = segmentation.GetNthSegment(i).GetName()
                self.segmentDropdown.addItem(name)
    
    @staticmethod
    def cleanName(name):
        return re.sub(r'[<>:"/\\|?*\']','', name)
    
    def updateAvailableSegments(self):
        self.availableSegmentsList.clear()
        self.selectedSegmentsList.clear()
        
        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for segNode in segmentationNodes:
            segmentation = segNode.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                segName = segmentation.GetNthSegment(i).GetName()
                listEntry = f"{segNode.GetName()}::{segName}"
                self.availableSegmentsList.addItem(listEntry)
    
    def addSelectedSegments(self):
        for item in self.availableSegmentsList.selectedItems():
            self.selectedSegmentsList.addItem(item.text())
            self.availableSegmentsList.takeItem(self.availableSegmentsList.row(item))
    
    def removeSelectedSegments(self):
        for item in self.selectedSegmentsList.selectedItems():
            self.availableSegmentsList.addItem(item.text())
            self.selectedSegmentsList.takeItem(self.selectedSegmentsList.row(item))

    def exportCSV(self):
        volumeNode = self.volumeSelectorExport.currentNode()
        selectedSegments = [self.selectedSegmentsList.item(i).text() for i in range(self.selectedSegmentsList.count)]
        outputFolder = self.outputPathButton.directory
        
        if not volumeNode:
            self.statusLabel.setText("A CT-Volume needs to be selected!")
            return
        
        if not selectedSegments:
            self.statusLabel.setText("A Segment needs to be selected!")
            return
        
        exportCount = 0

        for fullName in selectedSegments:
            if "::" not in fullName:
                continue

            segNodeName, segmentName = fullName.split("::", 1)

            try:
                segmentationNode = slicer.util.getNode(segNodeName)
            except slicer.util.MRMLNodeNotFoundException:
                self.statusLabel.setText(f"Segment {segNodeName} not found! Continue using next segment.")
                continue

            segmentID = None
            for segIndex in range(segmentationNode.GetSegmentation().GetNumberOfSegments()):
                currentID = segmentationNode.GetSegmentation().GetNthSegmentID(segIndex)
                currentSegment = segmentationNode.GetSegmentation().GetSegment(currentID)
                if currentSegment.GetName() == segmentName:
                    segmentID = currentID
                    break

            if not segmentID:
                self.statusLabel.setText(f"Segment '{segmentName}' not found.")
                continue

            labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Temp_Labelmap")
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, [segmentID], labelmapNode, volumeNode)

            ctArray = slicer.util.arrayFromVolume(volumeNode)
            labelArray = slicer.util.arrayFromVolume(labelmapNode)

            ijkToRAS = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRAS)
            z_ras_coords = [ijkToRAS.MultiplyPoint([0, 0, z, 1.0])[2] for z in range(ctArray.shape[0])]
            
            safeVolumeName = self.cleanName(volumeNode.GetName())
            safeSegNodeName = self.cleanName(segNodeName)

            locale.setlocale(locale.LC_ALL, '')

            decimal_point = locale.localeconv()["decimal_point"]
            csv_delimiter = ';' if decimal_point == ',' else ','

            sliceResults = []
            for z in range(ctArray.shape[0]):
                sliceCT = ctArray[z]
                sliceMask = labelArray[z] > 0
                huValues = sliceCT[sliceMask]
                if huValues.size > 0:
                    def fmt(x): return locale.format_string("%.9f", x, grouping=True)
                    sliceResults.append({
                        "SliceIndex": str(z),
                        "Z_Slice_mm": fmt(z_ras_coords[z]),
                        "Mean": fmt(np.mean(huValues)),
                        "StdDev": fmt(np.std(huValues)),
                        "Min": fmt(np.min(huValues)),
                        "Max": fmt(np.max(huValues)),
                        "VoxelCount": str(huValues.size),
                        "StdErr": fmt(np.std(huValues) / np.sqrt(huValues.size))
                    })

            outputPath = os.path.join(outputFolder, f"{safeVolumeName}_{safeSegNodeName}_statistics.csv")
            with open(outputPath, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=["SliceIndex", "Z_Slice_mm", "Mean", "StdDev", "Min", "Max", "VoxelCount", "StdErr"], delimiter=csv_delimiter)
                writer.writeheader()
                for row in sliceResults:
                    writer.writerow(row)
            
            slicer.mrmlScene.RemoveNode(labelmapNode)
            exportCount += 1

        self.statusLabel.setText(f"Exported {exportCount} segment(s) to :\n{outputFolder}")
