# SlicerModuleEdgeDetectionCylinderROI
Creates Cylinders in 3D Slicer based on 2 Markup-Planes. The center of the cylinders is calculated by using a "fake"-edge detection. The HU value of the center of the plane is taken and every pixel inside with +/-10% is seen as "inside-pixel", then the centroid of those pixels is calculated to set the center of one plane.
