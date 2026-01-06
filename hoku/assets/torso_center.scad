% scale(1000) import("torso_center.stl");

// Sketch torso_center_cylinder 300
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -37.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 300.000000;
translate([0, 0, -thickness]) {
  translate([-37.500000, -28.884482, 0]) {
    cylinder(r=78.986270,h=thickness);
  }
}
}
