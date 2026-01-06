% scale(1000) import("hip_center.stl");

// Sketch hip_center_cylinder 135
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -83.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 135.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=60.000000,h=thickness);
  }
}
}
