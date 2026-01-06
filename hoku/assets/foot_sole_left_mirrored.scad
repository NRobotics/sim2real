% scale(1000) import("foot_sole_left_mirrored.stl");

// Sketch foot_sole_left_mirrored_cylinders 165
multmatrix([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 62.67703768532171], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 165.000000;
translate([0, 0, -thickness]) {
  translate([-48.837965, 780.663971, 0]) {
    cylinder(r=7.500000,h=thickness);
  }
}
}

// Sketch foot_sole_left_mirrored_cylinders 225
multmatrix([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 102.67703768532171], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 225.000000;
translate([0, 0, -thickness]) {
  translate([-80.837965, 780.663971, 0]) {
    cylinder(r=7.500000,h=thickness);
  }
  translate([-64.837965, 780.663971, 0]) {
    cylinder(r=7.500000,h=thickness);
  }
}
}

// Sketch foot_sole_left_mirrored_cylinders 125
multmatrix([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 37.6770376853217], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 125.000000;
translate([0, 0, -thickness]) {
  translate([-96.837965, 780.663971, 0]) {
    cylinder(r=7.500000,h=thickness);
  }
}
}
