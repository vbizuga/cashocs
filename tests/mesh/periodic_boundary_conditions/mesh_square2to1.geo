lc = 10e-2;
ratio = 2;
radius_holes = 0.06;

// Points
Point(1) = {0, 0, 0, lc};
Point(2) = {0, 1, 0, lc*ratio};
Point(3) = {1, 1, 0, lc*ratio};
Point(4) = {1, 0, 0, lc};

// Edges of the sector
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Outer boundary loop
Curve Loop(1) = {1, 2, 3, 4};

// Hole points
Point(6) = {0.5, 0.2, 0, lc};
Point(7) = {0.5 + radius_holes, 0.2, 0, lc};
Point(8) = {0.5 - radius_holes, 0.2, 0, lc};

Circle(5) = {7, 6, 8};
Circle(6) = {8, 6, 7};

Curve Loop(2) = {5, 6};

// Surface with hole
Plane Surface(1) = {1, 2};

// Physical groups
Physical Surface(1) = {1};
Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};
Physical Curve(5) = {5, 6};