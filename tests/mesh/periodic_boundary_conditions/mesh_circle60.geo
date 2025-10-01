lc = 3.5e-2;
ratio = 0.875;
radius_inner = 0.5;
radius_outer = 1.0;
radius_holes = 0.06;

// Sector angle in radians
theta = 2* Pi / 5;

// Outer arc endpoint coordinates
x_outer = radius_outer * Cos(theta);
y_outer = radius_outer * Sin(theta);

// Inner arc endpoint coordinates
x_inner = radius_inner * Cos(theta);
y_inner = radius_inner * Sin(theta);

// Points
Point(1) = {0, 0, 0, lc};                            // Center
Point(2) = {radius_inner, 0, 0, lc*ratio};           // Inner start
Point(3) = {radius_outer, 0, 0, lc*ratio};           // Outer start
Point(4) = {x_outer, y_outer, 0, lc};                // Outer end
Point(5) = {x_inner, y_inner, 0, lc};                // Inner end

// Edges of the sector
Line(1) = {2, 3};             // Radial line (inner to outer)
Circle(2) = {3, 1, 4};        // Outer arc
Line(3) = {4, 5};             // Radial line (outer to inner)
Circle(4) = {5, 1, 2};        // Inner arc

// Outer boundary loop
Curve Loop(1) = {1, 2, 3, 4};

// Hole points
Point(6) = {radius_inner + 0.2, 0.2, 0, lc};                         // Circle center
Point(7) = {radius_inner + 0.2 + radius_holes, 0.2, 0, lc};          // 0 degrees
Point(8) = {radius_inner + 0.2 - radius_holes, 0.2, 0, lc};          // 180 degrees

Circle(5) = {7, 6, 8}; // First half of the circle
Circle(6) = {8, 6, 7}; // Second half of the circle

Curve Loop(2) = {5, 6}; // Complete hole loop

// Surface with hole
Plane Surface(1) = {1, 2};

// Physical groups
Physical Surface(1) = {1};
Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};
Physical Curve(5) = {5, 6};