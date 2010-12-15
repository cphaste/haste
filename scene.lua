-- import obj mesh type
require "helpers/obj.lua"

-- render configuration
render {
    size = {640, 480},
    max_bounces = 2,
    antialiasing = 1,
    direct_samples = 32,
    gamma_correction = 1 / 2.2
}

-- camera setup
camera {
	eye = {0, 3, 10},
	look = {0, 0, 0}
}

-- sphere area light
arealt = sphere {
    position = {0, 4, -6},
    radius = 2.5,
    color = {1, 1, 1},
    emissive = 1,
    ambient = 0,
    diffuse = 0,
    specular = 0
}

-- scene floor
floor = plane {
    color = {0.8, 0.8, 0.8},
    distance = -0.5
}

-- colored balls
ballR = sphere {
    position = {-3, 0, 0},
    radius = 1,
    color = {1.0, 0.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ballG = sphere {
    position = {0, 0, 2},
    radius= 1,
    color = {0.0, 1.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ballB = sphere {
    position = {3, 0, 0},
    radius = 1,
    color = {0.0, 0.0, 1.0},
    specular = 0.4,
    shininess = 0.01
}
