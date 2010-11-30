-- import obj mesh type
dofile "obj.lua"

-- render configuration
render {
    size = {800, 600},
    max_bounces = 3,
    antialiasing = 1,
    direct_samples = 10
}

-- camera setup
camera {
	eye = {0, 2, 8},
	look = {0, 0, 0}
}

-- simple pointlight, using defaults
--[[lt = light {
    color = {0.1, 0.1, 0.1},
}]]--

-- sphere area light
arealt = sphere {
    position = {0, 2, -2},
    radius = 2,
    color = {1.0, 1.0, 1.0},
    emissive = 0.8,
    ambient = 0.0,
    diffuse = 0.2,
    specular = 0.0
}

-- scene floor
floor = plane {
    color = {0.8, 0.8, 0.8},
    distance = -1
}

-- bunny mesh
--[[bunny = obj {
    mesh = "bunny3.obj",
    color = {1.0, 1.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}]]--

ball = sphere {
    position = {-3, 0, 0},
    radius = 1,
    color = {1.0, 0.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ball2 = sphere {
    position = {0, 0, 2},
    radius= 1,
    color = {0.0, 1.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ball3 = sphere {
    position = {3, 0, 0},
    radius = 1,
    color = {0.0, 0.0, 1.0},
    specular = 0.4,
    shininess = 0.01
}
