-- import obj mesh type
dofile "obj.lua"

-- render configuration
render {
    size = {800, 600},
    max_bounces = 3,
    antialiasing = 1
}

-- camera setup
camera {
	eye = {0, 1, 2},
	look = {0, 0, 0}
}

-- simple pointlight, using defaults
lt = light {}

-- scene floor
floor = plane {
    color = {1.0, 1.0, 1.0},
    distance = -1
}

-- bunny mesh
bunny = obj {
    mesh = "bunny3.obj",
    color = {1.0, 1.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

--[[ball = sphere {
    position = {-3, 0, 0},
    radius = 1,
    color = {1.0, 0.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ball3 = sphere {
    position = {3, 0, 0},
    radius = 1,
    color = {0.0, 0.0, 1.0},
    specular = 0.4,
    shininess = 0.01
}]]--
