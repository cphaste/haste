-- import obj mesh type
dofile "obj.lua"

-- render configuration
render {
    size = {800, 600},
    max_bounces = 2,
    antialiasing = 2,
    direct_samples = 10
}

-- camera setup
camera {
	eye = {1.0, 0.5, 1.0},
	look = {0, 0.25, 0}
}

-- simple pointlight, using defaults
--[[lt = light {
    color = {0.1, 0.1, 0.1},
}]]--

-- sphere area light

arealt = sphere {
    position = {0, 4, 0},
    radius = 2.5,
    color = {1.0, 1.0, 1.0},
    emissive = 1.0,
    ambient = 0.0,
    diffuse = 0.2,
    specular = 0.0
}

--[[
light1 = light {
	position = { 0, 4, 0 }
}
arealt = sphere {
    position = {0, 4, -1},
    radius = 2.5,
    color = {1.0, 1.0, 1.0},
    emissive = 1.0,
    ambient = 0.0,
    diffuse = 0.2,
    specular = 0.0
}
]]--

-- scene floor
floor = plane {
    color = {0.8, 0.8, 0.8},
    distance = -0.5
}
--[[
tri1 = triangle{
	vertex1={0,0,0}
	normal1={0,0,0}
	vertex1={0,0,0}
	normal1={0,0,0}
	vertex1={0,0,0}
	normal1={0,0,0}

	color={1.0,1.0,1.0}
}

tri1 = triangle{
	vertex2={-2,0,0},
	normal2={0,1,0},

	vertex3={0,0,2},
	normal3={0,1,0},

	vertex1={2,0,0},
	normal1={0,1,0},

	color={1.0,1.0,1.0}
}

tri1 = triangle{
	vertex1={-2,0,0},
	normal1={0,1,0},

	vertex2={0,0,-2},
	normal2={0,1,0},

	vertex3={2,0,0},
	normal3={0,1,0},

	color={1.0,1.0,1.0}
}]]--
--[[
tri2 = triangle{
	vertex1={-2,0,2},
	normal1={-0.77,0.77,0},

	vertex2={2,0,-2},
	normal2={0.77,0.77,0},

	vertex3={2,0,2},
	normal3={0,1,0},

	color={1.0,1.0,1.0}
}
--[[]]--
-- bunny mesh
bunny = obj {
    mesh = "girl.obj",
    color = {1.0, 1.0, 1.0},
    specular = 0.0,
    shininess = 0.01
}

--[[ball = sphere {
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
}]]--
