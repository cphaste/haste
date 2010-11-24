function obj(mesh)
	local verts = {}
	local norms = {}
	local line_num = 1
	local processed = false
	local center = {0, 0, 0}
	
	if mesh.position == nil then
		mesh.position = {0, 0, 0}
	end
	
	for line in io.lines(mesh.mesh) do
		if string.match(line, "^%s*$") ~= nil or string.match(line, "^%s*#") ~= nil then
			-- empty or comment line, ignore
		else
			-- vertex line
			local x, y, z = string.match(line, "^%s*v%s+([%d%-%.]+)%s+([%d%-%.]+)%s+([%d%-%.]+)")
			if x ~= nil then
				verts[#verts + 1] = {x, y, z}
				center = {center[1] + x,
				          center[2] + y,
				          center[3] + z}
			else
				-- vertex normal line
				local nx, ny, nz = string.match(line, "^%s*vn%s+([%d%-%.]+)%s+([%d%-%.]+)%s+([%d%-%.]+)")
				if nx ~= nil then
					norms[#norms + 1] = {nx, ny, nz}
				else
					-- face line
					local v1, n1, v2, n2, v3, n3 = string.match(line, "^%s*f%s+(%d+)%s*/%s*/%s*(%d+)%s+(%d+)%s*/%s*/%s*(%d+)%s+(%d+)%s*/%s*/%s*(%d+)")
					if v1 ~= nil then
					    -- convert to numbers
					    v1 = tonumber(v1)
					    v2 = tonumber(v2)
					    v3 = tonumber(v3)
					    n1 = tonumber(n1)
					    n2 = tonumber(n2)
					    n3 = tonumber(n3)
					
						-- process the mesh after all the verts and norms have been loaded
						if not processed then
							-- compute the mesh's center
							center = {center[1] / #verts,
							          center[2] / #verts,
							          center[3] / #verts}
							
							-- get the maximum distance from the center to the furthest vertex
							local max_dist = 0
							for i, v in ipairs(verts) do
								local dx = v[1] - center[1]
								local dy = v[2] - center[2]
								local dz = v[3] - center[3]
								local dist = math.sqrt(dx * dx + dy * dy + dz * dz)
								if dist > max_dist then
									max_dist = dist
								end
							end
							
							-- translate the verts to the center and scale it to fit inside a sphere of radius 1
							for i, v in ipairs(verts) do
							    verts[i] = {(v[1] - center[1] + mesh.position[1]) / max_dist,
							                (v[2] - center[2] + mesh.position[2]) / max_dist,
							                (v[3] - center[3] + mesh.position[3]) / max_dist}
							end
							
							processed = true
						end
					
						-- create the triangle
						local tri = {
							vertex1 = verts[v1],
							vertex2 = verts[v2],
							vertex3 = verts[v3],
							normal1 = norms[n1],
							normal2 = norms[n2],
							normal3 = norms[n3]
						}
						
						--print("===== TRI =====")
					    --print("\tv1, v2, v3 = " .. verts[v1] .. ", " .. verts[v2] .. ", " .. verts[v3])
						
						-- add surface properties
						for k, v in pairs(mesh) do
							tri[k] = v
						end
						
						-- insert the triangle
						triangle(tri)
					else
						print("Error on line " .. line_num .. " of " .. mesh_file .. "!")
					end
				end
			end
		end
		
		line_num = line_num + 1
	end
end
