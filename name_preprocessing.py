def extract_info(string):
    # Replace dashes with dots
    string = string.replace('-', '.')

    # Initialize variables
    width = None
    height = None
    depth = None
    voxel_size = None
    num_walls = None
    room_seed = None
    wall_edge_ratio = None

    # Iterate over the string and extract information
    i = 0
    while i < len(string):
        if string[i] == 'W':
            width_str = ''
            i += 1
            while i < len(string) and string[i].isdigit():
                width_str += string[i]
                i += 1
            width = int(width_str)
        elif string[i] == 'H':
            height_str = ''
            i += 1
            while i < len(string) and string[i].isdigit():
                height_str += string[i]
                i += 1
            height = int(height_str)
        elif string[i] == 'D':
            depth_str = ''
            i += 1
            while i < len(string) and string[i].isdigit():
                depth_str += string[i]
                i += 1
            depth = int(depth_str)
        elif string[i:i+2] == 'VS':
            voxel_size_str = ''
            i += 2
            while i < len(string) and (string[i].isdigit() or string[i] == '.'):
                voxel_size_str += string[i]
                i += 1
            voxel_size = float(voxel_size_str)
        elif string[i:i+2] == 'NW':
            num_walls_str = ''
            i += 2
            while i < len(string) and string[i].isdigit():
                num_walls_str += string[i]
                i += 1
            num_walls = int(num_walls_str)
        elif string[i] == 'S':
            room_seed_str = ''
            i += 1
            while i < len(string) and string[i].isdigit():
                room_seed_str += string[i]
                i += 1
            room_seed = int(room_seed_str)
        elif string[i] == 'R':
            wall_edge_ratio_str = ''
            i += 1
            while i < len(string) and (string[i].isdigit() or string[i] == '.'):
                wall_edge_ratio_str += string[i]
                i += 1
            wall_edge_ratio = float(wall_edge_ratio_str)
        else:
            i += 1

    return width, height, depth, voxel_size, num_walls, room_seed, wall_edge_ratio
