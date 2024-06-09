

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
        if (string[i] == 'W') and (string[i+1] != "O"):
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
            if string[i-1] != "Z":
                wall_edge_ratio_str = ''
                i += 1
                while i < len(string) and (string[i].isdigit() or string[i] == '.'):
                    wall_edge_ratio_str += string[i]
                    i += 1
                wall_edge_ratio = float(wall_edge_ratio_str)
            else:
                wall_edge_ratio_str = ''
                i += 1
                while i < len(string) and (string[i].isdigit() or string[i] == '.'):
                    wall_edge_ratio_str += string[i]
                    i += 1
                wall_edge_ratio = float(wall_edge_ratio_str)
        elif string[i:i+2] == "WO":
            i+=3
            if string[i-1] == "S":
                wall_orient = "same-side"
            elif string[i-1] == "R":
                wall_orient = "random"
            else:
                wall_orient = "alternate"


        else:
            i += 1

    return_tuple = width, height, depth, voxel_size, num_walls, room_seed, wall_edge_ratio

    if 'wall_orient' in locals():
        return_tuple = return_tuple + (wall_orient,)

    return return_tuple
