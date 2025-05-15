def generate_rgb_palette(num_objects):
    # Define the palette
    palette = [33554431, 32767, 2097151]
    
    # Initialize the list to store RGB tuples
    rgb_list = []
    
    # Generate RGB tuples for each object
    for i in range(num_objects):
        # Calculate the RGB values using the given formula
        r = (i * palette[0]) % 255
        g = (i * palette[1]) % 255
        b = (i * palette[2]) % 255
        
        # Append the RGB tuple to the list
        rgb_list.append((r, g, b))
    
    return rgb_list

