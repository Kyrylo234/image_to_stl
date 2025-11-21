import sys
from PIL import Image, ImageFilter
import numpy as np
from stl import mesh

def main():
    if len(sys.argv) < 4:
        print("Usage: python image_to_stl.py <input_image> <output_stl> <max_height>")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_height = float(sys.argv[3])
    blur_radius = int(sys.argv[4])


    img = Image.open(input_file)
    # Fix transparent PNGs (flatten to white)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha = img.convert("RGBA")
        background = Image.new("RGBA", alpha.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, alpha).convert("RGB")

    # Convert to grayscale, then apply blur
    img = img.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Without gausian blur
    #img = Image.open(input_file).convert('L')


    
    data = (1.0 - (np.array(img) / 255.0)) * max_height

    data = np.flipud(data)

    height, width = data.shape
    num_triangles = (width - 1) * (height - 1) * 2

    # Prepare STL mesh
    stl_mesh = mesh.Mesh(np.zeros(num_triangles, dtype=mesh.Mesh.dtype))

    

    tri_index = 0
    unit_scale = 0.01  # Force STL output in millimeters

    for y in range(height - 1):
        for x in range(width - 1):

            h1 = data[y, x]
            h2 = data[y, x+1]
            h3 = data[y+1, x]
            h4 = data[y+1, x+1]

            # XY before scaling
            X0 = x
            X1 = x + 1
            Y0 = y
            Y1 = y + 1

            # Apply uniform scale (mm output)
            X0 *= unit_scale
            X1 *= unit_scale
            Y0 *= unit_scale
            Y1 *= unit_scale

            h1 *= unit_scale
            h2 *= unit_scale
            h3 *= unit_scale
            h4 *= unit_scale

            # Triangle 1
            stl_mesh.vectors[tri_index] = np.array([
                [X0, Y0, h1],
                [X1, Y0, h2],
                [X0, Y1, h3]
            ])
            tri_index += 1

            # Triangle 2
            stl_mesh.vectors[tri_index] = np.array([
                [X1, Y0, h2],
                [X1, Y1, h4],
                [X0, Y1, h3]
            ])
            tri_index += 1

    # Save STL
    stl_mesh.save(output_file)
    print(f"STL file saved: {output_file}")

if __name__ == "__main__":
    main()