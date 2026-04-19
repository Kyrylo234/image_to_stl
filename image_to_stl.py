import gradio as gr
from PIL import Image, ImageFilter
import numpy as np
import struct
import trimesh
import tempfile

# ----------------------- TRIANGLE GENERATOR -----------------------
def generate_triangles_vectorized(data, scale):
    h, w = data.shape

    h1 = data[:-1, :-1] * scale
    h2 = data[:-1, 1:]  * scale
    h3 = data[1:, :-1]  * scale
    h4 = data[1:, 1:]   * scale

    y = np.arange(h - 1, dtype=np.float32) * scale
    x = np.arange(w - 1, dtype=np.float32) * scale
    X0, Y0 = np.meshgrid(x, y)

    X1 = X0 + scale
    Y1 = Y0 + scale

    return X0, Y0, X1, Y1, h1, h2, h3, h4


# ----------------------- STL STREAM WRITER -----------------------
def stream_stl(triangle_arrays, outfile, transparent_mask=None):
    X0, Y0, X1, Y1, h1, h2, h3, h4 = triangle_arrays
    h_blocks, w_blocks = h1.shape

    triangles = []

    def V(x, y, z):
        return np.array([float(x), float(y), float(z)], dtype=np.float32)

    for y in range(h_blocks):
        for x in range(w_blocks):

            skip = transparent_mask is not None and transparent_mask[y, x]
            if skip:
                continue

            v1 = V(X0[y, x], Y0[y, x], h1[y, x])
            v2 = V(X1[y, x], Y0[y, x], h2[y, x])
            v3 = V(X0[y, x], Y1[y, x], h3[y, x])
            v4 = V(X1[y, x], Y1[y, x], h4[y, x])

            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])

            if x == 0 or (transparent_mask is not None and transparent_mask[y, x-1]):
                b1 = V(v1[0], v1[1], 0)
                b3 = V(v3[0], v3[1], 0)
                triangles.append([b1, v1, b3])
                triangles.append([v1, v3, b3])

            if x == w_blocks - 1 or (transparent_mask is not None and transparent_mask[y, x+1]):
                b2 = V(v2[0], v2[1], 0)
                b4 = V(v4[0], v4[1], 0)
                triangles.append([b2, b4, v2])
                triangles.append([v2, b4, v4])

            if y == 0 or (transparent_mask is not None and transparent_mask[y-1, x]):
                b1 = V(v1[0], v1[1], 0)
                b2 = V(v2[0], v2[1], 0)
                triangles.append([b1, b2, v1])
                triangles.append([v1, b2, v2])

            if y == h_blocks - 1 or (transparent_mask is not None and transparent_mask[y+1, x]):
                b3 = V(v3[0], v3[1], 0)
                b4 = V(v4[0], v4[1], 0)
                triangles.append([b3, v3, b4])
                triangles.append([v3, v4, b4])

    for y in range(h_blocks):
        for x in range(w_blocks):

            if transparent_mask is not None and transparent_mask[y, x]:
                continue

            b1 = V(X0[y, x], Y0[y, x], 0)
            b2 = V(X1[y, x], Y0[y, x], 0)
            b3 = V(X0[y, x], Y1[y, x], 0)
            b4 = V(X1[y, x], Y1[y, x], 0)

            triangles.append([b1, b3, b2])
            triangles.append([b2, b3, b4])

    with open(outfile, "wb") as f:
        f.write(b"SolidHeightmap".ljust(80, b" "))
        f.write(struct.pack("<I", len(triangles)))

        for tri in triangles:
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in tri:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


# ----------------------- IMAGE → STL -----------------------
def image_to_stl(img, max_height, blur_radius, mode, transparency_threshold):
    alpha_mask = None
    if img.mode in ("RGBA", "LA"):
        alpha_channel = np.array(img.split()[-1])
        alpha_mask = alpha_channel < transparency_threshold

    if mode == "Binary":
        img_gray = img.convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        img_gray = img.convert("L")

    img_array = np.array(img_gray)

    if alpha_mask is not None:
        img_array[alpha_mask] = 255

    masked_img = Image.fromarray(img_array)
    img_blur = masked_img.filter(ImageFilter.GaussianBlur(blur_radius))

    data = (1 - (np.array(img_blur) / 255.0)) * max_height
    data = np.flipud(data)

    triangles = generate_triangles_vectorized(data, 0.01)

    stl_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
    glb_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    stl_path, glb_path = stl_tmp.name, glb_tmp.name
    stl_tmp.close()
    glb_tmp.close()

    mask_flipped = None
    if alpha_mask is not None:
        mask_flipped = np.flipud(alpha_mask)[:-1, :-1]

    stream_stl(triangles, stl_path, transparent_mask=mask_flipped)

    mesh = trimesh.load(stl_path)
    mesh.export(glb_path)

    return glb_path, stl_path


# ----------------------- UI -----------------------
with gr.Blocks(title="Image to STL Generator") as iface:
    gr.Markdown("## Image to STL Generator")

    image_input = gr.Image(
        type="pil",
        label="Upload Image",
        interactive=True,
        image_mode="RGBA"
    )

    gr.Markdown("<small>💡 High-contrast images produce the cleanest 3D surfaces.</small>")

    max_height = gr.Slider(
        0, 50, value=10,
        label="Max Height (mm)",
        info="Larger values produce steeper surfaces."
    )

    blur_radius = gr.Slider(
        0, 10, value=3, step=1,
        label="Blur Radius (px) (0 = no blur, 10 = strong blur)",
        info="Recommended: 3. Smooths the slopes."
    )

    mode = gr.Radio(
        ["Grayscale", "Binary"],
        value="Grayscale",
        label="Image Mode",
        info="Binary mode = image converted into either black or white pixels. Grayscale = height based on pixel colour."
    )

    transparency_threshold = gr.Slider(
        1, 255, value=255, step=1,
        label="Transparency Threshold",
        info="Pixels with alpha below this value will be removed from the model."
    )

    gr.Markdown("""
    **Tips:**
    - Use **Binary mode** for clean shapes (logos, icons).
    - Use **Grayscale mode** for detailed images (faces, landscapes).
    - Adjust **Transparency Threshold** to control which pixels are included.
    - A small **blur** helps remove noisy edges.
    """)

    generate_btn = gr.Button("Generate STL")

    model_output = gr.Model3D(label="3D Preview (GLB)")
    stl_output = gr.File(label="Download STL")

    generate_btn.click(
        fn=image_to_stl,
        inputs=[image_input, max_height, blur_radius, mode, transparency_threshold],
        outputs=[model_output, stl_output]
    )

iface.launch(server_name="0.0.0.0", prevent_thread_lock=False)