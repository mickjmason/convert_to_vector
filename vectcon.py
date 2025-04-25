## Much of this code was generated with ChatGPT but refined with hours of painful prompt battles
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from skimage import measure, morphology
import svgwrite

def quantize_image(img, n_colors=8):
    """Convert image to (w, h, 3) uint8, flatten, and KMeans quantize."""
    arr = np.array(img)
    w, h = arr.shape[1], arr.shape[0]
    flat = arr.reshape((-1, 3)).astype(float)
    kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
    labels = kmeans.fit_predict(flat)
    palette = kmeans.cluster_centers_.astype(np.uint8)
    quant = palette[labels].reshape((h, w, 3))
    return quant, labels.reshape((h, w)), palette

def extract_contours(label_img, color_idx):
    """Returns all region contours for a given label index."""
    mask = (label_img == color_idx)
    # Remove tiny holes
    mask = morphology.remove_small_holes(mask, area_threshold=30)
    mask = morphology.remove_small_objects(mask, min_size=30)
    contours = measure.find_contours(mask.astype(float), 0.5)
    # Convert (row, col) to (x, y)
    return [np.fliplr(cnt) for cnt in contours if len(cnt) > 10]

from shapely.geometry import LineString
def simplify_path_rdp(path, epsilon=1.5):
    path = np.asarray(path)
    if path.ndim > 2:
        path = path.reshape(-1, 2)
    if path.shape[0] < 3:
        return path
    return np.array(LineString(path).simplify(epsilon, preserve_topology=False))


def path_to_svgd(path):
    """Convert Nx2 array to SVG path string."""
    import numpy as np
    path = np.asarray(path)
    if path is None or path.size == 0:
        return ""
    # If path is 1D, can't be valid
    if path.ndim == 1:
        if path.shape[0] != 2:
            return ""
        # If it's a single point, cannot draw a region
        return ""
    # If path has more than 2 dimensions, flatten
    if path.ndim > 2:
        path = path.reshape(-1, 2)
    # Now, check if path is Nx2
    if path.shape[1] != 2 or path.shape[0] < 3:
        return ""
    d = f"M {path[0,0]:.2f},{path[0,1]:.2f} "
    for pt in path[1:]:
        d += f"L {pt[0]:.2f},{pt[1]:.2f} "
    d += "Z"
    return d

def parse_palette(palette_str):
    # Example: "255,0,0;0,255,0;0,0,255"
    palette = []
    for color in palette_str.split(";"):
        vals = [int(x) for x in color.split(",")]
        if len(vals) == 3:
            palette.append(tuple(vals))
        elif len(vals) == 4:
            # Ignore CMYK for now, only allow RGB for SVG
            # You could convert CMYK to RGB if you want
            c, m, y, k = [v / 255 for v in vals]
            # Simple conversion, not color-managed:
            r = 255 * (1.0 - c) * (1.0 - k)
            g = 255 * (1.0 - m) * (1.0 - k)
            b = 255 * (1.0 - y) * (1.0 - k)
            palette.append((int(r), int(g), int(b)))
    return np.array(palette, dtype=np.uint8)


def main(
    input_image='n.png',
    output_svg='output.svg',
    n_colors=8,
    gaussian_blur=2,
    simplify=1.5,
    min_area=40,
    palette=None
):
    print(f"{input_image} {output_svg} {n_colors} {gaussian_blur} {simplify} {min_area}")
    img = Image.open(input_image).convert('RGB')
    if gaussian_blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))
    if palette:
        # User-provided palette
        user_palette = parse_palette(palette)
        print("Using custom palette:", user_palette)
        arr = np.array(img)
        h, w = arr.shape[0], arr.shape[1]
        flat = arr.reshape((-1, 3)).astype(float)
        # Assign each pixel to the closest color in user_palette
        dists = np.sum((flat[:, None, :] - user_palette[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        label_img = labels.reshape((h, w))
        palette = user_palette
    else:
        quant, label_img, palette = quantize_image(img, n_colors=n_colors)
        h, w = label_img.shape
    dwg = svgwrite.Drawing(output_svg, size=(f"{w}px", f"{h}px"))
    for color_idx, rgb in enumerate(palette):
        mask = (label_img == color_idx)
        # Save mask for debugging
        Image.fromarray(mask.astype(np.uint8) * 255).save(f"mask_{color_idx}.png")
        contours = extract_contours(label_img, color_idx)
        color_hex = svgwrite.utils.rgb(rgb[0], rgb[1], rgb[2], mode='rgb')
        for cnt in contours:
            if len(cnt) < min_area: continue
            cnt_flipped = cnt.copy()
            #cnt_flipped[:,1] = h - cnt_flipped[:,1] - 1
            simp = simplify_path_rdp(cnt_flipped, epsilon=simplify)
            cnt_flat = np.asarray(cnt_flipped)
            pathstr = path_to_svgd(cnt_flat)
            if pathstr:
                dwg.add(dwg.path(d=pathstr, fill=color_hex, stroke='none', stroke_width=2, fill_rule='evenodd'))




    dwg.save()
    print(f"Saved SVG to {output_svg}")

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image (jpg/png)')
    parser.add_argument('output', help='Output SVG')
    parser.add_argument('--colors', type=int, default=8, help='Number of colors')
    parser.add_argument('--blur', type=float, default=2, help='Gaussian blur radius')
    parser.add_argument('--simplify', type=float, default=1.5, help='Simplify epsilon')
    parser.add_argument('--minarea', type=int, default=40, help='Min contour area (points)')
    parser.add_argument('--palette', type=str, default=None, help='Semicolon-separated list of RGB values, e.g. "255,0,0;0,255,0;0,0,255"')
    args = parser.parse_args()
    main(
        input_image=args.input,
        output_svg=args.output,
        n_colors=args.colors,
        gaussian_blur=args.blur,
        simplify=args.simplify,
        min_area=args.minarea,
        palette=args.palette
    )
