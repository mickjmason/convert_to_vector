## Much of this code was generated with ChatGPT but refined with hours of painful prompt battles
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from skimage import measure, morphology
import svgwrite
from scipy.signal import savgol_filter

def savgol_smooth_path(path, window=5, polyorder=2):
    path = np.asarray(path)
    if path.shape[0] < window:
        return path
    x = savgol_filter(path[:, 0], window_length=window, polyorder=polyorder, mode='nearest')
    y = savgol_filter(path[:, 1], window_length=window, polyorder=polyorder, mode='nearest')
    return np.stack((x, y), axis=1)


def quantize_image(img, n_colors=8):
    arr = np.array(img)
    w, h = arr.shape[1], arr.shape[0]
    flat = arr.reshape((-1, 3)).astype(float)
    kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
    labels = kmeans.fit_predict(flat)
    palette = kmeans.cluster_centers_.astype(np.uint8)
    quant = palette[labels].reshape((h, w, 3))
    return quant, labels.reshape((h, w)), palette

def extract_contours(label_img, color_idx):
    mask = (label_img == color_idx)
    mask = morphology.remove_small_holes(mask, area_threshold=30)
    mask = morphology.remove_small_objects(mask, min_size=30)
    contours = measure.find_contours(mask.astype(float), 0.5)
    return [np.fliplr(cnt) for cnt in contours]

def is_path_closed(path, tol=1e-2):
    """Returns True if the first and last point are nearly the same."""
    return np.allclose(path[0], path[-1], atol=tol)

def smooth_path_closed(path, window=5):
    path = np.asarray(path)
    N = path.shape[0]
    if N < window:
        return path
    out = np.zeros_like(path)
    half = window // 2
    for i in range(N):
        idx = [(i + j) % N for j in range(-half, half + 1)]
        out[i] = np.mean(path[idx], axis=0)
    return out

def smooth_path_open(path, window=5):
    path = np.asarray(path)
    if path.shape[0] < window:
        return path
    kernel = np.ones(window) / window
    x = np.convolve(path[:, 0], kernel, mode='same')
    y = np.convolve(path[:, 1], kernel, mode='same')
    return np.stack((x, y), axis=1)

def smooth_path_open_no_artifacts(path, window=3):
    path = np.asarray(path)
    N = path.shape[0]
    if N < window:
        return path
    half = window // 2
    out = np.zeros_like(path)
    for i in range(N):
        # Start and end points average only over valid neighbors
        start = max(0, i - half)
        end = min(N, i + half + 1)
        out[i] = np.mean(path[start:end], axis=0)
    return out


from shapely.geometry import LineString
def simplify_path_rdp(path, epsilon=1.5):
    path = np.asarray(path)
    if path.ndim > 2:
        path = path.reshape(-1, 2)
    if path.shape[0] < 3:
        return path
    result = np.array(LineString(path).simplify(epsilon, preserve_topology=False))
    return result

def path_to_svgd(path):
    path = np.asarray(path)
    if path is None or path.size == 0:
        return ""
    if path.ndim == 1:
        if path.shape[0] != 2:
            return ""
        return ""
    if path.ndim > 2:
        path = path.reshape(-1, 2)
    if path.shape[1] != 2 or path.shape[0] < 3:
        return ""
    d = f"M {path[0,0]:.2f},{path[0,1]:.2f} "
    for pt in path[1:]:
        d += f"L {pt[0]:.2f},{pt[1]:.2f} "
    d += "Z"
    return d

def parse_palette(palette_str):
    palette = []
    for color in palette_str.split(";"):
        vals = [int(x) for x in color.split(",")]
        if len(vals) == 3:
            palette.append(tuple(vals))
        elif len(vals) == 4:
            c, m, y, k = [v / 255 for v in vals]
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
    palette=None,
    smoothing_window=7
):
    print(f"{input_image} {output_svg} {n_colors} {gaussian_blur} {simplify} {min_area}")
    img = Image.open(input_image).convert('RGB')
    if gaussian_blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))
    if palette:
        user_palette = parse_palette(palette)
        print("Using custom palette:", user_palette)
        arr = np.array(img)
        h, w = arr.shape[0], arr.shape[1]
        flat = arr.reshape((-1, 3)).astype(float)
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
        Image.fromarray(mask.astype(np.uint8) * 255).save(f"mask_{color_idx}.png")
        contours = extract_contours(label_img, color_idx)
        color_hex = svgwrite.utils.rgb(rgb[0], rgb[1], rgb[2], mode='rgb')
        print(f"Color {color_idx}: {len(contours)} contours")
        for i, cnt in enumerate(contours):
            cnt = np.asarray(cnt)
            if cnt.ndim == 3 and cnt.shape[2] == 2:
                cnt = cnt.reshape(-1, 2)
            if cnt.ndim != 2 or cnt.shape[1] != 2 or cnt.shape[0] < 3:
                continue
            if cnt.shape[0] < min_area:
                continue

            # --- Smoothing step with Savitzky-Golay ---
            if cnt.shape[0] < smoothing_window:
                smoothed = cnt
            else:
                smoothed = savgol_smooth_path(cnt, window=smoothing_window, polyorder=2)

            simp = simplify_path_rdp(smoothed, epsilon=simplify)
            simp = np.asarray(simp)
            if simp.ndim == 0 or simp.size == 0:
                simp = smoothed
            if simp.ndim == 3 and simp.shape[2] == 2:
                simp = simp.reshape(-1, 2)
            if simp.ndim != 2 or simp.shape[1] != 2 or simp.shape[0] < 3:
                continue
            pathstr = path_to_svgd(simp)
            if pathstr:
                dwg.add(dwg.path(d=pathstr, fill=color_hex, stroke='none', stroke_width=2, fill_rule='evenodd'))



    dwg.save()
    print(f"Saved SVG to {output_svg}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image (jpg/png)')
    parser.add_argument('output', help='Output SVG')
    parser.add_argument('--colors', type=int, default=8, help='Number of colors')
    parser.add_argument('--blur', type=float, default=2, help='Gaussian blur radius')
    parser.add_argument('--simplify', type=float, default=1.5, help='Simplify epsilon')
    parser.add_argument('--minarea', type=int, default=40, help='Min contour area (points)')
    parser.add_argument('--palette', type=str, default=None, help='Semicolon-separated list of RGB values, e.g. "255,0,0;0,255,0;0,0,255"')
    parser.add_argument('--smoothwin', type=int, default=7, help='Smoothing window size (odd integer, default 7)')
    args = parser.parse_args()
    main(
        input_image=args.input,
        output_svg=args.output,
        n_colors=args.colors,
        gaussian_blur=args.blur,
        simplify=args.simplify,
        min_area=args.minarea,
        palette=args.palette,
        smoothing_window=args.smoothwin
    )
