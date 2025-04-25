# Bitmap to SVG Converter

This is a high-quality, customizable **bitmap (JPG/PNG) to SVG vector converter** script written in Python by ChatGpt, with lots and lots (and lots) of input from me.

It combines:
- Image quantization using k-means clustering,
- Morphological cleaning to remove noise,
- Smooth, artifact-free contour tracing,
- Savitzky-Golay smoothing,
- RDP (Douglas-Peucker) simplification,
- Full support for user-specified color palettes,
- Easy CLI (command line) usage.

---

## Features

- **Converts raster images (JPG, PNG) into vector SVGs with smooth, clean outlines.**
- Choose the number of colors, blurring amount, contour simplification, and even the exact colors.
- Advanced smoothing to reduce jagged edges, with user control over window size.
- Outputs SVGs that are easy to edit in vector tools (Illustrator, Inkscape, Figma, etc.).
- Open source and fully scriptable.

---

## Installation

Install Python 3.7+ if you don't have it already.

Install required Python libraries:

```bash
pip install numpy pillow scikit-image svgwrite scikit-learn shapely scipy
```
---

## Usage
```bash 
python bitmap2svg.py input.png output.svg [--colors N] [--blur RADIUS] [--simplify EPSILON] [--minarea MIN] [--palette PALETTE] [--smoothwin WIN]
```

### Example
```bash
python bitmap2svg.py myimage.jpg out.svg --colors 6 --blur 2 --simplify 1.2 --smoothwin 5
```

---

## Parameters

| Parameter      | Type    | Default | Description |
|----------------|---------|---------|-------------|
| `input`        | string  | *required* | Path to input image (PNG or JPG) |
| `output`       | string  | *required* | Path to output SVG file |
| `--colors`     | int     | 8       | Number of colors to use for SVG palette. Fewer colors = more stylized. |
| `--blur`       | float   | 2       | Amount of Gaussian blur before vectorizing. Higher values reduce noise but also detail. |
| `--simplify`   | float   | 1.5     | RDP simplification epsilon. Higher = smoother, fewer nodes. Lower = more detail. |
| `--minarea`    | int     | 40      | Minimum number of points for a region to be kept. Increases this to skip tiny shapes. |
| `--palette`    | string  | None    | Semicolon-separated RGB list, e.g. `"255,0,0;0,255,0;0,0,255"`. If not given, palette is auto-chosen by k-means. |
| `--smoothwin`  | int     | 7       | Smoothing window size for Savitzky-Golay smoothing. Odd integer, e.g. 3, 5, 7. Higher values = smoother, but too high can cause distortion. |

---

### Palette Example

To force output colors, use:
```bash
python bitmap2svg.py input.png output.svg --palette "0,0,0;255,255,255;255,0,0"
```

This restricts the SVG to only black, white and red.

---

## How It Works

1. **Preprocess**: Optionally blur the image to reduce noise.
2. **Quantization**: Convert the image to `N` colors using k-means clustering or a user-provided palette.
3. **Contour Extraction**: For each color, find all continuous color regions using skimage.
4. **Smoothing**: Optionally apply Savitzky-Golay smoothing to each shape’s path.
5. **Simplification**: Apply RDP simplification for edit-friendly SVG curves.
6. **SVG Generation**: Write each smoothed, simplified region as a `<path>` in the SVG.

---

## Tips

- Use lower `--colors` and higher `--simplify` for more abstract, “posterized” looks.
- Use lower `--simplify` and a small `--smoothwin` for highly accurate tracing.
- Set a custom `--palette` to lock output to brand or design colors.
- For best results, clean your input image (remove transparency, crop whitespace).

---

## Troubleshooting

- **Artifacts or stray lines:** Lower the smoothing window or increase `--minarea`.
- **Too blocky or too many points:** Raise or lower `--simplify`.
- **Not enough detail:** Lower `--simplify` and/or lower `--blur`.

---

Created with the help of [ChatGPT](https://chat.openai.com/) and hours of experimentation.
