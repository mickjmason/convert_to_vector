# Bitmap to SVG Converter (`bitmap2svg.py`)

Convert bitmap images (JPG, PNG, etc.) to smooth, colored vector SVGs using KMeans color clustering or your own custom color palette.

---

## ✨ Features

- **Automatic Color Reduction:** Use KMeans to reduce your image to any number of colors for clean SVGs.
- **Custom Palette Support:** Specify exact RGB or CMYK colors to use in the SVG output.
- **Contiguous Vector Regions:** Produces contiguous vector areas (not pixelated), great for editing in vector programs.
- **Noise Reduction:** Removes tiny objects and holes for smooth, clean results.
- **Path Simplification:** Optional smoothing of shapes with the Ramer–Douglas–Peucker algorithm.
- **SVG Output:** Outputs standard, scalable SVG with each region as a path and correct color fill.
- **Debugging:** Saves a black-and-white mask PNG for each color to help you debug or inspect region extraction.

---

## 🛠 Installation

Make sure you have Python 3.7 or higher.

Install dependencies using pip:

```bash
pip install pillow numpy scikit-image scikit-learn svgwrite shapely
```

---

## 🚀 Usage

### Basic usage (auto color quantization):

```bash
python bitmap2svg.py input.jpg output.svg --colors 8
```
Converts `input.jpg` to an SVG with 8 dominant colors.

### Specify your own color palette (RGB):

```bash
python bitmap2svg.py input.jpg output.svg --palette "255,0,0;0,255,0;0,0,255;255,255,0;0,0,0"
```
This uses red, green, blue, yellow, and black as the only colors in the output.

### Advanced options:

- `--blur N` — Apply Gaussian blur before quantizing (default: `2`)
- `--simplify N` — Simplify region borders (higher = simpler, default: `1.5`)
- `--minarea N` — Ignore regions with fewer than N points (default: `40`)

**Example:**

```bash
python bitmap2svg.py input.png output.svg --colors 5 --blur 1 --simplify 2 --minarea 30
```

---

## 🖼 How It Works

1. **Reads your image** (any format supported by PIL).
2. **Optionally blurs** to reduce noise.
3. **Reduces to a set of color regions** with either KMeans or your custom palette.
4. **Finds all contiguous areas** of each color using image processing.
5. **Converts those areas to SVG paths,** with optional smoothing.
6. **Outputs a standard SVG** file you can use in Inkscape, Illustrator, etc.

---

## ⚡ Example Palettes

- 3-color grayscale: `"0,0,0;128,128,128;255,255,255"`
- 5-color CMYK-style: `"0,255,255,0;255,0,255,0;255,255,0,0;0,0,0,255;255,255,255,0"`

---

## 📝 Notes

- For CMYK values, they are auto-converted to RGB using a simple formula. For best results, provide RGB.
- The script saves a `mask_N.png` for each color (N) for debugging (delete these if not needed).
- If your output SVG appears upside down, uncomment the Y-flip line in the code.

---

## 🛟 Troubleshooting

- If your SVG is all one color: Make sure you specified the correct number of colors/palette entries.
- If you get errors about `path.shape[1]`, your environment may have an old NumPy—update to the latest version.
- For jagged or noisy regions, increase `--blur` or `--simplify` (or both).
- For more details per region, decrease `--simplify` and `--minarea`.

---


## 🤝 Credits

Created by a collaborative workflow using OpenAI ChatGPT and iterative user feedback.

---

**Happy vectorizing!**
