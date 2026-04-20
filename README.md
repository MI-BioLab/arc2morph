# Arc2Morph: Identity-Preserving Facial Morphing with Arc2Face

This is the official repository for the paper "Arc2Morph: Identity-Preserving Facial Morphing with Arc2Face".

# How to use
1. Install [Arc2Face](https://github.com/foivospar/Arc2Face) with ControlNet.
2. Place the [morph_multiple.py](morph_multiple.py) script in the root directory of the Arc2Face repository.
3. Run it. The script expects a list of paths relative to a root, as in the example below.

```
relative/path/1.jpg relative/path/2.jpg
...
relative/path/m.jpg relative/path/n.jpg
```

Run it in the Arc2Face environment as:
```bash
python morph_multiple.py --root <your_images_root> --pairs <list_of_pairs.txt> --output <output_directory>
```

By default the chosen interpolation function is slerp applied to CLIP-encoded identities; if you want to override this and use the other interpolation modes, you can set the `--interp_mode` flag accordingly.
