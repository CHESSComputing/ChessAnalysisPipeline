# Common subpackage (CHAP.common)

The common subpackage contains modules that are common to various workflows. These common tools can be broadly divided into three groups: readers, writers, and processors. This document describes the various tools and how they can be integrated as parts of specific data analysis workflows.

## Usage of the common readers, processors and writers

## Common readers

## Common writers

## Common processors

### ImageProcessor

The `ImageProcessor` processor can be used to perform various visualization operations on images (slices) selected from a NeXus object. It can plot and/or return image slices or create animations for a set of image slices from a NeXus object with a default plottable data path. The NeXus object has to be passed to the processor as a pipeline item, either directly from another processor or as the output from a NeXus file reader.

The `ImageProcessor` accepts two additional optional input parameters, the processor's configuration (`config`) dictionary and the `save_figures` parameter, as well as the generic config parameters of the workflow `pipeline.yaml` file.

The `ImageProcessor` configuration (`config`) dictionary accepts the following parameters:

- `animation` (`bool`, optional): Create an animation for an image stack (ignored for a single image), defaults to `False`
- `axis` (`int`, `str`, optional): Axis direction or name for the image slice(s), defaults to `0`.
- `coord_range` (`float`, `list[float]`, optional): Coordinate value range of the selected image slice(s), up to three floating point numbers (start, end, step), defaults to `None`, which enables index_range to select the image slice(s). Include only `coord_range` or `index_range`, not both.
- `index_range` (`int`, `list[int]`, optional): Array index range of the selected image slice(s), up to three integers (start, end, step). Set `index_range` to -1 to select the center image slice of an image stack in the `axis` direction. Only used when coord_range = `None`. Defaults to `None`, which will include all slices.
- fileformat (`gif`, `jpeg`, `png`, `tif`, optional): Image (stack) return file type, defaults to `png` for a single image, `tif` for an image stack, or `gif` for an animation.
- vrange (`list[float, float]`: Data value range in image slice(s), defaults to `None`, which uses the full data value range in the slice(s).


The `save_figures` parameter instructs the processor to return the plottable image(s) or animation to be written to file downstream in the pipeline (defaults to `True`). It is ignored and set to `True` by default when the `interactive` parameter in the generic config block of the workflow `pipeline.yaml` file is set to `False`.

#### Examples

1. Interactively plotting center image slice in the `x` direction of the default plottable data object in the `combined_hollow_pyramid.nxs` file and writing the result as a `jpeg` image to `image.jpeg`:
   ```bash
   config:
     interactive: true
     log_level: INFO

   pipeline:
     - common.NexusReader:
         filename: combined_hollow_pyramid.nxs
     - common.ImageProcessor:
         config:
           axis: x
           index_range: -1
           fileformat: jpeg
     - common.ImageWriter:
         filename: image.jpeg
         force_overwrite: true
   ```


1. Interactively plotting every 5th image slice in the `z` direction of the default plottable data object in the `combined_hollow_pyramid.nxs` file and writing the result as a stack of `tif` images to `images.tif`:
   ```bash
   config:
     interactive: true
     log_level: INFO

   pipeline:
     - common.NexusReader:
         filename: combined_hollow_pyramid.nxs
     - common.ImageProcessor:
         config:
           axis: z
           index_range: [None, None, 5]
     - common.ImageWriter:
         filename: images
         force_overwrite: true
   ```

1. Creating an animation of the image slices in the `z` direction of the default plottable data object in the `combined_hollow_pyramid.nxs` file from `z = 0` to `z = 2` at intervals of `0.2` and writing the result to `animation.gif`:
   ```bash
   config:
     interactive: false
     log_level: INFO

   pipeline:
     - common.NexusReader:
         filename: combined_hollow_pyramid.nxs
     - common.ImageProcessor:
         config:
           axis: z
           coord_range: [0, 2, 0.2]
     - common.ImageWriter:
         filename: animation
         force_overwrite: true
   ```

