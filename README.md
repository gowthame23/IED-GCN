# IED-GCN Landslide Susceptibility

This workspace contains the raster conditioning factors from the landslide susceptibility paper. The current code focuses on:

- loading and normalizing the raster factors
- building superpixel graphs for IED-GCN
- training and inference when a landslide label raster is provided
- generating a label-free susceptibility map when no inventory raster exists

The current archive does not include a landslide inventory / label raster. When you run the pipeline anyway, it will now compute an unsupervised susceptibility score from the raster factors and save:

- `susceptibility_score.tif`
- `susceptibility_class.tif`
