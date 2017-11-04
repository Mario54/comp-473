
### generate_data.py

Generates 1000 images for each font, and stores them in the `data` folder.

### feature_extraction.py

Takes all the generated images, and extracts data using Gabor filters. It stores all the data points in `data.json`.

### classify.py

Takes `data.json`, and classifies part of the data using KNN. Outputs the results of the testing accuracy.
