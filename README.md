## Chioso: Segmentation-free Annotation of Spatial Transcriptomics (ST) at Sub-cellular Resolution

<img src="https://github.com/jiyuuchc/chioso/raw/main/.github/images/chioso_graph_abstract.png" width="800"> 

### Key Features of Chioso

- Pixel based annotation at subcellular-resolution
- Does NOT need cell segmentation input
- Scalable to very large dataset (e.g. [MOSTA](https://db.cngb.org/stomics/mosta/) dataset: 20 billion RNA reads, 1 billion locations, < 5 hours wall time)


### Installation

```
pip install git+https://github.com/jiyuuchc/chioso.git
```


### Usage

##### 0. Required Inputs
 1. ScRNAseq with cell type annotation in [h5ad](https://anndata.readthedocs.io) format
 2. Spatial data in space-deliminated text format with four feature columns: gene, x, y, counts. 
 3. Common genes in both datasets (or a subset genes of interests) as a list of string saved in a JSON file

#### 1. Convert input data to more efficient formats
```
python -m chioso.pp-ref --data <h5ad file> --genes <gene file> --outdir <outdir>

# repeat if more than one input file
python -m chioso.pp-spatial --data <st text file> --genes <gene file> --outdir <outdir> 
```

#### 2. Train predictive model based on the reference data
```
python -m chioso.train-predictor --config <cfg_predictor.py>
```
Default config files are under the [configs/](https://github.com/jiyuuchc/chioso/tree/main/configs)

#### 3. Train generative model on spatial data and reference data
```
python -m chioso.train-chioso --config <cfg_chioso.py>
```
Default config files are under the [configs/](https://github.com/jiyuuchc/chioso/tree/main/configs)

#### 4. Inference
```
python -m chioso.inference --config <cfg_chioso.py> --checkpoint <model checkpoint>
```
