# TAG-MTL-DR (new pipeline, independent of the thesis)

Corrected, tested version of the TAG-MTL-DR prototype: EfficientNet-B3 lesion
tokens, supervised U-Net vessel branch, topological vessel graph (end-points and
junctions, segments as edges) encoded by a GAT, cross-attention from lesion tokens
to graph nodes, CORAL ordinal head, QWK / Sev-NR / PDR-NR evaluation and
patient-grouped folds.

```bash
pip install timm torch-geometric scikit-image scikit-learn pillow
python tag_mtl_dr.py --smoke      # checks the whole pipeline on random data
```

Training: build a `FundusDataset` from `(image_path, grade[, vessel_mask_path])`
items (without masks, classical pseudo-labels are used), a `DataLoader` with
`collate`, then call `train_epoch` and `evaluate`. Use `patient_folds` for
patient-level cross-validation. No result has been produced with this code yet.
